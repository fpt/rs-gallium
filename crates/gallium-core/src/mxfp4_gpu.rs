//! MXFP4 matrix-vector product on the device the weight lives on (issue #343
//! stage 4).
//!
//! candle has no MXFP4 `GgmlDType`, so a GPT-OSS expert could only reach the
//! GPU by being expanded to f32 (~33 MB per `[2880, 2880]` matrix, 7.5x its
//! bytes) — which is why `cpuMoe` keeps those experts on the CPU, and why they
//! could not enter the device-resident `ExpertCache`. This is the kernel that
//! lets them: the raw 4.4 MB of MXFP4 sits on the device as a `U8` tensor and
//! [`mxfp4_matvec`] decodes it in registers, never materialising f32 weights.
//!
//! The block format and the arithmetic are the CPU kernel's
//! (`kernels::dequant_dot_mxfp4`): 17-byte blocks of one E8M0 scale byte and
//! 16 bytes of E2M1 nibbles (byte `i`: low nibble element `i`, high nibble
//! element `i + 16`), the doubled `E2M1_LUT`, the halved `e8m0_to_f32` scale.
//! On CUDA it is a runtime-compiled (NVRTC) kernel, one warp per output row;
//! elsewhere the op falls back to the CPU kernel, so the same call works on any
//! device.

use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor};

use crate::kernels::{BaselineKernels, Kernels};

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 17;

/// `y[1, d_out] = W · x`, where `w` is one `[d_out, d_in]` MXFP4 matrix as raw
/// row-major bytes (`U8`, `d_out * d_in / 32 * 17` long) and `x` is `d_in` f32
/// values (`[d_in]` or `[1, d_in]`), both on the same device.
pub fn mxfp4_matvec(w: &Tensor, x: &Tensor, d_out: usize, d_in: usize) -> Result<Tensor> {
    if !d_in.is_multiple_of(BLOCK) {
        candle_core::bail!("mxfp4_matvec: d_in {d_in} is not a multiple of {BLOCK}");
    }
    let x = x
        .flatten_all()?
        .to_dtype(candle_core::DType::F32)?
        .contiguous()?;
    w.contiguous()?
        .apply_op2_no_bwd(&x, &Mxfp4Matvec { d_out, d_in })?
        .reshape((1, d_out))
}

struct Mxfp4Matvec {
    d_out: usize,
    d_in: usize,
}

impl Mxfp4Matvec {
    fn row_bytes(&self) -> usize {
        self.d_in / BLOCK * BLOCK_BYTES
    }

    fn check(&self, w_len: usize, x_len: usize) -> Result<()> {
        if w_len != self.d_out * self.row_bytes() || x_len != self.d_in {
            candle_core::bail!(
                "mxfp4_matvec: weight {w_len} bytes / x {x_len} values do not fit [{}, {}]",
                self.d_out,
                self.d_in
            );
        }
        Ok(())
    }
}

impl CustomOp2 for Mxfp4Matvec {
    fn name(&self) -> &'static str {
        "mxfp4-matvec"
    }

    fn cpu_fwd(
        &self,
        w: &CpuStorage,
        wl: &Layout,
        x: &CpuStorage,
        xl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let w = &w.as_slice::<u8>()?[wl.start_offset()..];
        let x = &x.as_slice::<f32>()?[xl.start_offset()..];
        let w = &w[..wl.shape().elem_count()];
        let x = &x[..xl.shape().elem_count()];
        self.check(w.len(), x.len())?;
        let rb = self.row_bytes();
        let y: Vec<f32> = (0..self.d_out)
            .map(|r| BaselineKernels.dequant_dot_mxfp4(&w[r * rb..][..rb], x))
            .collect();
        Ok((CpuStorage::F32(y), Shape::from(self.d_out)))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        w: &candle_core::CudaStorage,
        wl: &Layout,
        x: &candle_core::CudaStorage,
        xl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        self.check(wl.shape().elem_count(), xl.shape().elem_count())?;
        let dev = w.device().clone();
        let w_slice = w.as_cuda_slice::<u8>()?.slice(wl.start_offset()..);
        let x_slice = x.as_cuda_slice::<f32>()?.slice(xl.start_offset()..);
        let y = dev.alloc_zeros::<f32>(self.d_out)?;

        let func = dev.get_or_load_custom_func("mxfp4_matvec", "gallium_mxfp4", ptx()?)?;
        const WARPS_PER_BLOCK: u32 = 8;
        let cfg = LaunchConfig {
            grid_dim: ((self.d_out as u32).div_ceil(WARPS_PER_BLOCK), 1, 1),
            block_dim: (32 * WARPS_PER_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let d_out = self.d_out as i32;
        let blocks_per_row = (self.d_in / BLOCK) as i32;
        let mut b = func.builder();
        b.arg(&w_slice);
        b.arg(&x_slice);
        b.arg(&y);
        b.arg(&d_out);
        b.arg(&blocks_per_row);
        unsafe { b.launch(cfg) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(y, dev),
            Shape::from(self.d_out),
        ))
    }
}

/// The kernel's source: `E2M1_LUT` and `e8m0_to_f32` (`quantized.rs`) written
/// out in CUDA, so the GPU decodes exactly what the CPU kernel does.
#[cfg(feature = "cuda")]
const SOURCE: &str = r#"
// In shared memory, not `__constant__`: constant memory serves a warp at full
// speed only when every lane reads the same entry, and here each lane decodes a
// different nibble — the reads serialized up to 16 ways.
__constant__ float E2M1_INIT[16] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f,
                                    0.f, -1.f, -2.f, -3.f, -4.f, -6.f, -8.f, -12.f};

__device__ __forceinline__ float e8m0_half(unsigned int b) {
    return b < 2 ? __uint_as_float(0x00200000u << b) : __uint_as_float((b - 1u) << 23);
}

// One warp per output row, two blocks per step: lanes 0-15 take block b and
// lanes 16-31 block b + 1, each lane one byte of its block — both nibbles —
// so the warp's loads are two contiguous 16-byte runs rather than 32 lanes
// each walking its own 17-byte stride (that layout measured 54 GB/s).
extern "C" __global__ void mxfp4_matvec(const unsigned char* __restrict__ w,
                                        const float* __restrict__ x,
                                        float* __restrict__ y,
                                        int d_out, int blocks_per_row) {
    __shared__ float E2M1_LUT[16];
    if (threadIdx.x < 16) E2M1_LUT[threadIdx.x] = E2M1_INIT[threadIdx.x];
    __syncthreads();
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= d_out) return;
    const unsigned char* r = w + (size_t)row * blocks_per_row * 17;
    const int half = lane >> 4;
    const int j = lane & 15;
    float acc = 0.f;
    #pragma unroll 4
    for (int b = half; b < blocks_per_row; b += 2) {
        const unsigned char* blk = r + b * 17;
        const unsigned int q = __ldg(blk + 1 + j);
        const float* xb = x + b * 32;
        const float d = E2M1_LUT[q & 15u] * __ldg(xb + j) + E2M1_LUT[q >> 4] * __ldg(xb + j + 16);
        acc += e8m0_half(__ldg(blk)) * d;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = acc;
}
"#;

/// The PTX, compiled once per process.
#[cfg(feature = "cuda")]
fn ptx() -> Result<&'static str> {
    use candle_core::cuda_backend::cudarc::nvrtc;
    static PTX: std::sync::OnceLock<std::result::Result<String, String>> =
        std::sync::OnceLock::new();
    PTX.get_or_init(|| {
        nvrtc::compile_ptx(SOURCE)
            .map(|p| p.to_src())
            .map_err(|e| format!("{e:?}"))
    })
    .as_deref()
    .map_err(|e| candle_core::Error::Msg(format!("mxfp4_matvec: NVRTC: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Deterministic MXFP4 bytes with realistic scales (E8M0 bytes near 127,
    /// i.e. scales near 1) and every nibble value represented.
    pub(crate) fn sample(d_out: usize, d_in: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
        let mut s = seed.wrapping_mul(2654435761).max(1);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        let blocks = d_out * d_in / BLOCK;
        let mut w = Vec::with_capacity(blocks * BLOCK_BYTES);
        for _ in 0..blocks {
            w.push(120 + (next() % 12) as u8);
            for _ in 0..16 {
                w.push(next() as u8);
            }
        }
        let x = (0..d_in)
            .map(|_| (next() % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        (w, x)
    }

    /// The op's CPU forward is the CPU kernel, row by row.
    #[test]
    fn cpu_matches_the_cpu_kernel() {
        let (d_out, d_in) = (64, 256);
        let (w, x) = sample(d_out, d_in, 7);
        let wt = Tensor::from_vec(w.clone(), w.len(), &Device::Cpu).unwrap();
        let xt = Tensor::from_vec(x.clone(), d_in, &Device::Cpu).unwrap();
        let y: Vec<f32> = mxfp4_matvec(&wt, &xt, d_out, d_in)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1()
            .unwrap();
        let rb = d_in / BLOCK * BLOCK_BYTES;
        for r in 0..d_out {
            assert_eq!(
                y[r],
                BaselineKernels.dequant_dot_mxfp4(&w[r * rb..][..rb], &x)
            );
        }
        assert_eq!(wt.dtype(), DType::U8);
    }

    /// The CUDA kernel decodes what the CPU kernel decodes. Not bit-identical —
    /// a warp reduces in a different order — so held to a relative tolerance on
    /// GPT-OSS's own shape. `cargo test -p gallium-core --features cuda --
    /// --ignored mxfp4_cuda`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs a CUDA device"]
    fn mxfp4_cuda_matches_cpu() {
        let dev = Device::new_cuda(0).unwrap();
        for &(d_out, d_in) in &[(2880usize, 2880usize), (5760, 2880), (37, 96)] {
            let (w, x) = sample(d_out, d_in, d_out as u32);
            let cpu = |t: Vec<u8>| Tensor::from_vec(t, w.len(), &Device::Cpu).unwrap();
            let xc = Tensor::from_vec(x.clone(), d_in, &Device::Cpu).unwrap();
            let want: Vec<f32> = mxfp4_matvec(&cpu(w.clone()), &xc, d_out, d_in)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_vec1()
                .unwrap();
            let got: Vec<f32> = mxfp4_matvec(
                &cpu(w.clone()).to_device(&dev).unwrap(),
                &xc.to_device(&dev).unwrap(),
                d_out,
                d_in,
            )
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1()
            .unwrap();
            let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let worst = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst <= 1e-4 * scale.max(1.0),
                "[{d_out}, {d_in}]: max |Δ| {worst} at scale {scale}"
            );
        }
    }
}

/// Kernel throughput on GPT-OSS's gate/up shape, against the bytes it has to
/// read. `cargo test -p gallium-core --features cuda --release -- --ignored
/// mxfp4_cuda_throughput --nocapture`.
#[cfg(all(test, feature = "cuda"))]
mod bench {
    use super::*;
    use candle_core::Device;

    #[test]
    #[ignore = "needs a CUDA device"]
    fn mxfp4_cuda_throughput() {
        let dev = Device::new_cuda(0).unwrap();
        let (d_out, d_in) = (5760usize, 2880usize);
        let (w, x) = tests::sample(d_out, d_in, 3);
        let bytes = w.len();
        let w = Tensor::from_vec(w, bytes, &dev).unwrap();
        let x = Tensor::from_vec(x, d_in, &dev).unwrap();
        for _ in 0..10 {
            mxfp4_matvec(&w, &x, d_out, d_in).unwrap();
        }
        dev.synchronize().unwrap();
        let n = 200;
        let t = std::time::Instant::now();
        for _ in 0..n {
            mxfp4_matvec(&w, &x, d_out, d_in).unwrap();
        }
        dev.synchronize().unwrap();
        let per = t.elapsed().as_secs_f64() / n as f64;
        println!(
            "[{d_out}, {d_in}] MXFP4 matvec: {:.1} µs, {:.0} GB/s of weight",
            per * 1e6,
            bytes as f64 / per / 1e9
        );
    }
}
