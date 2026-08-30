//! Quantized layer support for GGUF model loading.
//!
//! Provides `QVarBuilder` for loading GGUF files, and `QLinear` / `QNorm` as
//! drop-in replacements for `Linear` / `Norm` that work with quantized weights.

use candle_core::quantized::{gguf_file, GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{Device, Module, Result, Tensor};
use memmap2::Mmap;
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{Read, Seek};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// QVarBuilder: navigate GGUF tensors with dot-separated prefixes (like VarBuilder)
// ---------------------------------------------------------------------------

/// Shared mmap for a single GGUF file. Held by Arc so all tensors from the
/// same file keep the mapping alive without copying anything.
struct MmapSource {
    mmap: Arc<Mmap>,
    /// Absolute byte offset of the tensor-data section within the file.
    base: u64,
}

/// A GGUF tensor that is not materialized into heap memory until first access.
///
/// `Lazy` variant: on the first `get()` call the bytes are read from the mmap
/// (demand-paged by the OS), copied into a `QStorage`, and cached in `cell`.
/// All subsequent calls return the cached `Arc<QTensor>` with no I/O.
///
/// `Eager` variant: used by the legacy `from_gguf_content` path that loads
/// into heap up-front; kept for compatibility.
enum LazyQTensor {
    Lazy {
        source: Arc<MmapSource>,
        /// Byte offset of this tensor relative to `source.base`.
        offset: u64,
        /// Byte length of this tensor's raw quantized data.
        size: usize,
        dtype: GgmlDType,
        shape: candle_core::Shape,
        device: Device,
        /// Cached result; None until first `get()` call.
        cell: Mutex<Option<Arc<QTensor>>>,
    },
}

impl LazyQTensor {
    /// View this (merged, N-D) tensor as a stack of experts along dim 0, for
    /// per-expert lazy dequantization. Works for any GGML block quant (Q4_K,
    /// etc.) — the generic counterpart to `Tq2Tensor` (which is MXFP4-only).
    fn as_experts(&self) -> QExperts {
        match self {
            LazyQTensor::Lazy {
                source,
                offset,
                dtype,
                shape,
                ..
            } => QExperts {
                source: source.clone(),
                offset: *offset,
                dtype: *dtype,
                dims: shape.dims().to_vec(),
            },
        }
    }

    fn get(&self) -> Result<Arc<QTensor>> {
        match self {
            LazyQTensor::Lazy {
                source,
                offset,
                size,
                dtype,
                shape,
                device,
                cell,
            } => {
                let mut guard = cell.lock().unwrap();
                if let Some(qt) = guard.as_ref() {
                    return Ok(qt.clone());
                }
                let start = (source.base + offset) as usize;
                // Safety: from_data copies the bytes into QStorage before returning,
                // so the mmap slice only needs to live for the duration of this call.
                let raw = &source.mmap[start..start + size];
                let storage = QStorage::from_data(Cow::Borrowed(raw), device, *dtype)?;
                let qt = Arc::new(QTensor::new(storage, shape.clone())?);
                *guard = Some(qt.clone());
                Ok(qt)
            }
        }
    }

    /// Materialize on `device` rather than the builder's own, and **do not
    /// cache**. For a tensor that must live somewhere other than where the
    /// model runs — see [`QVarBuilder::get_on`].
    fn get_on(&self, device: &Device) -> Result<Arc<QTensor>> {
        match self {
            LazyQTensor::Lazy {
                source,
                offset,
                size,
                dtype,
                shape,
                ..
            } => {
                let start = (source.base + offset) as usize;
                let raw = &source.mmap[start..start + size];
                let storage = QStorage::from_data(Cow::Borrowed(raw), device, *dtype)?;
                Ok(Arc::new(QTensor::new(storage, shape.clone())?))
            }
        }
    }
}

/// An MXFP4 tensor whose bytes live in a file mmap.
/// Dequantized one expert at a time during forward pass — no heap copy at load time.
/// Dims are row-major (outer dimension first), e.g. `[n_expert, n_ff, n_embd]`.
#[derive(Clone)]
pub struct Tq2Tensor {
    source: Arc<MmapSource>,
    /// Byte offset of this tensor's data relative to `source.base`.
    offset: u64,
    pub dims: Vec<usize>,
}

impl Tq2Tensor {
    /// Dequantize the slice for expert `idx` into a float Tensor with shape `dims[1..]`.
    pub fn dequantize_expert(&self, idx: usize, device: &Device) -> Result<Tensor> {
        let n_elems_per_expert: usize = self.dims[1..].iter().product();
        let n_blocks = mxfp4_blocks_per_expert(n_elems_per_expert)?;
        let bytes_per_expert = n_blocks * MXFP4_BYTES_PER_BLOCK;
        let start = (self.source.base + self.offset) as usize + idx * bytes_per_expert;
        let raw = &self.source.mmap[start..start + bytes_per_expert];
        // Pre-warm: issue T2 (L3) prefetch hints for the first cache lines of this
        // expert's mmap region.  Cold mmap pages take 200-300 cycles from DRAM; firing
        // these hints before the dequant loop starts hides most of that latency.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use core::arch::x86_64::*;
            let n_lines = (raw.len() / 64).min(32);
            for i in 0..n_lines {
                _mm_prefetch(raw.as_ptr().add(i * 64) as *const i8, _MM_HINT_T2);
            }
        }
        let floats = dequantize_mxfp4(raw, n_elems_per_expert);
        Tensor::from_vec(floats, self.dims[1..].to_vec().as_slice(), device)
    }
}

/// A merged, N-D block-quantized tensor (e.g. GGUF MoE expert weights
/// `[n_expert, d_out, d_in]`) whose bytes live in the file mmap, dequantized one
/// expert at a time during the forward pass. Unlike [`Tq2Tensor`] (MXFP4-only),
/// this works for any GGML block quant (Q4_K, Q6_K, …) by slicing each expert's
/// byte range and going through candle's dequantizer.
#[derive(Clone)]
pub struct QExperts {
    source: Arc<MmapSource>,
    /// Byte offset of the merged tensor's data relative to `source.base`.
    offset: u64,
    dtype: GgmlDType,
    /// Row-major dims, outer (expert) dimension first: `[n_expert, d_out, d_in]`.
    dims: Vec<usize>,
}

impl QExperts {
    /// Number of experts (the leading dimension).
    pub fn n_experts(&self) -> usize {
        self.dims[0]
    }

    /// Per-expert shape (`dims[1..]`), e.g. `[d_out, d_in]`.
    pub fn expert_shape(&self) -> &[usize] {
        &self.dims[1..]
    }

    /// Dequantize expert `idx` into a float `Tensor` of shape `dims[1..]`. Each
    /// expert's elements are block-aligned (the merged tensor's per-expert element
    /// count is a multiple of the block size), so the byte range is contiguous.
    pub fn dequantize_expert(&self, idx: usize, device: &Device) -> Result<Tensor> {
        self.qtensor_expert(idx, device)?.dequantize(device)
    }

    /// Expert `idx` as a [`QTensor`], still quantized.
    ///
    /// This is the step [`Self::dequantize_expert`] takes before expanding, and
    /// on its own it is the expensive half: `QStorage::from_data` **copies the
    /// quantized bytes to the device**, which on Metal means allocating a buffer
    /// and uploading, every time it is called. Doing that per token per expert
    /// is what [`Self::qmatmuls`] exists to stop.
    pub fn qtensor_expert(&self, idx: usize, device: &Device) -> Result<QTensor> {
        let per_expert_elems: usize = self.dims[1..].iter().product();
        let block = self.dtype.block_size();
        let type_size = self.dtype.type_size();
        if per_expert_elems % block != 0 {
            candle_core::bail!(
                "expert elem count {per_expert_elems} not divisible by block size {block}"
            );
        }
        let bytes_per_expert = per_expert_elems / block * type_size;
        let start = (self.source.base + self.offset) as usize + idx * bytes_per_expert;
        let raw = &self.source.mmap[start..start + bytes_per_expert];
        let storage = QStorage::from_data(Cow::Borrowed(raw), device, self.dtype)?;
        QTensor::new(storage, candle_core::Shape::from(self.dims[1..].to_vec()))
    }

    /// One [`QMatMul`] per expert, built **once**, for a caller that would
    /// otherwise dequantize inside its forward pass.
    ///
    /// This is the fix for the shape a Metal memory graph shows as a sawtooth.
    /// Dequantizing per token allocates the expanded weight *and* re-uploads the
    /// quantized bytes, for every active expert of every layer: measured on
    /// LFM2.5-8B-A1B, about **4 GB allocated and freed per token**, with resident
    /// memory cycling 8 GB → 12 GB → 8 GB while decoding at ~1 tok/s.
    ///
    /// `QMatMul` does the multiply against the quantized weight instead, which
    /// `docs/CANDLE_BACKEND.md` had already measured as the fast path: a quantized
    /// matvec is 0.71 ms per projection on Metal, while the dequantizing variant
    /// (`forward_via_f16`) is 64 ms — ~95× worse. It also removes the transpose
    /// and dtype conversion a caller needs around a dequantized weight, since
    /// `QMatMul::forward` computes `x · Wᵀ` itself.
    ///
    /// The cost moved, not removed: every expert's bytes are uploaded to the
    /// device at load rather than on demand, so the whole tensor is resident.
    /// That is the same memory the model file already occupies and the trade the
    /// GPU wants — it is paid once instead of once per token.
    pub fn qmatmuls(&self, device: &Device) -> Result<Vec<QMatMul>> {
        (0..self.dims[0])
            .map(|i| QMatMul::from_qtensor(self.qtensor_expert(i, device)?))
            .collect()
    }
}

#[derive(Clone)]
pub struct QVarBuilder {
    /// Lazy-materialized quantized tensors. Arc lets pp() clones share the same map.
    data: Arc<HashMap<String, LazyQTensor>>,
    /// MXFP4 expert-weight tensors for per-expert lazy dequantization.
    tq2_raw: Arc<HashMap<String, Tq2Tensor>>,
    path: Vec<String>,
    device: Device,
}

impl QVarBuilder {
    /// Push a prefix, like VarBuilder::pp(). Returns a new builder scoped to "parent.child".
    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            tq2_raw: self.tq2_raw.clone(),
            path,
            device: self.device.clone(),
        }
    }

    /// View a merged block-quantized expert tensor (any GGML quant, e.g. Q4_K)
    /// for per-expert lazy dequantization. The generic counterpart to
    /// [`get_tq2`](Self::get_tq2), which handles only MXFP4.
    pub fn get_experts(&self, name: &str) -> Result<QExperts> {
        let path = self.full_path(name);
        self.data
            .get(&path)
            .map(|t| t.as_experts())
            .ok_or_else(|| candle_core::Error::Msg(format!("cannot find tensor: {path}")))
    }

    /// Get the mmap-backed MXFP4 tensor for per-expert lazy dequantization.
    pub fn get_tq2(&self, name: &str) -> Result<Tq2Tensor> {
        let path = self.full_path(name);
        self.tq2_raw
            .get(&path)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("no MXFP4 tensor: {path}")))
    }

    /// Full dot-joined path for a tensor name.
    fn full_path(&self, name: &str) -> String {
        if self.path.is_empty() {
            name.to_string()
        } else {
            format!("{}.{name}", self.path.join("."))
        }
    }

    /// Materialize and return the quantized tensor for `name`.
    /// On the first call for a given tensor, copies bytes from the mmap into
    /// `QStorage` and caches the result. Subsequent calls are cache hits.
    pub fn get(&self, name: &str) -> Result<Arc<QTensor>> {
        let path = self.full_path(name);
        self.data
            .get(&path)
            .ok_or_else(|| candle_core::Error::Msg(format!("cannot find tensor: {path}")))?
            .get()
    }

    /// Like [`get`](Self::get) but materializes on `device` instead of the
    /// builder's own, and does not cache the result.
    ///
    /// For the odd tensor that must not land on the compute device: Gemma 4
    /// E4B's `per_layer_token_embd.weight` is `[vocab, n_layers·ple_dim]`, ~11 GB
    /// once dequantized to f32, which OOMs a 12 GB card on its own — while a
    /// forward only ever needs the handful of rows for the current tokens.
    /// Keeping it in host memory and gathering per token is the fix; this is how
    /// the model loader asks for it there. Not cached, because the shared cell
    /// would then hand this off-device copy to the next ordinary `get()`.
    pub fn get_on(&self, name: &str, device: &Device) -> Result<Arc<QTensor>> {
        let path = self.full_path(name);
        self.data
            .get(&path)
            .ok_or_else(|| candle_core::Error::Msg(format!("cannot find tensor: {path}")))?
            .get_on(device)
    }

    /// Check if a tensor exists (without materializing it).
    pub fn contains(&self, name: &str) -> bool {
        let path = self.full_path(name);
        self.data.contains_key(&path)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// List all tensor names (useful for debugging).
    pub fn tensor_names(&self) -> Vec<&str> {
        self.data.keys().map(|s| s.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// GGUF metadata reader (for extracting config from GGUF header)
// ---------------------------------------------------------------------------

/// Open a GGUF file with mmap and return a lazy `QVarBuilder`.
///
/// The file is memory-mapped once; no tensor bytes are read from disk at this
/// point.  Each tensor is materialized (bytes copied from the mmap into a
/// `QStorage`) on the first `QVarBuilder::get()` call for that tensor and
/// cached thereafter.  MXFP4 expert tensors (`Tq2Tensor`) are never pre-copied;
/// `dequantize_expert` reads directly from the mmap slice at forward time.
///
/// Benefits over the previous eager-load approach:
/// - Model "load" is near-instant (just an mmap syscall + header parse).
/// - Only the tensor pages that are actually touched land in physical RAM; the
///   OS can evict cold pages under memory pressure.
/// - Peak RSS is bounded by the working set rather than the full file size.
pub fn load_gguf<P: AsRef<std::path::Path>>(
    path: P,
    device: &Device,
) -> Result<(GgufMetadata, QVarBuilder)> {
    let file = std::fs::File::open(path.as_ref())?;

    // mmap the file so all tensor data is addressable without explicit reads.
    // Safety: we never write through this mapping and hold it for the lifetime
    // of the QVarBuilder via Arc<MmapSource>.
    let mmap = unsafe { Mmap::map(&file)? };
    let mmap = Arc::new(mmap);

    // Parse header (metadata KVs + tensor infos) from a cursor into the mmap.
    // This avoids a second open() and any seeks on the original File handle.
    let (metadata_map, tensor_infos, tensor_data_offset) = {
        let mut cursor = std::io::Cursor::new(mmap.as_ref());
        parse_gguf_tolerant(&mut cursor)?
    };

    let source = Arc::new(MmapSource {
        mmap,
        base: tensor_data_offset,
    });

    let mut lazy_tensors: HashMap<String, LazyQTensor> = HashMap::new();
    let mut tq2_map: HashMap<String, Tq2Tensor> = HashMap::new();

    for (name, info) in &tensor_infos {
        let n_elems: usize = info.dims.iter().product();

        if info.dtype_u32 == MXFP4_TYPE {
            // MXFP4: no pre-copy; dequantize_expert slices the mmap on demand.
            let n_blocks = n_elems / MXFP4_BLOCK_SIZE;
            let _size = n_blocks * MXFP4_BYTES_PER_BLOCK; // kept for future bounds checking
            tq2_map.insert(
                name.clone(),
                Tq2Tensor {
                    source: source.clone(),
                    offset: info.offset,
                    dims: info.dims.clone(),
                },
            );
        } else {
            let dtype = ggml_dtype_from_u32(info.dtype_u32)?;
            let block_size = dtype.block_size();
            let type_size = dtype.type_size();
            if n_elems % block_size != 0 {
                candle_core::bail!(
                    "tensor {name}: elem count {n_elems} not divisible by block size {block_size}"
                );
            }
            let size = n_elems / block_size * type_size;
            let shape = candle_core::Shape::from(info.dims.clone());
            lazy_tensors.insert(
                name.clone(),
                LazyQTensor::Lazy {
                    source: source.clone(),
                    offset: info.offset,
                    size,
                    dtype,
                    shape,
                    device: device.clone(),
                    cell: Mutex::new(None),
                },
            );
        }
    }

    let vb = QVarBuilder {
        data: Arc::new(lazy_tensors),
        tq2_raw: Arc::new(tq2_map),
        path: Vec::new(),
        device: device.clone(),
    };
    let metadata = GgufMetadata {
        metadata: metadata_map,
    };
    Ok((metadata, vb))
}

// ─── MXFP4 (OCP MX Float4 E2M1) constants ───────────────────────────────────
//
// Type 39 in GGUF. Used by GPT-OSS for MoE expert weight matrices.
// Ref: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

const MXFP4_TYPE: u32 = 39;
const MXFP4_BLOCK_SIZE: usize = 32;
const MXFP4_BYTES_PER_BLOCK: usize = 17; // 1 byte E8M0 scale + 16 bytes (32 nibbles)

/// E2M1 FP4 dequant lookup table (multiplied by 2 relative to true FP4 values).
/// Index is the 4-bit code; value × scale gives the dequantized float.
/// Matches gguf Python library: (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)
const E2M1_LUT: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Convert an E8M0 exponent byte to f32 scale.
///
/// For byte >= 2: scale = f32 with exponent bits = (byte-1), mantissa = 0
///   → scale = 2^(byte - 128)
/// For byte < 2: tiny denormal-like value (essentially 0 scale).
fn e8m0_to_f32(byte: u8) -> f32 {
    if byte < 2 {
        // Very small denormal: 2^(-126 - (1 - byte)) ≈ 0
        f32::from_bits(0x0020_0000u32 << (byte as u32))
    } else {
        // Normal: set exponent bits = byte - 1, mantissa = 0
        f32::from_bits((byte as u32 - 1) << 23)
    }
}

/// Dequantize MXFP4 raw bytes → f32.
///
/// Block layout (17 bytes / 32 elements):
///   [0]      scale: E8M0 exponent byte
///   [1..16]  qs: 32 × E2M1 nibbles, lower nibble of byte[i] → element[i],
///                upper nibble of byte[i] → element[i + 16]
///
/// Dequant: value[i] = e8m0_to_f32(scale) * E2M1_LUT[nibble]
fn dequantize_mxfp4(raw: &[u8], n_elems: usize) -> Vec<f32> {
    // Zeroed rather than `with_capacity` + `set_len`. The old comment claimed
    // every element is written before use, which holds only while `n_elems` is
    // a whole number of blocks: the dequant loops cover `len / 32 * 32`
    // elements and leave any remainder untouched, so a ragged tensor handed
    // uninitialized floats to `Tensor::from_vec`. `vec![0.0; n]` costs nothing
    // measurable here — a zeroed f32 allocation this size is `alloc_zeroed`,
    // which the allocator serves from fresh zero pages.
    //
    // `mxfp4_blocks_per_expert` is what actually refuses a ragged tensor; this
    // is the private function's own precondition, for a caller added later.
    debug_assert_eq!(n_elems % MXFP4_BLOCK_SIZE, 0);
    let mut out = vec![0.0; n_elems];
    dequantize_mxfp4_into(raw, &mut out);
    out
}

/// Blocks per expert, refusing a tensor that is not a whole number of them.
///
/// The count is load-bearing twice over: it sets the expert's byte stride, and
/// it bounds the dequant loops. A remainder is therefore not a rounding
/// question — it shifts the read offset of every expert after the first, and
/// leaves a zero tail in each — so it is an error, the same way
/// [`QExperts::dequantize_expert`] refuses it for the generic block quants.
fn mxfp4_blocks_per_expert(n_elems: usize) -> Result<usize> {
    if n_elems % MXFP4_BLOCK_SIZE != 0 {
        candle_core::bail!(
            "MXFP4 expert of {n_elems} elements is not a whole number of \
             {MXFP4_BLOCK_SIZE}-element blocks"
        );
    }
    Ok(n_elems / MXFP4_BLOCK_SIZE)
}

/// Write-into variant: dequantizes into a caller-owned slice, avoiding allocation.
/// `out` must have length == n_elems for this tensor.
fn dequantize_mxfp4_into(raw: &[u8], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { dequantize_mxfp4_avx2(raw, out) };
    }
    dequantize_mxfp4_scalar(raw, out);
}

fn dequantize_mxfp4_scalar(raw: &[u8], out: &mut [f32]) {
    let n_blocks = out.len() / MXFP4_BLOCK_SIZE;
    for blk in 0..n_blocks {
        let base = blk * MXFP4_BYTES_PER_BLOCK;
        let scale = e8m0_to_f32(raw[base]);
        let out_base = blk * MXFP4_BLOCK_SIZE;
        for j in 0..16usize {
            let byte = raw[base + 1 + j];
            out[out_base + j] = E2M1_LUT[(byte & 0xF) as usize] as f32 * scale;
            out[out_base + j + 16] = E2M1_LUT[(byte >> 4) as usize] as f32 * scale;
        }
    }
}

/// AVX2 fast path: processes one block (32 elements, 17 bytes) per iteration.
///
/// Per block:
///   1. Load 16 nibble bytes.
///   2. Unpack into low/high nibble vectors (elements 0-15 and 16-31).
///   3. Resolve i8 values via `pshufb` (16-entry in-register LUT).
///   4. Widen i8 → i32 → f32 in groups of 8 and multiply by scale.
///   5. Four `storeu_ps` writes cover all 32 output elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_mxfp4_avx2(raw: &[u8], out: &mut [f32]) {
    use core::arch::x86_64::*;

    // pshufb LUT: nibble index → E2M1 i8 value.
    // _mm_set_epi8(e15,…,e0): e0 lives at byte-lane 0, e15 at lane 15.
    // E2M1_LUT = [0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12]
    let lut = _mm_set_epi8(-12, -8, -6, -4, -3, -2, -1, 0, 12, 8, 6, 4, 3, 2, 1, 0_i8);
    let nibble_mask = _mm_set1_epi8(0x0F_u8 as i8);

    // T0 prefetch distance: 16 blocks = 272 bytes ≈ 4 cache lines ahead.
    // At ~5 cycles/iteration this gives ~80 cycles lead time — enough for L3 hits.
    // Combined with the T2 pre-warm in dequantize_expert this covers DRAM latency too.
    const PREFETCH_DIST: usize = 16;
    let n_blocks = out.len() / MXFP4_BLOCK_SIZE;
    for blk in 0..n_blocks {
        let rb = blk * MXFP4_BYTES_PER_BLOCK;
        let ob = blk * MXFP4_BLOCK_SIZE;

        if blk + PREFETCH_DIST < n_blocks {
            _mm_prefetch(
                raw.as_ptr()
                    .add((blk + PREFETCH_DIST) * MXFP4_BYTES_PER_BLOCK)
                    as *const i8,
                _MM_HINT_T0,
            );
        }

        let scale = e8m0_to_f32(raw[rb]);
        let sv = _mm256_set1_ps(scale);

        // Load the 16 packed-nibble bytes for this block.
        let qs = _mm_loadu_si128(raw.as_ptr().add(rb + 1) as *const __m128i);

        // lo = bits[3:0] of each byte  → E2M1 values for elements  0..15
        // hi = bits[7:4] of each byte  → E2M1 values for elements 16..31
        let lo = _mm_and_si128(qs, nibble_mask);
        let hi = _mm_and_si128(_mm_srli_epi16(qs, 4), nibble_mask);

        // pshufb: 16-entry in-register LUT, nibble → i8.
        let lo_i8 = _mm_shuffle_epi8(lut, lo);
        let hi_i8 = _mm_shuffle_epi8(lut, hi);

        // Convert 8 bytes of i8 → 8×i32 → 8×f32, multiply by scale.
        // _mm256_cvtepi8_epi32 reads the 8 lowest bytes of its __m128i arg.
        // Shift by 8 bytes to expose the upper half.
        macro_rules! to_f32x8 {
            ($v:expr) => {
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32($v)), sv)
            };
        }

        let out_ptr = out.as_mut_ptr().add(ob);
        _mm256_storeu_ps(out_ptr, to_f32x8!(lo_i8));
        _mm256_storeu_ps(out_ptr.add(8), to_f32x8!(_mm_srli_si128::<8>(lo_i8)));
        _mm256_storeu_ps(out_ptr.add(16), to_f32x8!(hi_i8));
        _mm256_storeu_ps(out_ptr.add(24), to_f32x8!(_mm_srli_si128::<8>(hi_i8)));
    }
}

// ─── Minimal GGUF parser (tolerates unknown tensor dtypes) ───────────────────

#[derive(Clone, Copy)]
enum GgufVersion {
    V1,
    V2V3,
}

struct RawTensorInfo {
    dims: Vec<usize>, // already reversed to row-major
    dtype_u32: u32,
    offset: u64,
}

fn parse_gguf_tolerant<R: Read + Seek>(
    r: &mut R,
) -> Result<(
    HashMap<String, gguf_file::Value>,
    HashMap<String, RawTensorInfo>,
    u64,
)> {
    // Magic
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    let magic_u32 = u32::from_le_bytes(magic);
    match magic_u32 {
        0x46554747 | 0x47475546 => {}
        _ => candle_core::bail!("unknown GGUF magic 0x{magic_u32:08x}"),
    }
    // Version
    let mut ver_bytes = [0u8; 4];
    r.read_exact(&mut ver_bytes)?;
    let ver = match u32::from_le_bytes(ver_bytes) {
        1 => GgufVersion::V1,
        2 | 3 => GgufVersion::V2V3,
        v => candle_core::bail!("unknown GGUF version {v}"),
    };

    // Counts
    let (tensor_count, kv_count) = match ver {
        GgufVersion::V1 => {
            let tc = gguf_read_u32(r)? as usize;
            let kc = gguf_read_u32(r)? as usize;
            (tc, kc)
        }
        GgufVersion::V2V3 => {
            let tc = gguf_read_u64(r)? as usize;
            let kc = gguf_read_u64(r)? as usize;
            (tc, kc)
        }
    };

    // Metadata KVs
    let mut metadata = HashMap::new();
    for _ in 0..kv_count {
        let key = gguf_read_string(r, ver)?;
        let vtype = gguf_read_u32(r)?;
        let value = gguf_read_value(r, vtype, ver)?;
        metadata.insert(key, value);
    }

    // Tensor infos (tolerating unknown dtypes)
    let mut tensor_infos: HashMap<String, RawTensorInfo> = HashMap::new();
    for _ in 0..tensor_count {
        let name = gguf_read_string(r, ver)?;
        let n_dims = gguf_read_u32(r)? as usize;
        let mut dims: Vec<usize> = match ver {
            GgufVersion::V1 => (0..n_dims)
                .map(|_| gguf_read_u32(r).map(|v| v as usize))
                .collect::<Result<_>>()?,
            GgufVersion::V2V3 => (0..n_dims)
                .map(|_| gguf_read_u64(r).map(|v| v as usize))
                .collect::<Result<_>>()?,
        };
        dims.reverse();
        let dtype_u32 = gguf_read_u32(r)?;
        let offset = gguf_read_u64(r)?;
        tensor_infos.insert(
            name,
            RawTensorInfo {
                dims,
                dtype_u32,
                offset,
            },
        );
    }

    // Tensor data offset (aligned)
    let pos = r.stream_position()?;
    let alignment: u64 = match metadata.get("general.alignment") {
        Some(gguf_file::Value::U32(v)) => *v as u64,
        Some(gguf_file::Value::U8(v)) => *v as u64,
        Some(gguf_file::Value::U16(v)) => *v as u64,
        _ => 32,
    };
    let tensor_data_offset = pos.div_ceil(alignment) * alignment;
    Ok((metadata, tensor_infos, tensor_data_offset))
}

/// Map a GGUF dtype u32 to `GgmlDType`. Mirrors candle's private `from_u32`.
fn ggml_dtype_from_u32(u: u32) -> Result<GgmlDType> {
    match u {
        0 => Ok(GgmlDType::F32),
        1 => Ok(GgmlDType::F16),
        2 => Ok(GgmlDType::Q4_0),
        3 => Ok(GgmlDType::Q4_1),
        6 => Ok(GgmlDType::Q5_0),
        7 => Ok(GgmlDType::Q5_1),
        8 => Ok(GgmlDType::Q8_0),
        9 => Ok(GgmlDType::Q8_1),
        10 => Ok(GgmlDType::Q2K),
        11 => Ok(GgmlDType::Q3K),
        12 => Ok(GgmlDType::Q4K),
        13 => Ok(GgmlDType::Q5K),
        14 => Ok(GgmlDType::Q6K),
        15 => Ok(GgmlDType::Q8K),
        30 => Ok(GgmlDType::BF16),
        v => candle_core::bail!("unknown GgmlDType {v}"),
    }
}

// Low-level GGUF readers using plain std::io::Read

fn gguf_read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn gguf_read_u16<R: Read>(r: &mut R) -> Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn gguf_read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn gguf_read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn gguf_read_f32<R: Read>(r: &mut R) -> Result<f32> {
    Ok(f32::from_bits(gguf_read_u32(r)?))
}
fn gguf_read_f64<R: Read>(r: &mut R) -> Result<f64> {
    Ok(f64::from_bits(gguf_read_u64(r)?))
}
fn gguf_read_string<R: Read>(r: &mut R, ver: GgufVersion) -> Result<String> {
    let len = match ver {
        GgufVersion::V1 => gguf_read_u32(r)? as usize,
        GgufVersion::V2V3 => gguf_read_u64(r)? as usize,
    };
    let mut v = vec![0u8; len];
    r.read_exact(&mut v)?;
    while let Some(0) = v.last() {
        v.pop();
    }
    Ok(String::from_utf8_lossy(&v).into_owned())
}

fn gguf_read_value<R: Read>(r: &mut R, vtype: u32, ver: GgufVersion) -> Result<gguf_file::Value> {
    match vtype {
        0 => Ok(gguf_file::Value::U8(gguf_read_u8(r)?)),
        1 => Ok(gguf_file::Value::I8(gguf_read_u8(r)? as i8)),
        2 => Ok(gguf_file::Value::U16(gguf_read_u16(r)?)),
        3 => Ok(gguf_file::Value::I16(gguf_read_u16(r)? as i16)),
        4 => Ok(gguf_file::Value::U32(gguf_read_u32(r)?)),
        5 => Ok(gguf_file::Value::I32(gguf_read_u32(r)? as i32)),
        6 => Ok(gguf_file::Value::F32(gguf_read_f32(r)?)),
        7 => Ok(gguf_file::Value::Bool(gguf_read_u8(r)? != 0)),
        8 => Ok(gguf_file::Value::String(gguf_read_string(r, ver)?)),
        9 => {
            let elem_type = gguf_read_u32(r)?;
            let len = match ver {
                GgufVersion::V1 => gguf_read_u32(r)? as usize,
                GgufVersion::V2V3 => gguf_read_u64(r)? as usize,
            };
            let vs = (0..len)
                .map(|_| gguf_read_value(r, elem_type, ver))
                .collect::<Result<Vec<_>>>()?;
            Ok(gguf_file::Value::Array(vs))
        }
        10 => Ok(gguf_file::Value::U64(gguf_read_u64(r)?)),
        11 => Ok(gguf_file::Value::I64(gguf_read_u64(r)? as i64)),
        12 => Ok(gguf_file::Value::F64(gguf_read_f64(r)?)),
        v => candle_core::bail!("unknown GGUF value type {v}"),
    }
}

/// Wrapper around GGUF metadata for convenient access.
pub struct GgufMetadata {
    pub metadata: HashMap<String, gguf_file::Value>,
}

impl GgufMetadata {
    pub fn get_str(&self, key: &str) -> Result<String> {
        match self.metadata.get(key) {
            Some(gguf_file::Value::String(s)) => Ok(s.clone()),
            Some(v) => candle_core::bail!("expected string for {key}, got {v:?}"),
            None => candle_core::bail!("missing metadata key: {key}"),
        }
    }

    pub fn get_u32(&self, key: &str) -> Result<u32> {
        match self.metadata.get(key) {
            Some(v) => v.to_u32(),
            None => candle_core::bail!("missing metadata key: {key}"),
        }
    }

    pub fn get_f32(&self, key: &str) -> Result<f32> {
        match self.metadata.get(key) {
            Some(v) => v.to_f32(),
            None => candle_core::bail!("missing metadata key: {key}"),
        }
    }

    pub fn get_u32_or(&self, key: &str, default: u32) -> u32 {
        self.get_u32(key).unwrap_or(default)
    }

    pub fn get_f32_or(&self, key: &str, default: f32) -> f32 {
        self.get_f32(key).unwrap_or(default)
    }

    pub fn get_str_array(&self, key: &str) -> Result<Vec<String>> {
        match self.metadata.get(key) {
            Some(gguf_file::Value::Array(arr)) => {
                let mut result = Vec::new();
                for v in arr {
                    match v {
                        gguf_file::Value::String(s) => result.push(s.clone()),
                        _ => candle_core::bail!("expected string array for {key}"),
                    }
                }
                Ok(result)
            }
            Some(v) => candle_core::bail!("expected array for {key}, got {v:?}"),
            None => candle_core::bail!("missing metadata key: {key}"),
        }
    }

    /// Read an array of integers (e.g. the per-layer `*.attention.head_count_kv`
    /// LFM2 uses to mark conv vs. attention layers). Accepts any int width.
    pub fn get_i64_array(&self, key: &str) -> Result<Vec<i64>> {
        match self.metadata.get(key) {
            Some(gguf_file::Value::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    gguf_file::Value::I8(x) => Ok(*x as i64),
                    gguf_file::Value::I16(x) => Ok(*x as i64),
                    gguf_file::Value::I32(x) => Ok(*x as i64),
                    gguf_file::Value::I64(x) => Ok(*x),
                    other => Ok(other.to_u32().unwrap_or(0) as i64),
                })
                .collect(),
            Some(v) => candle_core::bail!("expected array for {key}, got {v:?}"),
            None => candle_core::bail!("missing metadata key: {key}"),
        }
    }

    /// Read an array of booleans. GGUF bool values use `Value::Bool`; also accepts numeric.
    pub fn get_bool_array(&self, key: &str) -> Result<Vec<bool>> {
        match self.metadata.get(key) {
            Some(gguf_file::Value::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    gguf_file::Value::Bool(b) => Ok(*b),
                    gguf_file::Value::U8(n) => Ok(*n != 0),
                    gguf_file::Value::I8(n) => Ok(*n != 0),
                    gguf_file::Value::U16(n) => Ok(*n != 0),
                    gguf_file::Value::I16(n) => Ok(*n != 0),
                    gguf_file::Value::U32(n) => Ok(*n != 0),
                    gguf_file::Value::I32(n) => Ok(*n != 0),
                    v => candle_core::bail!("expected bool/int in array for {key}, got {v:?}"),
                })
                .collect(),
            Some(v) => candle_core::bail!("expected array for {key}, got {v:?}"),
            None => candle_core::bail!("missing metadata key: {key}"),
        }
    }

    /// Read a per-layer array of u32 values. Accepts ARRAY(I32) and ARRAY(U32).
    /// If the field is a scalar, wraps it in a single-element Vec.
    pub fn get_u32_array(&self, key: &str) -> Result<Vec<u32>> {
        match self.metadata.get(key) {
            Some(gguf_file::Value::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    gguf_file::Value::U8(n) => Ok(*n as u32),
                    gguf_file::Value::I8(n) => Ok(*n as u32),
                    gguf_file::Value::U16(n) => Ok(*n as u32),
                    gguf_file::Value::I16(n) => Ok(*n as u32),
                    gguf_file::Value::U32(n) => Ok(*n),
                    gguf_file::Value::I32(n) => Ok(*n as u32),
                    v => candle_core::bail!("expected integer in array for {key}, got {v:?}"),
                })
                .collect(),
            Some(v) => Ok(vec![v.to_u32()?]),
            None => candle_core::bail!("missing metadata key: {key}"),
        }
    }
}

// ---------------------------------------------------------------------------
// QLinear: quantized linear layer (drop-in replacement for candle_nn::Linear)
// ---------------------------------------------------------------------------

/// A linear layer that can hold either quantized (QMatMul) or float weights.
pub struct QLinear {
    weight: candle_core::quantized::QMatMul,
    bias: Option<Tensor>,
}

impl QLinear {
    /// Create from a QTensor weight (typical GGUF loading path).
    pub fn new(weight: QTensor, bias: Option<Tensor>) -> Result<Self> {
        let weight = candle_core::quantized::QMatMul::from_qtensor(weight)?;
        Ok(Self { weight, bias })
    }

    /// Create from an Arc<QTensor>.
    pub fn from_arc(weight: Arc<QTensor>, bias: Option<Tensor>) -> Result<Self> {
        let weight = candle_core::quantized::QMatMul::from_arc(weight)?;
        Ok(Self { weight, bias })
    }

    /// Load from QVarBuilder (looks for "weight" and optionally "bias").
    pub fn load(vb: &QVarBuilder) -> Result<Self> {
        let weight = vb.get("weight")?;
        let bias = if vb.contains("bias") {
            Some(vb.get("bias")?.dequantize(vb.device())?)
        } else {
            None
        };
        Self::from_arc(weight, bias)
    }
}

impl QLinear {
    /// Returns the bias RMS for diagnostics.
    pub fn bias_rms(&self) -> Option<f32> {
        self.bias.as_ref().and_then(|b| {
            b.flatten_all()
                .ok()?
                .to_vec1::<f32>()
                .ok()
                .map(|v| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt())
        })
    }
    pub fn bias_shape(&self) -> Option<candle_core::Shape> {
        self.bias.as_ref().map(|b| b.shape().clone())
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.apply(&self.weight)?;
        match &self.bias {
            Some(bias) => out.broadcast_add(bias),
            None => Ok(out),
        }
    }
}

// ---------------------------------------------------------------------------
// QNorm: quantized RMSNorm / LayerNorm (dequantizes weight on load)
// ---------------------------------------------------------------------------

/// Normalization from quantized weights. Dequantizes the weight tensor on load
/// since norm weights are small and always used at full precision.
pub enum QNorm {
    Rms { weight: Tensor, eps: f64 },
    Layer { ln: candle_nn::LayerNorm },
}

impl QNorm {
    pub fn rms_from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self::Rms { weight, eps })
    }

    pub fn rms_load(eps: f64, vb: &QVarBuilder) -> Result<Self> {
        let weight = vb.get("weight")?.dequantize(vb.device())?;
        Ok(Self::Rms { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Rms { weight, eps } => candle_nn::ops::rms_norm(x, weight, *eps as f32),
            Self::Layer { ln } => ln.forward(x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An E8M0 byte of 128 is exactly 1.0, which is what makes the dequant
    /// tests below readable: every output is its raw LUT value.
    #[test]
    fn an_e8m0_byte_of_128_is_unit_scale() {
        assert_eq!(e8m0_to_f32(128), 1.0);
        assert_eq!(e8m0_to_f32(129), 2.0);
        assert_eq!(e8m0_to_f32(127), 0.5);
    }

    /// Pins the block layout the doc comment describes: the low nibble of byte
    /// `j` is element `j`, the high nibble is element `j + 16`. Nothing else
    /// tested this, and the two implementations (scalar and the AVX2 path taken
    /// on x86) have to agree on it.
    #[test]
    fn one_block_dequantizes_in_the_documented_nibble_order() {
        // Unit scale, then byte j carrying nibble j in both halves: j | (j << 4).
        let mut raw = vec![128u8];
        raw.extend((0..16u8).map(|j| j | (j << 4)));

        let out = dequantize_mxfp4(&raw, MXFP4_BLOCK_SIZE);

        assert_eq!(out.len(), MXFP4_BLOCK_SIZE);
        for j in 0..16 {
            let expected = E2M1_LUT[j] as f32;
            assert_eq!(out[j], expected, "low nibble of byte {j}");
            assert_eq!(out[j + 16], expected, "high nibble of byte {j}");
        }
    }

    /// The reason this function no longer allocates uninitialized: every
    /// element of a multi-block tensor must be written. A tail left untouched
    /// used to be uninitialized memory handed to `Tensor::from_vec`, and is now
    /// merely zero — so a test that catches a short write is worth having.
    #[test]
    fn every_element_of_a_multi_block_tensor_is_written() {
        const BLOCKS: usize = 3;
        let mut raw = Vec::new();
        for _ in 0..BLOCKS {
            raw.push(128u8); // unit scale
            raw.extend([0x11u8; 16]); // both nibbles = 1 → LUT[1] = 1.0
        }

        let out = dequantize_mxfp4(&raw, BLOCKS * MXFP4_BLOCK_SIZE);

        assert_eq!(out.len(), BLOCKS * MXFP4_BLOCK_SIZE);
        assert!(
            out.iter().all(|&v| v == 1.0),
            "some elements were never written: {:?}",
            out.iter().enumerate().find(|(_, &v)| v != 1.0)
        );
    }

    /// A ragged expert is refused in every build, not just in debug. The
    /// remainder is what makes it worth refusing: it under-counts the byte
    /// stride, so every expert after the first would read from the wrong
    /// offset — silently, and in release, where a `debug_assert` says nothing.
    #[test]
    fn a_ragged_expert_count_is_an_error_rather_than_a_short_read() {
        assert!(mxfp4_blocks_per_expert(MXFP4_BLOCK_SIZE + 1).is_err());
        assert!(mxfp4_blocks_per_expert(MXFP4_BLOCK_SIZE - 1).is_err());

        assert_eq!(mxfp4_blocks_per_expert(MXFP4_BLOCK_SIZE).unwrap(), 1);
        assert_eq!(mxfp4_blocks_per_expert(MXFP4_BLOCK_SIZE * 4).unwrap(), 4);
    }

    /// Says which tensor and which block size, since the shape is what the
    /// reader has to go and check.
    #[test]
    fn the_ragged_expert_error_names_the_count_and_the_block_size() {
        let err = mxfp4_blocks_per_expert(100).unwrap_err().to_string();
        assert!(err.contains("100"), "{err}");
        assert!(err.contains(&MXFP4_BLOCK_SIZE.to_string()), "{err}");
    }

    /// Negative E2M1 codes are the upper half of the LUT, and a sign error
    /// there would be invisible in a test that only used small positives.
    #[test]
    fn negative_codes_and_a_non_unit_scale_survive_dequant() {
        // Scale 2.0, then bytes whose low nibble is 0x7 (12) and high nibble
        // 0xF (-12).
        let mut raw = vec![129u8];
        raw.extend([0xF7u8; 16]);

        let out = dequantize_mxfp4(&raw, MXFP4_BLOCK_SIZE);

        for j in 0..16 {
            assert_eq!(out[j], 24.0, "low nibble (12 × 2) at {j}");
            assert_eq!(out[j + 16], -24.0, "high nibble (-12 × 2) at {j}");
        }
    }
}

#[cfg(test)]
mod qmatmul_equivalence {
    use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
    use candle_core::{Device, Module, Tensor};

    /// The two multiplies are **close but not the same operation**, and this
    /// records by how much — because the natural assumption behind
    /// [`QExperts::qmatmuls`] is that swapping them is free, and it is not.
    ///
    /// Dequantizing first gives an f32 weight and an f32 product. `QMatMul` runs
    /// ggml's kernel instead, which quantizes the *activations* to 8 bits and
    /// dots them against the weight blocks (`vec_dot_q4k_q8k` for a Q4_K
    /// weight). So the MoE rewrite changed the model's arithmetic as well as its
    /// speed. Measured here at well under 1% relative, and in the direction of
    /// agreement rather than away from it: this is the same arithmetic
    /// llama.cpp performs for the same GGUF, which is what the two backends are
    /// otherwise compared against.
    ///
    /// The structured weight matters: a transposed-vs-not mistake would show up
    /// as a large error rather than this one.
    #[test]
    fn a_quantized_matmul_matches_dequantize_then_matmul() {
        let device = Device::Cpu;
        // Q4_K quantizes along the last dimension in blocks of 256, so `in_dim`
        // has to be a multiple of it — the same constraint the merged expert
        // tensors satisfy.
        let (out_dim, in_dim, batch) = (128usize, 256usize, 3usize);

        // A weight with structure, so a transposed-vs-not mistake cannot pass.
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 37) as f32 - 18.0) / 19.0)
            .collect();
        let w = Tensor::from_vec(w, (out_dim, in_dim), &device).unwrap();
        let qt = QTensor::quantize(&w, GgmlDType::Q4K).unwrap();

        let x: Vec<f32> = (0..batch * in_dim)
            .map(|i| ((i % 13) as f32 - 6.0) / 7.0)
            .collect();
        let x = Tensor::from_vec(x, (batch, in_dim), &device).unwrap();

        // The path the MoE forward used to take.
        let dequantized = qt.dequantize(&device).unwrap();
        let by_hand = x
            .matmul(&dequantized.t().unwrap().contiguous().unwrap())
            .unwrap();

        // The path it takes now.
        let by_qmatmul = QMatMul::from_qtensor(qt).unwrap().forward(&x).unwrap();

        assert_eq!(by_hand.dims(), by_qmatmul.dims());
        let a = by_hand.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = by_qmatmul.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let scale = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        let relative = worst / scale;
        assert!(
            relative < 1e-2,
            "quantized matmul differs from dequantize-then-matmul by {worst} \
             (scale {scale}, {relative} relative) — more than 8-bit activation \
             quantization explains, so check the layout before the numerics"
        );
    }
}
