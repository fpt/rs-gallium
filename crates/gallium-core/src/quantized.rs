//! Quantized layer support for GGUF model loading.
//!
//! Provides `QVarBuilder` for loading GGUF files, and `QLinear` / `QNorm` as
//! drop-in replacements for `Linear` / `Norm` that work with quantized weights.

use candle_core::quantized::{gguf_file, GgmlDType, QStorage, QTensor};
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
            LazyQTensor::Lazy { source, offset, dtype, shape, device, .. } => QExperts {
                source: source.clone(),
                offset: *offset,
                dtype: *dtype,
                dims: shape.dims().to_vec(),
                device: device.clone(),
            },
        }
    }

    fn get(&self) -> Result<Arc<QTensor>> {
        match self {
            LazyQTensor::Lazy { source, offset, size, dtype, shape, device, cell } => {
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
        let n_blocks = n_elems_per_expert / MXFP4_BLOCK_SIZE;
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
    device: Device,
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
        let qt = QTensor::new(storage, candle_core::Shape::from(self.dims[1..].to_vec()))?;
        qt.dequantize(device)
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

    let source = Arc::new(MmapSource { mmap, base: tensor_data_offset });

    let mut lazy_tensors: HashMap<String, LazyQTensor> = HashMap::new();
    let mut tq2_map: HashMap<String, Tq2Tensor> = HashMap::new();

    for (name, info) in &tensor_infos {
        let n_elems: usize = info.dims.iter().product();

        if info.dtype_u32 == MXFP4_TYPE {
            // MXFP4: no pre-copy; dequantize_expert slices the mmap on demand.
            let n_blocks = n_elems / MXFP4_BLOCK_SIZE;
            let _size = n_blocks * MXFP4_BYTES_PER_BLOCK; // kept for future bounds checking
            tq2_map.insert(name.clone(), Tq2Tensor {
                source: source.clone(),
                offset: info.offset,
                dims: info.dims.clone(),
            });
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
            lazy_tensors.insert(name.clone(), LazyQTensor::Lazy {
                source: source.clone(),
                offset: info.offset,
                size,
                dtype,
                shape,
                device: device.clone(),
                cell: Mutex::new(None),
            });
        }
    }

    let vb = QVarBuilder {
        data: Arc::new(lazy_tensors),
        tq2_raw: Arc::new(tq2_map),
        path: Vec::new(),
        device: device.clone(),
    };
    let metadata = GgufMetadata { metadata: metadata_map };
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
    // Safety: every element is written by dequantize_mxfp4_into before use.
    let mut out = Vec::with_capacity(n_elems);
    unsafe { out.set_len(n_elems) };
    dequantize_mxfp4_into(raw, &mut out);
    out
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
            out[out_base + j     ] = E2M1_LUT[(byte & 0xF) as usize] as f32 * scale;
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
    let lut = _mm_set_epi8(
        -12, -8, -6, -4, -3, -2, -1, 0,
         12,  8,  6,  4,  3,  2,  1, 0_i8,
    );
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
                raw.as_ptr().add((blk + PREFETCH_DIST) * MXFP4_BYTES_PER_BLOCK) as *const i8,
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
        _mm256_storeu_ps(out_ptr,         to_f32x8!(lo_i8));
        _mm256_storeu_ps(out_ptr.add(8),  to_f32x8!(_mm_srli_si128::<8>(lo_i8)));
        _mm256_storeu_ps(out_ptr.add(16), to_f32x8!(hi_i8));
        _mm256_storeu_ps(out_ptr.add(24), to_f32x8!(_mm_srli_si128::<8>(hi_i8)));
    }
}

// ─── Minimal GGUF parser (tolerates unknown tensor dtypes) ───────────────────

#[derive(Clone, Copy)]
enum GgufVersion { V1, V2V3 }

struct RawTensorInfo {
    dims: Vec<usize>, // already reversed to row-major
    dtype_u32: u32,
    offset: u64,
}

fn parse_gguf_tolerant<R: Read + Seek>(
    r: &mut R,
) -> Result<(HashMap<String, gguf_file::Value>, HashMap<String, RawTensorInfo>, u64)> {
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
            GgufVersion::V1 => (0..n_dims).map(|_| gguf_read_u32(r).map(|v| v as usize)).collect::<Result<_>>()?,
            GgufVersion::V2V3 => (0..n_dims).map(|_| gguf_read_u64(r).map(|v| v as usize)).collect::<Result<_>>()?,
        };
        dims.reverse();
        let dtype_u32 = gguf_read_u32(r)?;
        let offset = gguf_read_u64(r)?;
        tensor_infos.insert(name, RawTensorInfo { dims, dtype_u32, offset });
    }

    // Tensor data offset (aligned)
    let pos = r.stream_position()?;
    let alignment: u64 = match metadata.get("general.alignment") {
        Some(gguf_file::Value::U32(v)) => *v as u64,
        Some(gguf_file::Value::U8(v))  => *v as u64,
        Some(gguf_file::Value::U16(v)) => *v as u64,
        _ => 32,
    };
    let tensor_data_offset = pos.div_ceil(alignment) * alignment;
    Ok((metadata, tensor_infos, tensor_data_offset))
}

/// Map a GGUF dtype u32 to `GgmlDType`. Mirrors candle's private `from_u32`.
fn ggml_dtype_from_u32(u: u32) -> Result<GgmlDType> {
    match u {
        0  => Ok(GgmlDType::F32),
        1  => Ok(GgmlDType::F16),
        2  => Ok(GgmlDType::Q4_0),
        3  => Ok(GgmlDType::Q4_1),
        6  => Ok(GgmlDType::Q5_0),
        7  => Ok(GgmlDType::Q5_1),
        8  => Ok(GgmlDType::Q8_0),
        9  => Ok(GgmlDType::Q8_1),
        10 => Ok(GgmlDType::Q2K),
        11 => Ok(GgmlDType::Q3K),
        12 => Ok(GgmlDType::Q4K),
        13 => Ok(GgmlDType::Q5K),
        14 => Ok(GgmlDType::Q6K),
        15 => Ok(GgmlDType::Q8K),
        30 => Ok(GgmlDType::BF16),
        v  => candle_core::bail!("unknown GgmlDType {v}"),
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
        GgufVersion::V1    => gguf_read_u32(r)? as usize,
        GgufVersion::V2V3  => gguf_read_u64(r)? as usize,
    };
    let mut v = vec![0u8; len];
    r.read_exact(&mut v)?;
    while let Some(0) = v.last() { v.pop(); }
    Ok(String::from_utf8_lossy(&v).into_owned())
}

fn gguf_read_value<R: Read>(r: &mut R, vtype: u32, ver: GgufVersion) -> Result<gguf_file::Value> {
    match vtype {
        0  => Ok(gguf_file::Value::U8(gguf_read_u8(r)?)),
        1  => Ok(gguf_file::Value::I8(gguf_read_u8(r)? as i8)),
        2  => Ok(gguf_file::Value::U16(gguf_read_u16(r)?)),
        3  => Ok(gguf_file::Value::I16(gguf_read_u16(r)? as i16)),
        4  => Ok(gguf_file::Value::U32(gguf_read_u32(r)?)),
        5  => Ok(gguf_file::Value::I32(gguf_read_u32(r)? as i32)),
        6  => Ok(gguf_file::Value::F32(gguf_read_f32(r)?)),
        7  => Ok(gguf_file::Value::Bool(gguf_read_u8(r)? != 0)),
        8  => Ok(gguf_file::Value::String(gguf_read_string(r, ver)?)),
        9  => {
            let elem_type = gguf_read_u32(r)?;
            let len = match ver {
                GgufVersion::V1   => gguf_read_u32(r)? as usize,
                GgufVersion::V2V3 => gguf_read_u64(r)? as usize,
            };
            let vs = (0..len).map(|_| gguf_read_value(r, elem_type, ver)).collect::<Result<Vec<_>>>()?;
            Ok(gguf_file::Value::Array(vs))
        }
        10 => Ok(gguf_file::Value::U64(gguf_read_u64(r)?)),
        11 => Ok(gguf_file::Value::I64(gguf_read_u64(r)? as i64)),
        12 => Ok(gguf_file::Value::F64(gguf_read_f64(r)?)),
        v  => candle_core::bail!("unknown GGUF value type {v}"),
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
            Some(gguf_file::Value::Array(arr)) => {
                arr.iter().map(|v| match v {
                    gguf_file::Value::Bool(b) => Ok(*b),
                    gguf_file::Value::U8(n)  => Ok(*n != 0),
                    gguf_file::Value::I8(n)  => Ok(*n != 0),
                    gguf_file::Value::U16(n) => Ok(*n != 0),
                    gguf_file::Value::I16(n) => Ok(*n != 0),
                    gguf_file::Value::U32(n) => Ok(*n != 0),
                    gguf_file::Value::I32(n) => Ok(*n != 0),
                    v => candle_core::bail!("expected bool/int in array for {key}, got {v:?}"),
                }).collect()
            }
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
                    gguf_file::Value::U8(n)  => Ok(*n as u32),
                    gguf_file::Value::I8(n)  => Ok(*n as u32),
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
            b.flatten_all().ok()?.to_vec1::<f32>().ok().map(|v| {
                (v.iter().map(|x| x*x).sum::<f32>() / v.len() as f32).sqrt()
            })
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
