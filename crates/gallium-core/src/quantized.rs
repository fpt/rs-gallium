//! Quantized layer support for GGUF model loading.
//!
//! Provides `QVarBuilder` for loading GGUF files, and `QLinear` / `QNorm` as
//! drop-in replacements for `Linear` / `Norm` that work with quantized weights.

use candle_core::quantized::{gguf_file, GgmlDType, QStorage, QTensor};
use candle_core::{Device, Module, Result, Tensor};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// QVarBuilder: navigate GGUF tensors with dot-separated prefixes (like VarBuilder)
// ---------------------------------------------------------------------------

/// A TQ2_0 tensor stored as raw bytes for lazy per-expert dequantization.
/// Dims are row-major (outer dimension first), e.g. `[n_expert, n_ff, n_embd]`.
#[derive(Clone)]
pub struct Tq2Tensor {
    pub bytes: Arc<Vec<u8>>,
    pub dims: Vec<usize>,
}

impl Tq2Tensor {
    /// Dequantize the slice for expert `idx` into a float Tensor with shape `dims[1..]`.
    pub fn dequantize_expert(&self, idx: usize, device: &Device) -> Result<Tensor> {
        let n_elems_per_expert: usize = self.dims[1..].iter().product();
        let n_blocks = n_elems_per_expert / MXFP4_BLOCK_SIZE;
        let bytes_per_expert = n_blocks * MXFP4_BYTES_PER_BLOCK;
        let start = idx * bytes_per_expert;
        let floats = dequantize_mxfp4(&self.bytes[start..start + bytes_per_expert], n_elems_per_expert);
        Tensor::from_vec(floats, self.dims[1..].to_vec().as_slice(), device)
    }
}

#[derive(Clone)]
pub struct QVarBuilder {
    data: Arc<HashMap<String, Arc<QTensor>>>,
    /// Raw TQ2_0 bytes for lazy per-expert dequantization.
    tq2_raw: Arc<HashMap<String, Tq2Tensor>>,
    path: Vec<String>,
    device: Device,
}

impl QVarBuilder {
    /// Load all tensors from a GGUF file into memory.
    /// Note: does not support MXFP4 (type 39). Use `load_gguf()` instead.
    pub fn from_gguf<P: AsRef<std::path::Path>>(path: P, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;
        Self::from_gguf_content(&content, &mut file, device)
    }

    /// Load from an already-parsed GGUF Content + reader.
    pub fn from_gguf_content<R: std::io::Seek + std::io::Read>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut data = HashMap::new();
        for tensor_name in content.tensor_infos.keys() {
            let tensor = content.tensor(reader, tensor_name, device)?;
            data.insert(tensor_name.to_string(), Arc::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            tq2_raw: Arc::new(HashMap::new()),
            path: Vec::new(),
            device: device.clone(),
        })
    }

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

    /// Get the raw MXFP4 data for lazy per-expert dequantization.
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

    /// Get a quantized tensor by name (with prefix).
    pub fn get(&self, name: &str) -> Result<Arc<QTensor>> {
        let path = self.full_path(name);
        self.data
            .get(&path)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg(format!("cannot find tensor: {path}")))
    }

    /// Check if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        let path = self.full_path(name);
        self.data.contains_key(&path)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Access the underlying GGUF metadata (call from_gguf_with_metadata instead).
    /// List all tensor names (useful for debugging).
    pub fn tensor_names(&self) -> Vec<&str> {
        self.data.keys().map(|s| s.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// GGUF metadata reader (for extracting config from GGUF header)
// ---------------------------------------------------------------------------

/// Read GGUF metadata and create a QVarBuilder in one step.
/// Supports MXFP4 (type 39) tensors with lazy per-expert dequantization.
/// Returns (metadata, var_builder).
pub fn load_gguf<P: AsRef<std::path::Path>>(
    path: P,
    device: &Device,
) -> Result<(GgufMetadata, QVarBuilder)> {
    let mut file = std::fs::File::open(path.as_ref())?;
    let (metadata_map, tensor_infos, tensor_data_offset) = parse_gguf_tolerant(&mut file)?;

    let mut qtensors: HashMap<String, Arc<QTensor>> = HashMap::new();
    let mut tq2_map: HashMap<String, Tq2Tensor> = HashMap::new();

    for (name, info) in &tensor_infos {
        let n_elems: usize = info.dims.iter().product();

        if info.dtype_u32 == MXFP4_TYPE {
            // MXFP4: store raw bytes for lazy per-expert dequantization at forward time.
            let n_blocks = n_elems / MXFP4_BLOCK_SIZE;
            let raw_size = n_blocks * MXFP4_BYTES_PER_BLOCK;
            let mut raw = vec![0u8; raw_size];
            file.seek(SeekFrom::Start(tensor_data_offset + info.offset))?;
            file.read_exact(&mut raw)?;
            tq2_map.insert(name.clone(), Tq2Tensor {
                bytes: Arc::new(raw),
                dims: info.dims.clone(),
            });
        } else {
            // Known quantization: create QTensor from raw bytes.
            let dtype = ggml_dtype_from_u32(info.dtype_u32)?;
            let block_size = dtype.block_size();
            let type_size = dtype.type_size();
            if n_elems % block_size != 0 {
                candle_core::bail!(
                    "tensor {name}: elem count {n_elems} not divisible by block size {block_size}"
                );
            }
            let raw_size = n_elems / block_size * type_size;
            let mut raw = vec![0u8; raw_size];
            file.seek(SeekFrom::Start(tensor_data_offset + info.offset))?;
            file.read_exact(&mut raw)?;
            let shape = candle_core::Shape::from(info.dims.clone());
            // Use Cow::Borrowed to keep `raw` alive during as_t_slice's unsafe reinterpret-cast
            // (Cow::Owned causes use-after-free: from_data drops the Vec before .to_vec() copies it)
            let storage = QStorage::from_data(Cow::Borrowed(&raw), device, dtype)?;
            let qtensor = QTensor::new(storage, shape)?;
            qtensors.insert(name.clone(), Arc::new(qtensor));
        }
    }

    let vb = QVarBuilder {
        data: Arc::new(qtensors),
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
    let n_blocks = n_elems / MXFP4_BLOCK_SIZE;
    let mut out = vec![0f32; n_elems];
    for blk in 0..n_blocks {
        let base = blk * MXFP4_BYTES_PER_BLOCK;
        let scale = e8m0_to_f32(raw[base]);
        let out_base = blk * MXFP4_BLOCK_SIZE;
        // Lower nibbles → elements 0..16, upper nibbles → elements 16..32
        for j in 0..16usize {
            let byte = raw[base + 1 + j];
            out[out_base + j     ] = E2M1_LUT[(byte & 0xF) as usize] as f32 * scale;
            out[out_base + j + 16] = E2M1_LUT[(byte >> 4) as usize] as f32 * scale;
        }
    }
    out
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

    /// Read an array of booleans. GGUF bool values use `Value::Bool`; also accepts numeric.
    pub fn get_bool_array(&self, key: &str) -> Result<Vec<bool>> {
        match self.metadata.get(key) {
            Some(gguf_file::Value::Array(arr)) => {
                arr.iter().map(|v| match v {
                    gguf_file::Value::Bool(b) => Ok(*b),
                    _ => Ok(v.to_u32().unwrap_or(0) != 0),
                }).collect()
            }
            Some(v) => candle_core::bail!("expected array for {key}, got {v:?}"),
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
