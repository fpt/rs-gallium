use candle_core::{DType, Device, Result};
use candle_nn::VarBuilder;
use std::path::PathBuf;

/// Model source format.
pub enum ModelSource {
    Safetensors(Vec<PathBuf>),
}

/// Load model weights from safetensors files.
///
/// Uses memory-mapped I/O for efficient loading (no full copy into RAM).
pub fn load_safetensors(
    paths: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device) }
}

/// Read a JSON config file and deserialize it.
pub fn load_config<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> anyhow::Result<T> {
    let content = std::fs::read_to_string(path)?;
    let config: T = serde_json::from_str(&content)?;
    Ok(config)
}
