//! Which device a model runs on.
//!
//! Accelerator support is decided at build time (see this crate's Cargo.toml:
//! macOS compiles candle's Metal backend in), and which of the compiled-in
//! devices to actually use is decided here, at run time, by `GALLIUM_DEVICE`.
//!
//! The split matters for benchmarking: one binary can run the same model on CPU
//! and on the GPU, so a comparison is a change of one env var rather than a
//! rebuild.

use anyhow::Result;
use candle_core::Device;

/// Resolve a device from a spec: `auto` (the default), `cpu`, `metal`, `cuda`,
/// or an ordinal-bearing form like `metal:0` / `cuda:1`.
///
/// `auto` prefers an accelerator this build has and this machine can create, and
/// falls back to CPU. A spec that *names* an accelerator is a request that can
/// fail: asking for Metal on a build without it, or a machine without it, is an
/// error rather than a silent CPU run, because a benchmark that quietly measured
/// the wrong device is worse than one that refuses.
pub fn resolve_device(spec: Option<&str>) -> Result<Device> {
    let spec = spec
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("auto");
    let (kind, ordinal) = split_ordinal(spec)?;

    match kind.to_ascii_lowercase().as_str() {
        "auto" => Ok(auto_device()),
        "cpu" => Ok(Device::Cpu),
        "metal" | "mps" => Device::new_metal(ordinal)
            .map_err(|e| anyhow::anyhow!("GALLIUM_DEVICE={spec} is unavailable: {e}")),
        "cuda" | "gpu" => Device::new_cuda(ordinal)
            .map_err(|e| anyhow::anyhow!("GALLIUM_DEVICE={spec} is unavailable: {e}")),
        other => anyhow::bail!(
            "unknown GALLIUM_DEVICE '{other}' (expected auto, cpu, metal, or cuda, \
             optionally with an ordinal like metal:0)"
        ),
    }
}

/// `metal:1` → ("metal", 1); `metal` → ("metal", 0).
fn split_ordinal(spec: &str) -> Result<(&str, usize)> {
    match spec.split_once(':') {
        None => Ok((spec, 0)),
        Some((kind, n)) => {
            let ordinal = n
                .parse()
                .map_err(|_| anyhow::anyhow!("'{n}' is not a device ordinal in '{spec}'"))?;
            Ok((kind, ordinal))
        }
    }
}

/// The best device this build and machine offer, CPU last.
///
/// Failure to create an accelerator is logged and not fatal: `auto` promises a
/// working device, not a fast one.
fn auto_device() -> Device {
    match Device::new_metal(0) {
        Ok(device) => return device,
        Err(e) if candle_core::utils::metal_is_available() => {
            tracing::warn!("Metal is compiled in but unusable, falling back to CPU: {e}");
        }
        Err(_) => {}
    }
    match Device::new_cuda(0) {
        Ok(device) => return device,
        Err(e) if candle_core::utils::cuda_is_available() => {
            tracing::warn!("CUDA is compiled in but unusable, falling back to CPU: {e}");
        }
        Err(_) => {}
    }
    Device::Cpu
}

/// A short name for logs and banners: "cpu", "metal", "cuda".
///
/// No ordinal: candle's `MetalDevice` exposes an opaque id rather than the index
/// it was created from, and a number that does not match what `GALLIUM_DEVICE`
/// takes would be worse than none.
pub fn device_name(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Metal(_) => "metal",
        Device::Cuda(_) => "cuda",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `auto` must always produce something, whatever the build has.
    #[test]
    fn auto_always_resolves() {
        assert!(resolve_device(None).is_ok());
        assert!(resolve_device(Some("auto")).is_ok());
        assert!(resolve_device(Some("  ")).is_ok());
    }

    #[test]
    fn cpu_is_always_available() {
        let d = resolve_device(Some("cpu")).unwrap();
        assert!(matches!(d, Device::Cpu));
        assert_eq!(device_name(&d), "cpu");
    }

    /// A named accelerator is a request, so a bad name is an error rather than a
    /// silent CPU run — the failure mode this is guarding against is a benchmark
    /// that measured CPU while reporting Metal.
    #[test]
    fn an_unknown_device_is_an_error() {
        assert!(resolve_device(Some("tpu")).is_err());
        assert!(resolve_device(Some("metal:x")).is_err());
    }

    /// On a build with Metal compiled in (macOS), asking for it must work; on any
    /// other build, asking must fail rather than quietly give CPU.
    #[test]
    fn naming_metal_matches_what_the_build_has() {
        let asked = resolve_device(Some("metal"));
        assert_eq!(asked.is_ok(), candle_core::utils::metal_is_available());
    }
}
