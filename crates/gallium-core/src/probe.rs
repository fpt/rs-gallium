//! Where does a forward pass start disagreeing with itself on another device?
//!
//! [`docs/CANDLE_BACKEND.md`] §6b and §6c both end on the same missing
//! measurement. Two `engine_logits` comparisons found candle's prefill ~0.2–0.3
//! of a logit away — from llama.cpp on Metal, and from candle's *own* CPU
//! forward on CUDA — with every isolated suspect cleared: the Q4_K dequant is
//! bit-identical across devices, cuBLAS is honest fp32, and the quantized matmul
//! kernels are accurate enough at the shapes involved. That leaves two
//! explanations the final logits cannot tell apart. Either a small per-layer
//! difference compounds through the residual stream, or one operator somewhere
//! carries the whole thing and the rest is innocent. The two look completely
//! different one layer at a time and identical at the head, which is the only
//! place anything has been measured so far.
//!
//! So: a fingerprint per stage, cheap enough to leave in the model, emitted only
//! when someone asks for it.
//!
//! ```text
//! GALLIUM_DEVICE=cpu   RUST_LOG=gallium::layers=trace gallium … 2> cpu.log
//! GALLIUM_DEVICE=metal RUST_LOG=gallium::layers=trace gallium … 2> metal.log
//! uv run python scripts/layer_diff.py cpu.log metal.log
//! ```
//!
//! The two devices need not be the same machine — the logs are text, and CPU is
//! the reference both a Metal box and a CUDA box can produce. That matters here
//! because they *are* different machines.
//!
//! Reading the diff: a **step** at one stage is an operator, and the stage says
//! which one. A ratio that climbs smoothly from the first stage to the last is
//! compounding, and no single operator is at fault. Stage 0 is the embedding
//! lookup, before any arithmetic — a difference there is the GGUF read itself
//! and nothing downstream is worth reading.

use candle_core::{DType, Result, Tensor};

/// The tracing target. Off unless `RUST_LOG` names it, and the tensor is never
/// read back when it is off — a `to_vec1` on an accelerator is a sync point.
pub const TARGET: &str = "gallium::layers";

/// Channels sampled from the final position's hidden state.
const SAMPLES: usize = 16;

/// Record one stage of a forward pass, if anyone is listening.
///
/// `stage` is the caller's numbering; the convention this crate's models follow
/// is 0 for the embedding output and `i + 1` for the output of block `i`, so a
/// disagreement at stage `n` entered in block `n - 1`.
pub fn hidden(stage: usize, h: &Tensor) {
    if !tracing::enabled!(target: "gallium::layers", tracing::Level::TRACE) {
        return;
    }
    match fingerprint(h) {
        Ok(line) => tracing::trace!(target: "gallium::layers", "stage={stage:02} {line}"),
        Err(e) => tracing::trace!(target: "gallium::layers", "stage={stage:02} unavailable: {e}"),
    }
}

/// Aggregate statistics plus a fixed sample of the last position's row.
///
/// Both halves are needed and neither is enough. The aggregates cover every
/// element but average away a difference confined to a few channels; the sample
/// is pointwise but sees 16 channels of thousands. The last position is the one
/// the logits are read from, so it is the row a divergence has to reach to
/// matter.
fn fingerprint(h: &Tensor) -> Result<String> {
    let h = h.to_dtype(DType::F32)?;
    let width = *h.dims().last().unwrap_or(&1);
    let flat = h.flatten_all()?.to_vec1::<f32>()?;
    let n = flat.len().max(1) as f32;

    let mean = flat.iter().sum::<f32>() / n;
    let rms = (flat.iter().map(|v| v * v).sum::<f32>() / n).sqrt();
    let absmax = flat.iter().fold(0f32, |m, v| m.max(v.abs()));

    let row = &flat[flat.len().saturating_sub(width)..];
    let stride = (width / SAMPLES).max(1);
    let sample: Vec<String> = row
        .iter()
        .step_by(stride)
        .take(SAMPLES)
        .map(|v| format!("{v:.6e}"))
        .collect();

    Ok(format!(
        "rms={rms:.6e} absmax={absmax:.6e} mean={mean:.6e} last=[{}]",
        sample.join(",")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn a_fingerprint_reports_the_last_row_not_the_first() {
        // Two positions that differ: the fingerprint must describe the second,
        // since that is the one the logits come from. A probe that silently
        // reported position 0 would agree across devices for the wrong reason.
        let t = Tensor::from_vec(
            vec![0f32, 0., 0., 0., 1., 2., 3., 4.],
            (1, 2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let line = fingerprint(&t).unwrap();
        assert!(
            line.contains("last=[1.000000e0,2.000000e0,3.000000e0,4.000000e0]"),
            "{line}"
        );
    }

    #[test]
    fn the_aggregates_cover_every_element() {
        let t = Tensor::from_vec(vec![3f32, -4.], (1, 1, 2), &Device::Cpu).unwrap();
        let line = fingerprint(&t).unwrap();
        // rms over both elements is sqrt((9+16)/2) = 3.535534, and absmax sees
        // the negative one — a divergence is a magnitude, never a sign.
        assert!(line.contains("rms=3.535534e0"), "{line}");
        assert!(line.contains("absmax=4.000000e0"), "{line}");
    }
}

/// Where does a forward pass spend its *time*, stage by stage?
///
/// The fingerprint probe above answers "where does it start disagreeing";
/// this answers the other question a slow device raises. Off unless `RUST_LOG`
/// names [`TIMING_TARGET`], and then every `lap` **synchronises the device** —
/// reads one element back — before taking the clock, so the interval is the
/// stage's own GPU work rather than the time to enqueue it. That serialises
/// the pipeline, so the absolute total runs a little long; the *split* is
/// what it is for. Nothing is read back when it is off.
///
/// ```text
/// RUST_LOG=gallium::timing=trace gallium --config … 2> timing.log
/// grep 'stage timing' timing.log
/// ```
pub const TIMING_TARGET: &str = "gallium::timing";

pub struct StageTimer {
    enabled: bool,
    last: std::time::Instant,
    acc: Vec<(&'static str, std::time::Duration)>,
}

impl StageTimer {
    pub fn start() -> Self {
        Self {
            enabled: tracing::enabled!(target: TIMING_TARGET, tracing::Level::TRACE),
            last: std::time::Instant::now(),
            acc: Vec::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Close the interval since the previous lap and charge it to `stage`,
    /// after waiting for `done` — a tensor the stage produced — to be
    /// computed. Stages repeat across layers and are summed by name.
    pub fn lap(&mut self, stage: &'static str, done: &Tensor) {
        if !self.enabled {
            return;
        }
        // One element, whatever the dtype: enough to force the queue to drain.
        let _ = done
            .flatten_all()
            .and_then(|t| t.narrow(0, 0, 1))
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1::<f32>());
        let now = std::time::Instant::now();
        let d = now - self.last;
        self.last = now;
        match self.acc.iter_mut().find(|(s, _)| *s == stage) {
            Some((_, t)) => *t += d,
            None => self.acc.push((stage, d)),
        }
    }

    /// Emit the per-stage totals for this forward, longest first, with the
    /// share of the whole.
    pub fn report(&self, what: &str, tokens: usize) {
        if !self.enabled {
            return;
        }
        let total: f64 = self.acc.iter().map(|(_, d)| d.as_secs_f64()).sum();
        let mut rows = self.acc.clone();
        rows.sort_by(|a, b| b.1.cmp(&a.1));
        let body: Vec<String> = rows
            .iter()
            .map(|(s, d)| {
                let ms = d.as_secs_f64() * 1000.0;
                format!("{s}={ms:.0}ms({:.0}%)", ms / (total * 10.0))
            })
            .collect();
        tracing::trace!(
            target: TIMING_TARGET,
            "stage timing {what} tokens={tokens} total={:.0}ms: {}",
            total * 1000.0,
            body.join(" ")
        );
    }
}
