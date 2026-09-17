pub mod gemma4;
#[cfg(feature = "vision")]
pub mod gemma4_image;
/// Re-exported so `tests/integration.rs` (a separate crate) can reach the raw
/// kernel for its microbenchmark, isolating candle-flash-attn's own hdim-512
/// correctness from this crate's call-site code — see
/// `gemma4_q::QAttention::flash_attention`'s doc comment (issue #308).
/// `candle-flash-attn` can't be a `[dev-dependencies]` entry (Cargo refuses
/// an optional dev-dependency outright), so a re-export is the way out.
#[cfg(feature = "flash-attn")]
pub use candle_flash_attn;
pub mod gemma4_q;
pub mod gemma4_vision;
pub mod gpt_oss;
pub mod gpt_oss_q;
pub mod lfm2moe_q;
pub mod loader;
pub mod qwen35_q;
