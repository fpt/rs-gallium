use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

/// Normalization layer — wraps candle-nn's implementations.
pub enum Norm {
    Rms(candle_nn::RmsNorm),
    Layer(candle_nn::LayerNorm),
    /// Qwen3.5-style RMSNorm: weight initialized to zeros, formula = norm(x) * (1 + weight).
    /// Standard RMSNorm (weight=ones) is equivalent here at init, but stored checkpoints
    /// contain delta values. We add 1 at load time to restore correct scaling.
    RmsOnePlus { weight: Tensor, eps: f64 },
}

impl Norm {
    pub fn rms(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self::Rms(candle_nn::rms_norm(size, eps, vb)?))
    }

    /// Qwen3.5 variant: loaded weight is a delta from zero; effective scale = (1 + weight).
    pub fn rms_one_plus(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self::RmsOnePlus { weight, eps })
    }

    pub fn layer(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let cfg = candle_nn::LayerNormConfig {
            eps,
            ..Default::default()
        };
        Ok(Self::Layer(candle_nn::layer_norm(size, cfg, vb)?))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Rms(n) => n.forward(x),
            Self::Layer(n) => n.forward(x),
            Self::RmsOnePlus { weight, eps } => {
                let orig = x.dtype();
                let xf = x.to_dtype(DType::F32)?;
                let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
                let normed = xf.broadcast_div(&(var + *eps)?.sqrt()?)?;
                // Effective scale = (1 + stored_weight)
                let scale = (weight.to_dtype(DType::F32)? + 1.0_f64)?;
                normed.broadcast_mul(&scale)?.to_dtype(orig)
            }
        }
    }
}
