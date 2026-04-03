use candle_core::{Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

/// Activation function.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    Silu,
    Gelu,
    #[serde(alias = "gelu_pytorch_tanh")]
    GeluTanh,
    Relu,
}

impl Activation {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Silu => candle_nn::ops::silu(x),
            Self::Gelu => x.gelu_erf(),
            Self::GeluTanh => x.gelu(),
            Self::Relu => x.relu(),
        }
    }
}

/// Gated Feed-Forward Network (SwiGLU, GeGLU, etc.).
///
/// Computes: down_proj(activation(gate_proj(x)) * up_proj(x))
/// With optional clamping (GPT-OSS swiglu_limit).
pub struct GatedFFN {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: Activation,
    /// Optional clamp value for numerical stability (GPT-OSS: 7.0).
    clamp: Option<f32>,
}

impl GatedFFN {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        activation: Activation,
        clamp: Option<f32>,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
            activation,
            clamp,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.activation.forward(&self.gate_proj.forward(x)?)?;
        let gate = if let Some(limit) = self.clamp {
            gate.clamp(-limit, limit)?
        } else {
            gate
        };
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}

/// Mixture-of-Experts FFN.
///
/// Routes each token to `num_experts_per_tok` experts and combines their outputs.
pub struct MoEFFN {
    experts: Vec<GatedFFN>,
    /// Optional shared expert that processes all tokens (Qwen 3.5 MoE).
    shared_expert: Option<GatedFFN>,
    gate: Linear,
    num_experts_per_tok: usize,
}

impl MoEFFN {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        activation: Activation,
        clamp: Option<f32>,
        shared_expert_intermediate: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let experts = (0..num_experts)
            .map(|i| {
                GatedFFN::new(
                    hidden_size,
                    intermediate_size,
                    activation,
                    clamp,
                    vb.pp(format!("experts.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let shared_expert = shared_expert_intermediate
            .map(|inter| {
                GatedFFN::new(
                    hidden_size,
                    inter,
                    activation,
                    clamp,
                    vb.pp("shared_expert"),
                )
            })
            .transpose()?;

        let gate = linear_no_bias(hidden_size, num_experts, vb.pp("gate"))?;

        Ok(Self {
            experts,
            shared_expert,
            gate,
            num_experts_per_tok,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, hidden) = x.dims3()?;
        let x_flat = x.reshape((b * seq_len, hidden))?; // (tokens, hidden)

        // Router logits and top-k selection
        let router_logits = self.gate.forward(&x_flat)?; // (tokens, num_experts)
        let router_probs = candle_nn::ops::softmax_last_dim(&router_logits)?;

        let num_tokens = b * seq_len;
        let router_probs_vec: Vec<Vec<f32>> = router_probs.to_vec2()?;

        // For each token, find top-k experts
        let mut output_data = vec![Tensor::zeros((1, hidden), x.dtype(), x.device())?; num_tokens];

        for tok_idx in 0..num_tokens {
            let probs = &router_probs_vec[tok_idx];
            let mut indexed: Vec<(usize, f32)> =
                probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(self.num_experts_per_tok);

            // Renormalize selected expert weights
            let total: f32 = indexed.iter().map(|(_, p)| p).sum();

            let token = x_flat.narrow(0, tok_idx, 1)?; // (1, hidden)
            let mut tok_out = Tensor::zeros((1, hidden), x.dtype(), x.device())?;

            for (expert_idx, weight) in &indexed {
                let expert_out = self.experts[*expert_idx].forward(&token)?;
                tok_out = (tok_out + expert_out * (*weight as f64 / total as f64))?;
            }
            output_data[tok_idx] = tok_out;
        }

        let mut output = Tensor::cat(&output_data, 0)?; // (tokens, hidden)

        // Add shared expert contribution
        if let Some(ref shared) = self.shared_expert {
            output = (output + shared.forward(&x_flat)?)?;
        }

        output.reshape((b, seq_len, hidden))
    }
}

/// Enum dispatch for FFN variants.
pub enum FfnImpl {
    Gated(GatedFFN),
    MoE(MoEFFN),
}

impl FfnImpl {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gated(ffn) => ffn.forward(x),
            Self::MoE(moe) => moe.forward(x),
        }
    }
}
