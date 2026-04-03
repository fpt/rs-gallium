use candle_core::{DType, Device, Result, Tensor};

/// Build a causal attention mask: (seq_len, total_len) where total_len = pos + seq_len.
/// Entries are 0.0 (attend) or -inf (block).
pub fn build_causal_mask(seq_len: usize, pos: usize, device: &Device) -> Result<Tensor> {
    let total_len = pos + seq_len;
    let mask = Tensor::zeros((seq_len, total_len), DType::F32, device)?;
    // For each query position i (0..seq_len), it can attend to positions 0..=(pos+i).
    // Mask out positions (pos+i+1)..total_len with -inf.
    if seq_len <= 1 {
        return Ok(mask);
    }
    let mut mask_data = vec![0.0f32; seq_len * total_len];
    for i in 0..seq_len {
        let query_pos = pos + i;
        for j in (query_pos + 1)..total_len {
            mask_data[i * total_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(mask_data, (seq_len, total_len), device)
}

/// Build a sliding-window + causal mask: each query attends to at most `window_size`
/// previous positions (inclusive of itself), with causal constraint.
pub fn build_sliding_window_mask(
    seq_len: usize,
    pos: usize,
    window_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let total_len = pos + seq_len;
    if seq_len <= 1 && total_len <= window_size {
        return Tensor::zeros((seq_len, total_len), DType::F32, device);
    }
    let mut mask_data = vec![0.0f32; seq_len * total_len];
    for i in 0..seq_len {
        let query_pos = pos + i;
        for j in 0..total_len {
            // Block if: future (causal) or too far in the past (sliding window)
            let is_future = j > query_pos;
            let is_outside_window = query_pos >= window_size && j < query_pos - window_size + 1;
            if is_future || is_outside_window {
                mask_data[i * total_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask_data, (seq_len, total_len), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_shape() {
        let device = Device::Cpu;
        let mask = build_causal_mask(4, 0, &device).unwrap();
        assert_eq!(mask.dims(), &[4, 4]);
    }

    #[test]
    fn test_causal_mask_values() {
        let device = Device::Cpu;
        let mask = build_causal_mask(3, 0, &device).unwrap();
        let data: Vec<Vec<f32>> = mask.to_vec2().unwrap();
        // Row 0: can attend to pos 0 only
        assert_eq!(data[0][0], 0.0);
        assert!(data[0][1].is_infinite());
        // Row 2: can attend to 0,1,2
        assert_eq!(data[2][0], 0.0);
        assert_eq!(data[2][1], 0.0);
        assert_eq!(data[2][2], 0.0);
    }

    #[test]
    fn test_sliding_window_mask() {
        let device = Device::Cpu;
        let mask = build_sliding_window_mask(4, 0, 2, &device).unwrap();
        let data: Vec<Vec<f32>> = mask.to_vec2().unwrap();
        // Row 0 (pos 0, window=2): attend to [0]
        assert_eq!(data[0][0], 0.0);
        // Row 2 (pos 2, window=2): attend to [1, 2], not [0]
        assert!(data[2][0].is_infinite());
        assert_eq!(data[2][1], 0.0);
        assert_eq!(data[2][2], 0.0);
        // Row 3 (pos 3, window=2): attend to [2, 3], not [0, 1]
        assert!(data[3][0].is_infinite());
        assert!(data[3][1].is_infinite());
        assert_eq!(data[3][2], 0.0);
        assert_eq!(data[3][3], 0.0);
    }
}
