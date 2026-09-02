//! Gemma 4 image preprocessing — raw image bytes → the `pixel_values` /
//! `pixel_position_ids` the vision tower ([`crate::gemma4_vision`]) expects.
//!
//! Ported from transformers `Gemma4ImageProcessor`
//! (`image_processing_gemma4.py`): aspect-ratio-preserving resize into a patch
//! budget, rescale to `[0, 1]` (no mean/std — `do_normalize=False`, the tower
//! does the `2x-1` shift itself), SigLIP2-style patchification, and per-patch
//! `(x, y)` position ids.
//!
//! v1 scope: **one image, one tile, no pan-and-scan, no padding.** The reference
//! pads every image's patch grid to `max_soft_tokens * pooling_kernel_size^2`
//! and strips the padding back off after pooling; for a single image that is a
//! no-op, so `process` returns exactly the real patch grid and the vision
//! tower's `output_len = num_patches / pooling_kernel_size^2` lands on the real
//! soft-token count with no zero rows.

use candle_core::{DType, Device, Result, Tensor};

use crate::gemma4_vision::Gemma4VisionConfig;

/// `_SUPPORTED_SOFT_TOKENS` from the reference — `max_soft_tokens` must be one
/// of these.
const SUPPORTED_SOFT_TOKENS: [usize; 5] = [70, 140, 280, 560, 1120];

/// The Gemma 4 image processor: turns a decoded RGB image into the patch tensor
/// and position ids the vision tower consumes.
pub struct Gemma4ImageProcessor {
    patch_size: usize,
    pooling_kernel_size: usize,
    max_soft_tokens: usize,
}

/// What [`Gemma4ImageProcessor::process`] produces for one image.
pub struct ProcessedImage {
    /// `[1, num_patches, 3 * patch_size^2]` f32 in `[0, 1]`.
    pub pixel_values: Tensor,
    /// `[1, num_patches, 2]` i64 — `(x_patch, y_patch)` per patch, row-major.
    pub pixel_position_ids: Tensor,
    /// `num_patches / pooling_kernel_size^2` — how many `<image_soft_token>`
    /// placeholders the prompt must carry for this image.
    pub num_soft_tokens: usize,
}

impl Gemma4ImageProcessor {
    pub fn from_config(cfg: &Gemma4VisionConfig) -> Self {
        Self {
            patch_size: cfg.patch_size,
            pooling_kernel_size: cfg.pooling_kernel_size,
            max_soft_tokens: cfg.default_output_length,
        }
    }

    /// Largest patch budget: `max_soft_tokens * pooling_kernel_size^2`.
    fn max_patches(&self) -> usize {
        self.max_soft_tokens * self.pooling_kernel_size * self.pooling_kernel_size
    }

    /// Decode `bytes` (PNG/JPEG/WebP/…) and preprocess. `image` sniffs the
    /// format, so the caller does not decode.
    pub fn process(&self, bytes: &[u8], device: &Device) -> Result<ProcessedImage> {
        if !SUPPORTED_SOFT_TOKENS.contains(&self.max_soft_tokens) {
            candle_core::bail!(
                "Gemma 4 max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS:?}, got {}",
                self.max_soft_tokens
            );
        }
        let img = image::load_from_memory(bytes)
            .map_err(|e| candle_core::Error::Msg(format!("image decode failed: {e}")))?
            .to_rgb8();
        self.process_rgb(&img, device)
    }

    /// Preprocess an already-decoded RGB image. Split out for testing.
    pub fn process_rgb(&self, img: &image::RgbImage, device: &Device) -> Result<ProcessedImage> {
        let (w0, h0) = (img.width() as usize, img.height() as usize);
        let (target_h, target_w) = self.aspect_ratio_preserving_size(h0, w0)?;

        // Bicubic + antialias in the reference (`tvF.resize(..., BICUBIC,
        // antialias=True)`). CatmullRom is `image`'s bicubic-family filter with
        // the same windowed-sinc shape; not bit-identical, close enough for a
        // caption. This is the main source of numeric drift vs. transformers.
        let resized = if target_w == w0 && target_h == h0 {
            img.clone()
        } else {
            image::imageops::resize(
                img,
                target_w as u32,
                target_h as u32,
                image::imageops::FilterType::CatmullRom,
            )
        };

        let ps = self.patch_size;
        let npw = target_w / ps;
        let nph = target_h / ps;
        let num_patches = nph * npw;
        let patch_len = 3 * ps * ps;

        // Patchify exactly as SigLIP2's `convert_image_to_patches`:
        //   reshape(C, nph, ps, npw, ps).permute(1, 3, 2, 4, 0).reshape(nph*npw, ps*ps*C)
        // → patch index = pr*npw + pc; within a patch the flat order is
        //   (row, col, channel) with channel innermost (RGB interleaved).
        let mut pixels = vec![0f32; num_patches * patch_len];
        for pr in 0..nph {
            for pc in 0..npw {
                let patch_base = (pr * npw + pc) * patch_len;
                for iy in 0..ps {
                    for ix in 0..ps {
                        let px = resized.get_pixel((pc * ps + ix) as u32, (pr * ps + iy) as u32);
                        let o = patch_base + iy * (ps * 3) + ix * 3;
                        pixels[o] = px[0] as f32 / 255.0;
                        pixels[o + 1] = px[1] as f32 / 255.0;
                        pixels[o + 2] = px[2] as f32 / 255.0;
                    }
                }
            }
        }

        // Position ids: patch (pc, pr) at row-major index pr*npw+pc carries (x=pc, y=pr).
        let mut positions = vec![0i64; num_patches * 2];
        for pr in 0..nph {
            for pc in 0..npw {
                let i = (pr * npw + pc) * 2;
                positions[i] = pc as i64;
                positions[i + 1] = pr as i64;
            }
        }

        let pixel_values = Tensor::from_vec(pixels, (1, num_patches, patch_len), device)?;
        let pixel_position_ids =
            Tensor::from_vec(positions, (1, num_patches, 2), device)?.to_dtype(DType::I64)?;

        Ok(ProcessedImage {
            pixel_values,
            pixel_position_ids,
            num_soft_tokens: num_patches / (self.pooling_kernel_size * self.pooling_kernel_size),
        })
    }

    /// Port of `get_aspect_ratio_preserving_size`. Returns `(height, width)`, both
    /// multiples of `pooling_kernel_size * patch_size`, producing at most
    /// `max_patches` patches.
    fn aspect_ratio_preserving_size(&self, height: usize, width: usize) -> Result<(usize, usize)> {
        if height == 0 || width == 0 {
            candle_core::bail!("image has a zero dimension ({height}x{width})");
        }
        let ps = self.patch_size;
        let side_mult = self.pooling_kernel_size * ps;
        let total_px = (height * width) as f64;
        let target_px = (self.max_patches() * ps * ps) as f64;
        let factor = (target_px / total_px).sqrt();

        let floor_to = |v: f64| -> usize { ((v / side_mult as f64).floor() as usize) * side_mult };
        let mut target_h = floor_to(factor * height as f64);
        let mut target_w = floor_to(factor * width as f64);

        if target_h == 0 && target_w == 0 {
            candle_core::bail!("image {height}x{width} rounds to 0x0 at side multiple {side_mult}");
        }
        let max_side = (self.max_patches() / (self.pooling_kernel_size * self.pooling_kernel_size))
            * side_mult;
        if target_h == 0 {
            target_h = side_mult;
            target_w = ((width / height) * side_mult).min(max_side).max(side_mult);
        } else if target_w == 0 {
            target_w = side_mult;
            target_h = ((height / width) * side_mult).min(max_side).max(side_mult);
        }

        if target_h * target_w > self.max_patches() * ps * ps {
            candle_core::bail!(
                "resizing {height}x{width} to {target_h}x{target_w} exceeds the patch budget"
            );
        }
        Ok((target_h, target_w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Gemma4VisionConfig {
        // Only the fields the processor reads matter here.
        serde_json::from_value(serde_json::json!({
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 16,
            "num_attention_heads": 12,
            "rms_norm_eps": 1e-6,
            "patch_size": 16,
            "pooling_kernel_size": 3,
            "position_embedding_size": 10240,
            "default_output_length": 280
        }))
        .unwrap()
    }

    fn solid(w: u32, h: u32) -> image::RgbImage {
        image::RgbImage::from_pixel(w, h, image::Rgb([128, 64, 32]))
    }

    #[test]
    fn grid_is_a_multiple_of_the_pool_kernel_and_within_budget() {
        let p = Gemma4ImageProcessor::from_config(&cfg());
        for (w, h) in [(640, 480), (1920, 1080), (480, 1200), (37, 4000), (16, 16)] {
            let out = p.process_rgb(&solid(w, h), &Device::Cpu).unwrap();
            let (_, np, plen) = out.pixel_values.dims3().unwrap();
            assert_eq!(plen, 3 * 16 * 16);
            assert_eq!(np % (3 * 3), 0, "{w}x{h}: patch count not a multiple of 9");
            assert!(np <= 280 * 9, "{w}x{h}: {np} patches over budget");
            assert_eq!(out.num_soft_tokens, np / 9);
            assert_eq!(out.pixel_position_ids.dims3().unwrap(), (1, np, 2));
        }
    }

    #[test]
    fn pixels_are_rescaled_to_unit_range() {
        let p = Gemma4ImageProcessor::from_config(&cfg());
        let out = p.process_rgb(&solid(640, 480), &Device::Cpu).unwrap();
        let v = out
            .pixel_values
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(v.iter().all(|x| (0.0..=1.0).contains(x)));
        // 128/255 for the red channel of a solid image.
        assert!((v[0] - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn position_ids_span_the_patch_grid() {
        let p = Gemma4ImageProcessor::from_config(&cfg());
        let out = p.process_rgb(&solid(960, 480), &Device::Cpu).unwrap();
        let (_, np, _) = out.pixel_values.dims3().unwrap();
        let ids = out
            .pixel_position_ids
            .reshape((np, 2))
            .unwrap()
            .to_vec2::<i64>()
            .unwrap();
        let max_x = ids.iter().map(|p| p[0]).max().unwrap();
        let max_y = ids.iter().map(|p| p[1]).max().unwrap();
        // Row-major: index 0 is (0,0); the last is (max_x, max_y).
        assert_eq!(ids[0], vec![0, 0]);
        assert_eq!(ids[np - 1], vec![max_x, max_y]);
        assert_eq!((max_x + 1) * (max_y + 1), np as i64);
    }
}
