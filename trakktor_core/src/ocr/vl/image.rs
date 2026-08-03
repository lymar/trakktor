//! Turning a picture into the patch tensor the vision tower expects.
//!
//! Four steps, all of them the published processor's: pick a size, resize to
//! it, normalize, cut into patches.
//!
//! **The size is not a scale factor but a rounding.** Both sides go to a
//! multiple of `patch_size × merge_size` = 28, and only if the resulting area
//! falls outside `[min_pixels, max_pixels]` is the picture scaled at all — down
//! by `√(hw / max)` with the sides rounded *down*, or up by `√(min / hw)` with
//! them rounded *up*. A crop of a text block therefore usually keeps its own
//! resolution, which is exactly what a reader of small type wants.
//!
//! **The resampling is Pillow's bicubic**, which is the same kernel as
//! Catmull-Rom (`a = −0.5`) with its support widened when the picture shrinks.
//! That is not the same thing as the fixed-point bilinear the classic detector
//! needs — the two engines of this domain preprocess differently on purpose,
//! each faithful to its own reference.

#[cfg(test)]
mod tests;

use image::{RgbImage, imageops::FilterType};

use super::config::ImageConfig;
use crate::ocr::error::OcrError;

/// A picture ready for the vision tower: patches laid out as
/// `[patches, channels, patch, patch]`, plus the grid they came from.
#[derive(Debug, Clone)]
pub struct Prepared {
    /// `patches × channels × patch_size × patch_size` values, row-major.
    pub pixels: Vec<f32>,
    /// `(t, h, w)`: patches along time, height and width. `t` is always 1 —
    /// this engine reads pictures, not video.
    pub grid: (usize, usize, usize),
}

impl Prepared {
    /// Patches in the grid.
    pub fn patches(&self) -> usize { self.grid.0 * self.grid.1 * self.grid.2 }

    /// How many places in the prompt this picture takes: the projector folds
    /// `merge × merge` patches into one token.
    pub fn tokens(&self, merge: usize) -> usize {
        self.patches() / (merge * merge)
    }
}

/// The largest ratio between the sides the processor accepts. Beyond it the
/// smart resize cannot satisfy both bounds and upstream refuses outright — a
/// sliver of a crop is a caller's mistake, not something to silently pad.
const MAX_ASPECT: f64 = 200.0;

/// Works out the size a picture is resized to.
///
/// Both sides come back as multiples of `factor`, and the area lands inside
/// `[min_pixels, max_pixels]` whenever the aspect ratio allows it.
pub fn smart_resize(
    height: u32,
    width: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32), OcrError> {
    // A picture thinner than one patch is grown until it is not, keeping its
    // proportions; this happens before the aspect-ratio check, as upstream.
    let (mut height, mut width) = (height as f64, width as f64);
    let factor = factor as f64;
    if height < factor {
        width = (width * factor / height).round();
        height = factor;
    }
    if width < factor {
        height = (height * factor / width).round();
        width = factor;
    }

    let (long, short) = if height > width {
        (height, width)
    } else {
        (width, height)
    };
    if long / short > MAX_ASPECT {
        return Err(OcrError::InvalidOptions(format!(
            "the picture is {long:.0}×{short:.0}, a ratio of {:.0}:1; the \
             model accepts at most {MAX_ASPECT:.0}:1",
            long / short
        )));
    }

    let mut bar_h = (height / factor).round() * factor;
    let mut bar_w = (width / factor).round() * factor;
    if bar_h * bar_w > max_pixels as f64 {
        let beta = (height * width / max_pixels as f64).sqrt();
        bar_h = factor.max((height / beta / factor).floor() * factor);
        bar_w = factor.max((width / beta / factor).floor() * factor);
    } else if bar_h * bar_w < min_pixels as f64 {
        let beta = (min_pixels as f64 / (height * width)).sqrt();
        bar_h = (height * beta / factor).ceil() * factor;
        bar_w = (width * beta / factor).ceil() * factor;
    }
    Ok((bar_h as u32, bar_w as u32))
}

/// Resizes, normalizes and patchifies one picture.
pub fn prepare(
    picture: &RgbImage,
    cfg: &ImageConfig,
) -> Result<Prepared, OcrError> {
    let (height, width) = smart_resize(
        picture.height(),
        picture.width(),
        cfg.factor(),
        cfg.min_pixels,
        cfg.max_pixels,
    )?;

    let resized = if height == picture.height() && width == picture.width() {
        std::borrow::Cow::Borrowed(picture)
    } else {
        std::borrow::Cow::Owned(image::imageops::resize(
            picture,
            width,
            height,
            FilterType::CatmullRom,
        ))
    };

    let patch = cfg.patch_size as usize;
    let grid_h = height as usize / patch;
    let grid_w = width as usize / patch;
    let stride = width as usize * 3;
    let raw = resized.as_raw();

    let per_patch = 3 * patch * patch;
    let mut pixels = vec![0f32; grid_h * grid_w * per_patch];
    for (index, chunk) in pixels.chunks_exact_mut(per_patch).enumerate() {
        let base_y = (index / grid_w) * patch;
        let base_x = (index % grid_w) * patch;
        let mut at = 0;
        for channel in 0..3usize {
            for dy in 0..patch {
                let row = (base_y + dy) * stride + base_x * 3 + channel;
                for dx in 0..patch {
                    let value = raw[row + dx * 3] as f32 * cfg.rescale_factor;
                    chunk[at] = (value - cfg.image_mean[channel]) /
                        cfg.image_std[channel];
                    at += 1;
                }
            }
        }
    }

    Ok(Prepared {
        pixels,
        grid: (1, grid_h, grid_w),
    })
}
