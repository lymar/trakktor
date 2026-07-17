//! Sample-format conversions of the PCM pipeline.
//!
//! Ports the conversion semantics of ffmpeg's libswresample
//! (`audioconvert.c`): input formats widen to the pipeline's working format
//! with exact scalings, and the final quantization to s16 follows the NEON
//! float→s16 kernel of the reference aarch64 build (saturating truncation to
//! Q31, then a rounding, saturating shift down to 16 bits).
//!
//! One deliberate deviation: the reference applies its NEON quantizer to the
//! first `len & !15` samples of every internal chunk and a `lrintf`-based
//! scalar tail to the remainder, so a handful of samples per stream see
//! round-half-to-even instead. This crate quantizes every sample with the
//! same (NEON) rule — the results differ only for values landing exactly on
//! a half-LSB boundary inside those tails, which full-entropy audio does not
//! produce in practice.

/// u8 (offset binary) → s16: exact, `(x − 0x80) << 8`.
pub fn u8_to_s16(x: u8) -> i16 { ((i32::from(x) - 0x80) << 8) as i16 }

/// s16 → f32: exact scaling by 2⁻¹⁵ into `[-1, 1)`.
pub fn s16_to_f32(x: i16) -> f32 { f32::from(x) * (1.0 / 32768.0) }

/// s32 → f32: round to nearest float, then exact scaling by 2⁻³¹.
///
/// 24-bit sources are represented as `s32 << 8` upstream, matching the
/// reference pcm decoders, so they take this path.
pub fn s32_to_f32(x: i32) -> f32 { (x as f32) * (1.0 / 2_147_483_648.0) }

/// f32 → s16: the quantizer of the reference pipeline.
///
/// Semantics of the aarch64 kernel (`fcvtzs #31` + `sqrshrn #16`): saturating
/// truncation toward zero to Q31 fixed point, then `(q31 + 2¹⁵) >> 16` with
/// saturation. NaN maps to 0. Both intermediate steps are exact here: the
/// f32→f64 widening and the power-of-two scaling introduce no rounding.
pub fn f32_to_s16(x: f32) -> i16 {
    // `as` from f64 to i32 truncates toward zero, saturates, and maps NaN to
    // zero — the exact `fcvtzs` behavior.
    let q31 = (f64::from(x) * 2_147_483_648.0) as i32;
    let v = (i64::from(q31) + (1 << 15)) >> 16;
    v.clamp(-32768, 32767) as i16
}

/// f32 → s16 via the reference C path: `av_clip_int16(lrintf(x * 32768.0f))`.
///
/// The multiply happens in f32 (it may round), then round-half-to-even.
/// Kept for boundary tests documenting the divergence; the pipeline uses
/// [`f32_to_s16`].
#[cfg(test)]
pub fn f32_to_s16_lrint(x: f32) -> i16 {
    let y = (x * 32768.0_f32).round_ties_even();
    (y as i32).clamp(-32768, 32767) as i16
}

/// f64 → s16: the reference double path, `av_clip_int16(lrint(x * 32768.0))`.
pub fn f64_to_s16(x: f64) -> i16 {
    let y = (x * 32768.0).round_ties_even();
    (y as i64).clamp(-32768, 32767) as i16
}

#[cfg(test)]
mod tests;
