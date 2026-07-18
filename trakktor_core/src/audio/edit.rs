//! Cutting decoded audio into new buffers, with short de-click fades.
//!
//! The editor copies sample ranges of the *original* decoded audio verbatim —
//! at its native rate, channel count, and sample format — so the result keeps
//! full quality (the detector's 16 kHz mono is used only to find the ranges).
//! The only samples it changes are the few milliseconds of linear fade at each
//! piece boundary: concatenating two non-adjacent ranges leaves an amplitude
//! step that clicks, and a fade to and from near-silence removes it. Because
//! the ranges are chosen to keep a little silence around speech, the fades land
//! in low-amplitude audio and stay inaudible.

#[cfg(test)]
mod tests;

use super::{decode::DecodedAudio, pipeline::NativeBuf};

/// Concatenates the given time ranges (seconds, on the original timeline) into
/// one new buffer with the source's native rate/layout/format.
#[must_use]
pub fn cut(
    audio: &DecodedAudio,
    ranges: &[(f64, f64)],
    fade_ms: u32,
) -> DecodedAudio {
    let spans = to_spans(audio, ranges);
    rebuild(
        audio,
        build(audio.samples(), &spans, fade_len(audio, fade_ms)),
    )
}

/// Writes each range to its own new buffer — one clip per range, aligned 1:1
/// with `ranges` (a range that rounds to nothing yields an empty clip).
#[must_use]
pub fn split(
    audio: &DecodedAudio,
    ranges: &[(f64, f64)],
    fade_ms: u32,
) -> Vec<DecodedAudio> {
    let fade = fade_len(audio, fade_ms);
    ranges
        .iter()
        .map(|&range| {
            let span = to_spans(audio, &[range]);
            rebuild(audio, build(audio.samples(), &span, fade))
        })
        .collect()
}

/// Wraps a freshly built sample buffer as a [`DecodedAudio`] sharing the
/// source's rate, channel layout, and bit depth.
fn rebuild(source: &DecodedAudio, samples: NativeBuf) -> DecodedAudio {
    DecodedAudio::from_parts(
        source.sample_rate(),
        source.channel_mask(),
        source.source_bits(),
        samples,
    )
}

/// Converts second ranges to half-open frame spans, rounding to the nearest
/// sample and clamping to the audio, dropping any empty span.
fn to_spans(
    audio: &DecodedAudio,
    ranges: &[(f64, f64)],
) -> Vec<(usize, usize)> {
    let rate = f64::from(audio.sample_rate());
    let frames = audio.frames() as f64;
    ranges
        .iter()
        .filter_map(|&(start, end)| {
            let a = (start * rate).round().clamp(0.0, frames) as usize;
            let b = (end * rate).round().clamp(0.0, frames) as usize;
            (b > a).then_some((a, b))
        })
        .collect()
}

/// Fade length in frames.
fn fade_len(audio: &DecodedAudio, fade_ms: u32) -> usize {
    fade_ms as usize * audio.sample_rate() as usize / 1000
}

/// Builds one planar buffer from the concatenated spans, per native format.
fn build(src: &NativeBuf, spans: &[(usize, usize)], fade: usize) -> NativeBuf {
    match src {
        NativeBuf::U8(p) => {
            NativeBuf::U8(build_planar(p, spans, fade, scale_u8))
        },
        NativeBuf::S16(p) => {
            NativeBuf::S16(build_planar(p, spans, fade, scale_i16))
        },
        NativeBuf::S32(p) => {
            NativeBuf::S32(build_planar(p, spans, fade, scale_i32))
        },
        NativeBuf::F32(p) => {
            NativeBuf::F32(build_planar(p, spans, fade, scale_f32))
        },
        NativeBuf::F64(p) => {
            NativeBuf::F64(build_planar(p, spans, fade, scale_f64))
        },
    }
}

/// Concatenates `spans` of every channel plane into a new buffer, applying a
/// `fade`-frame fade-in at each piece's head and fade-out at its tail. The fade
/// is clamped to half the piece so the two ramps never overlap.
fn build_planar<T: Copy>(
    planes: &[Vec<T>],
    spans: &[(usize, usize)],
    fade: usize,
    scale: impl Fn(T, f64) -> T,
) -> Vec<Vec<T>> {
    let mut out: Vec<Vec<T>> = vec![Vec::new(); planes.len()];
    for &(a, b) in spans {
        let piece = b - a;
        let ramp = fade.min(piece / 2);
        for (plane, dst) in planes.iter().zip(out.iter_mut()) {
            let base = dst.len();
            dst.extend_from_slice(&plane[a..b]);
            let seg = &mut dst[base..base + piece];
            for i in 0..ramp {
                // Linear gain in (0, 1): head ramps up, tail mirrors it down,
                // so both ends approach silence and the join does not click.
                let gain = (i as f64 + 0.5) / ramp as f64;
                seg[i] = scale(seg[i], gain);
                let tail = piece - 1 - i;
                seg[tail] = scale(seg[tail], gain);
            }
        }
    }
    out
}

/// WAV 8-bit is unsigned with the center at 128, so fade toward the center.
fn scale_u8(x: u8, gain: f64) -> u8 {
    (128.0 + (f64::from(x) - 128.0) * gain)
        .round()
        .clamp(0.0, 255.0) as u8
}

fn scale_i16(x: i16, gain: f64) -> i16 {
    (f64::from(x) * gain)
        .round()
        .clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16
}

fn scale_i32(x: i32, gain: f64) -> i32 {
    (x as f64 * gain)
        .round()
        .clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

fn scale_f32(x: f32, gain: f64) -> f32 { (f64::from(x) * gain) as f32 }

fn scale_f64(x: f64, gain: f64) -> f64 { x * gain }
