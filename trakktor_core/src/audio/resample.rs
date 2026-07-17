//! Rational polyphase resampler.
//!
//! A faithful reimplementation of the swresample engine of ffmpeg
//! (`libswresample/resample.c` and friends) with its default options:
//! Kaiser-windowed sinc bank (β = 9, cutoff 0.97, `filter_size` 32,
//! 1024 phases capped to the exact rational phase count), integer phase
//! stepping, mirrored stream edges, and the summation order of the aarch64
//! kernels of the reference build — expressed portably (no assembly): the
//! four-lane fused-multiply-add structure is reproduced with scalar
//! `mul_add`, which yields identical bits on any platform.
//!
//! The engine is generic over the working element: `f32` (the main path),
//! `f64`, and `i16` (integer path with a Q15 filter bank).

use std::f64::consts::PI;

use super::error::AudioError;

mod bessel;

/// `filter_size` — the default half-quality knob of swresample.
const FILTER_SIZE: usize = 32;
/// `1 << phase_shift` — the default phase count before rational reduction.
const PHASE_COUNT_DEFAULT: i64 = 1 << 10;
/// Default Kaiser window β.
const KAISER_BETA: f64 = 9.0;
/// Default cutoff ratio (applied when downsampling).
const CUTOFF: f64 = 0.97;

/// A working element of the resampler (`f32`, `f64`, or `i16`).
pub(crate) trait Element: Copy + Default + 'static {
    /// The fixed-point scale of the filter bank (`1 << filter_shift`).
    const FILTER_SCALE: f64;

    /// Quantizes one double-precision tap (already divided by the bank
    /// norm and multiplied by [`Self::FILTER_SCALE`]).
    fn tap(value: f64) -> Self;

    /// One output sample of the `common` kernel (integer phase step).
    fn conv_common(src: &[Self], taps: &[Self]) -> Self;

    /// One output sample of the `linear` kernel (interpolates between two
    /// adjacent phases; used when the rational reduction cannot make the
    /// phase step exact).
    fn conv_linear(
        src: &[Self],
        taps: &[Self],
        taps_next: &[Self],
        frac: i64,
        src_incr: i64,
    ) -> Self;
}

impl Element for f32 {
    const FILTER_SCALE: f64 = 1.0;

    fn tap(value: f64) -> Self { value as f32 }

    /// The aarch64 kernel of the reference: eight taps per iteration into
    /// four lane accumulators (two fused multiply-adds per lane), a pairwise
    /// lane reduction, then a scalar fused tail.
    fn conv_common(src: &[Self], taps: &[Self]) -> Self {
        let len = taps.len();
        let x8 = len & !7;
        let x4 = len & !3;
        let mut lanes = [0.0f32; 4];
        let mut i = 0;
        if x8 >= 8 {
            while i < x8 {
                for j in 0..4 {
                    lanes[j] = src[i + j].mul_add(taps[i + j], lanes[j]);
                }
                for j in 0..4 {
                    lanes[j] =
                        src[i + 4 + j].mul_add(taps[i + 4 + j], lanes[j]);
                }
                i += 8;
            }
        } else if x4 >= 4 {
            while i < x4 {
                for j in 0..4 {
                    lanes[j] = src[i + j].mul_add(taps[i + j], lanes[j]);
                }
                i += 4;
            }
        }
        let mut val = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        while i < len {
            val = src[i].mul_add(taps[i], val);
            i += 1;
        }
        val
    }

    /// The C `resample_linear` shape: two sequential accumulators, then a
    /// double-precision interpolation between the phases.
    fn conv_linear(
        src: &[Self],
        taps: &[Self],
        taps_next: &[Self],
        frac: i64,
        src_incr: i64,
    ) -> Self {
        let inv_src_incr = 1.0 / src_incr as f64;
        let mut val = 0.0f32;
        let mut v2 = 0.0f32;
        for i in 0..taps.len() {
            val = src[i].mul_add(taps[i], val);
            v2 = src[i].mul_add(taps_next[i], v2);
        }
        (f64::from(val) + f64::from(v2 - val) * inv_src_incr * frac as f64)
            as f32
    }
}

impl Element for f64 {
    const FILTER_SCALE: f64 = 1.0;

    fn tap(value: f64) -> Self { value }

    /// The C `resample_common` shape (no aarch64 kernel exists for doubles):
    /// even/odd interleaved accumulators, summed at the end.
    fn conv_common(src: &[Self], taps: &[Self]) -> Self {
        let len = taps.len();
        let mut val = 0.0f64;
        let mut val2 = 0.0f64;
        let mut i = 0;
        while i + 1 < len {
            val = src[i].mul_add(taps[i], val);
            val2 = src[i + 1].mul_add(taps[i + 1], val2);
            i += 2;
        }
        if i < len {
            val = src[i].mul_add(taps[i], val);
        }
        val + val2
    }

    fn conv_linear(
        src: &[Self],
        taps: &[Self],
        taps_next: &[Self],
        frac: i64,
        src_incr: i64,
    ) -> Self {
        let inv_src_incr = 1.0 / src_incr as f64;
        let mut val = 0.0f64;
        let mut v2 = 0.0f64;
        for i in 0..taps.len() {
            val = src[i].mul_add(taps[i], val);
            v2 = src[i].mul_add(taps_next[i], v2);
        }
        let x = (v2 - val) * inv_src_incr;
        x.mul_add(frac as f64, val)
    }
}

impl Element for i16 {
    const FILTER_SCALE: f64 = 32768.0; // 1 << 15

    /// `av_clip_int16(lrintf(tap * scale / norm))` — note the argument is
    /// narrowed to f32 first (that is what `lrintf` does in C).
    fn tap(value: f64) -> Self {
        let f = value as f32;
        (f.round_ties_even() as i32).clamp(-32768, 32767) as i16
    }

    /// The aarch64 s16 kernel: widening multiply-accumulate into four i32
    /// lanes (modular arithmetic), pairwise reduction, scalar tail, then a
    /// rounding shift with saturation.
    fn conv_common(src: &[Self], taps: &[Self]) -> Self {
        let len = taps.len();
        let x8 = len & !7;
        let x4 = len & !3;
        let mut lanes = [0i32; 4];
        let mut i = 0;
        if x8 >= 8 {
            while i < x8 {
                for j in 0..4 {
                    lanes[j] = lanes[j].wrapping_add(
                        i32::from(src[i + j]) * i32::from(taps[i + j]),
                    );
                }
                for j in 0..4 {
                    lanes[j] = lanes[j].wrapping_add(
                        i32::from(src[i + 4 + j]) * i32::from(taps[i + 4 + j]),
                    );
                }
                i += 8;
            }
        } else if x4 >= 4 {
            while i < x4 {
                for j in 0..4 {
                    lanes[j] = lanes[j].wrapping_add(
                        i32::from(src[i + j]) * i32::from(taps[i + j]),
                    );
                }
                i += 4;
            }
        }
        let mut val = lanes[0]
            .wrapping_add(lanes[1])
            .wrapping_add(lanes[2].wrapping_add(lanes[3]));
        while i < len {
            val = val.wrapping_add(i32::from(src[i]) * i32::from(taps[i]));
            i += 1;
        }
        (((val.wrapping_add(1 << 14)) >> 15).clamp(-32768, 32767)) as i16
    }

    /// The C s16 `resample_linear`: unsigned modular accumulators with the
    /// rounding offset folded in, 64-bit interpolation, truncating division.
    fn conv_linear(
        src: &[Self],
        taps: &[Self],
        taps_next: &[Self],
        frac: i64,
        src_incr: i64,
    ) -> Self {
        let mut val: u32 = 1 << 14;
        let mut v2: u32 = 1 << 14;
        for i in 0..taps.len() {
            let s = i32::from(src[i]);
            val = val.wrapping_add((s * i32::from(taps[i])) as u32);
            v2 = v2.wrapping_add((s * i32::from(taps_next[i])) as u32);
        }
        let diff = v2.wrapping_sub(val) as i32;
        let corr = i64::from(diff) * frac / src_incr;
        let val = val.wrapping_add(corr as u32);
        ((val as i32) >> 15).clamp(-32768, 32767) as i16
    }
}

/// The immutable geometry of a rate conversion: the polyphase filter bank
/// plus the integer stepping constants.
struct FilterBank<T> {
    /// Taps, `(phase_count + 1)` rows of `filter_alloc` elements; the extra
    /// row supports the linear kernel's phase + 1 access.
    bank: Vec<T>,
    filter_length: usize,
    filter_alloc: usize,
    phase_count: i64,
    src_incr: i64,
    dst_incr_div: i64,
    dst_incr_mod: i64,
    /// Initial phase index (negative: the filter is centered).
    initial_index: i64,
    /// Whether the linear kernel is in effect (inexact rational step).
    linear: bool,
}

/// Reduces `num/den` to lowest terms.
fn reduce(num: i64, den: i64) -> (i64, i64) {
    fn gcd(mut a: i64, mut b: i64) -> i64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    let g = gcd(num, den);
    (num / g, den / g)
}

impl<T: Element> FilterBank<T> {
    /// Mirrors `resample_init` of the reference with its default options.
    fn new(out_rate: u32, in_rate: u32) -> Result<Self, AudioError> {
        if out_rate == 0 || in_rate == 0 {
            return Err(AudioError::Decode("zero sample rate".into()));
        }
        let out_rate = i64::from(out_rate);
        let in_rate = i64::from(in_rate);

        let factor = (out_rate as f64 * CUTOFF / in_rate as f64).min(1.0);
        let mut filter_length =
            ((FILTER_SIZE as f64 / factor).ceil() as usize).max(1);
        if filter_length > 1 {
            filter_length = filter_length.next_multiple_of(2);
        }

        let mut phase_count = PHASE_COUNT_DEFAULT;
        // exact_rational (default on): shrink the phase bank to the exact
        // rational phase count when it fits.
        let (num, _den) = reduce(out_rate, in_rate);
        if num <= PHASE_COUNT_DEFAULT {
            phase_count = num;
        }

        let filter_alloc = filter_length.next_multiple_of(8);
        let mut bank =
            vec![T::default(); filter_alloc * (phase_count as usize + 1)];
        build_filter::<T>(
            &mut bank,
            factor,
            filter_length,
            filter_alloc,
            phase_count as usize,
            KAISER_BETA,
        );
        // The guard row (used by the linear kernel at phase_count − 1):
        // row `phase_count` is row 0 rotated right by one element.
        {
            let row0: Vec<T> = bank[..filter_alloc - 1].to_vec();
            let last = bank[filter_alloc - 1];
            let guard = filter_alloc * phase_count as usize;
            bank[guard] = last;
            bank[guard + 1..guard + filter_alloc].copy_from_slice(&row0);
        }

        let (mut src_incr, mut dst_incr) =
            reduce(out_rate, in_rate * phase_count);
        while dst_incr < (1 << 20) && src_incr < (1 << 20) {
            dst_incr *= 2;
            src_incr *= 2;
        }
        let dst_incr_div = dst_incr / src_incr;
        let dst_incr_mod = dst_incr % src_incr;

        Ok(Self {
            bank,
            filter_length,
            filter_alloc,
            phase_count,
            src_incr,
            dst_incr_div,
            dst_incr_mod,
            initial_index: -phase_count * ((filter_length as i64 - 1) / 2),
            // linear_interp defaults on but only takes effect when the
            // phase step is inexact (see `resample.c`: the kernel choice
            // tests `frac || dst_incr_mod`, and frac stays zero otherwise).
            linear: dst_incr_mod != 0,
        })
    }

    fn row(&self, phase: i64) -> &[T] {
        let start = self.filter_alloc * phase as usize;
        &self.bank[start..start + self.filter_length]
    }

    /// `dst_incr` is kept via its split parts; the full value re-derives as
    /// `div * src_incr + mod`.
    fn dst_incr(&self) -> i64 {
        self.dst_incr_div * self.src_incr + self.dst_incr_mod
    }
}

/// Literal port of `build_filter` (Kaiser branch) from the reference.
fn build_filter<T: Element>(
    bank: &mut [T],
    mut factor: f64,
    tap_count: usize,
    alloc: usize,
    phase_count: usize,
    beta: f64,
) {
    let ph_nb = if phase_count % 2 != 0 {
        phase_count
    } else {
        phase_count / 2 + 1
    };
    let center = (tap_count - 1) / 2;
    let mut tab = vec![0.0f64; tap_count];
    let mut norm = 0.0f64;

    // When upsampling the sinc needs no stretching, and a sine lookup with
    // alternating signs replaces per-tap sines (the order of floating-point
    // operations matters, so the trick is reproduced as is).
    if factor > 1.0 {
        factor = 1.0;
    }
    let mut sin_lut = vec![0.0f64; ph_nb.max(1)];
    if factor == 1.0 {
        for (ph, slot) in sin_lut.iter_mut().enumerate() {
            *slot = (PI * ph as f64 / phase_count as f64).sin() *
                (if center % 2 == 1 { 1.0 } else { -1.0 });
        }
    }
    for ph in 0..ph_nb {
        let mut s = sin_lut[ph];
        for i in 0..tap_count {
            let x = PI *
                ((i as f64 - center as f64) - ph as f64 / phase_count as f64) *
                factor;
            let mut y = if x == 0.0 {
                1.0
            } else if factor == 1.0 {
                s / x
            } else {
                x.sin() / x
            };
            // Kaiser window.
            let w = 2.0 * x / (factor * tap_count as f64 * PI);
            y *= bessel::bessel_i0(beta * (1.0 - w * w).max(0.0).sqrt());

            tab[i] = y;
            s = -s;
            if ph == 0 {
                norm += y;
            }
        }

        // Quantize this phase, normalized so a constant signal stays put.
        for i in 0..tap_count {
            bank[ph * alloc + i] = T::tap(tab[i] * T::FILTER_SCALE / norm);
        }
        // Mirror the symmetric upper half of the bank (even phase counts).
        if phase_count % 2 == 0 {
            for i in 0..tap_count {
                bank[(phase_count - ph) * alloc + tap_count - 1 - i] =
                    bank[ph * alloc + i];
            }
        }
    }
}

/// Streaming rational resampler over planar channels.
///
/// Push samples with [`Resampler::feed`], then drain the tail once with
/// [`Resampler::finish`]. The output is identical for any chunking of the
/// input. Stream edges are mirrored, matching the reference engine: the
/// filter history before the first sample reflects the signal start (and
/// the first output cannot be produced until `filter_length + 1` samples
/// arrived), and at end of stream the tail is extended by a mirrored
/// suffix once.
pub(crate) struct Resampler<T: Element> {
    fb: FilterBank<T>,
    channels: usize,
    /// Pre-priming accumulator, capped at `filter_length + 1` per channel.
    staged: Vec<Vec<T>>,
    /// Working history per channel; `hist[ch][pos..]` is unconsumed.
    hist: Vec<Vec<T>>,
    pos: usize,
    /// Current phase index (into the bank) and fractional step remainder.
    index: i64,
    frac: i64,
    primed: bool,
    flushed: bool,
}

impl<T: Element> Resampler<T> {
    pub fn new(
        out_rate: u32,
        in_rate: u32,
        channels: usize,
    ) -> Result<Self, AudioError> {
        assert!(channels > 0);
        assert_ne!(out_rate, in_rate, "resampler needs a rate change");
        let fb = FilterBank::new(out_rate, in_rate)?;
        let index = fb.initial_index;
        Ok(Self {
            fb,
            channels,
            staged: vec![Vec::new(); channels],
            hist: vec![Vec::new(); channels],
            pos: 0,
            index,
            frac: 0,
            primed: false,
            flushed: false,
        })
    }

    /// Pushes one planar chunk (`chunk[ch]` per channel, equal lengths) and
    /// returns whatever output it unlocks.
    pub fn feed(&mut self, chunk: &[&[T]]) -> Vec<Vec<T>> {
        assert_eq!(chunk.len(), self.channels);
        let mut offset = 0;
        if !self.primed {
            let fl = self.fb.filter_length;
            let need = (fl + 1).saturating_sub(self.staged[0].len());
            let take = need.min(chunk[0].len());
            for (stage, data) in self.staged.iter_mut().zip(chunk) {
                stage.extend_from_slice(&data[..take]);
            }
            offset = take;
            if self.staged[0].len() == fl + 1 {
                self.prime();
            } else {
                return vec![Vec::new(); self.channels];
            }
        }
        for (hist, data) in self.hist.iter_mut().zip(chunk) {
            hist.extend_from_slice(&data[offset..]);
        }
        self.produce()
    }

    /// Flushes the stream: mirrors the tail once (as the reference does at
    /// end of input) and returns the remaining output.
    pub fn finish(&mut self) -> Vec<Vec<T>> {
        if self.flushed {
            return vec![Vec::new(); self.channels];
        }
        self.flushed = true;
        let fl = self.fb.filter_length;
        if !self.primed {
            let len = self.staged[0].len();
            let reflection = (len.min(fl) + 1) / 2;
            for stage in &mut self.staged {
                for j in 0..reflection {
                    let v = stage[len - 1 - j];
                    stage.push(v);
                }
            }
            if self.staged[0].len() >= fl + 1 {
                self.prime();
            } else {
                // Too short to ever produce output — as in the reference.
                return vec![Vec::new(); self.channels];
            }
        } else {
            let avail = self.hist[0].len() - self.pos;
            let reflection = (avail.min(fl) + 1) / 2;
            for hist in &mut self.hist {
                let len = hist.len();
                for j in 0..reflection {
                    let v = hist[len - 1 - j];
                    hist.push(v);
                }
            }
        }
        self.produce()
    }

    /// Builds the mirrored stream head from the staged samples and moves
    /// them into the working history (`invert_initial_buffer`).
    fn prime(&mut self) {
        let fl = self.fb.filter_length;
        let center = (fl - 1) / 2;
        for (stage, hist) in self.staged.iter_mut().zip(&mut self.hist) {
            debug_assert!(stage.len() >= fl + 1);
            hist.reserve(fl + stage.len());
            // x[-n] = x[n]: the head mirrors samples 1..=filter_length.
            for n in (1..=fl).rev() {
                hist.push(stage[n]);
            }
            hist.extend_from_slice(stage);
            stage.clear();
        }
        // The kernel starts `center` samples before x[0] at phase 0.
        self.pos = fl - center;
        self.index += self.fb.phase_count * center as i64;
        debug_assert_eq!(self.index, 0);
        self.primed = true;
    }

    /// Produces every output sample the available history allows
    /// (`multiple_resample` with an unbounded destination).
    fn produce(&mut self) -> Vec<Vec<T>> {
        let fb = &self.fb;
        let avail = (self.hist[0].len() - self.pos) as i64;
        let end_index = (1 + avail - fb.filter_length as i64) * fb.phase_count;
        let delta_frac = (end_index - self.index) * fb.src_incr - self.frac;
        let dst_incr = fb.dst_incr();
        let n = ((delta_frac + dst_incr - 1) / dst_incr).max(0) as usize;

        let mut out = Vec::with_capacity(self.channels);
        let mut committed = None;
        for hist in &self.hist {
            let mut index = self.index;
            let mut frac = self.frac;
            let mut sample_index = self.pos;
            while index >= fb.phase_count {
                sample_index += 1;
                index -= fb.phase_count;
            }
            let mut ch_out = Vec::with_capacity(n);
            for _ in 0..n {
                let src = &hist[sample_index..sample_index + fb.filter_length];
                let sample = if fb.linear {
                    T::conv_linear(
                        src,
                        fb.row(index),
                        fb.row(index + 1),
                        frac,
                        fb.src_incr,
                    )
                } else {
                    T::conv_common(src, fb.row(index))
                };
                ch_out.push(sample);

                frac += fb.dst_incr_mod;
                index += fb.dst_incr_div;
                if frac >= fb.src_incr {
                    frac -= fb.src_incr;
                    index += 1;
                }
                while index >= fb.phase_count {
                    sample_index += 1;
                    index -= fb.phase_count;
                }
            }
            committed = Some((index, frac, sample_index));
            out.push(ch_out);
        }
        if let Some((index, frac, pos)) = committed {
            self.index = index;
            self.frac = frac;
            self.pos = pos;
        }
        // Drop consumed history once it grows past a block.
        if self.pos > 1 << 14 {
            for hist in &mut self.hist {
                hist.drain(..self.pos);
            }
            self.pos = 0;
        }
        out
    }
}

#[cfg(test)]
mod tests;
