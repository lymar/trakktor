//! Channel mixing (the rematrix stage).
//!
//! Ports the automatic mixing-matrix construction and application of ffmpeg's
//! libswresample (`rematrix.c`) with its defaults: center/surround mix levels
//! of √½, an LFE level of 0 (the LFE channel is dropped on downmix), no
//! matrix encoding, and row normalization capped at 1.0 for integer outputs.
//! Coefficients are built in double precision; the per-format application
//! kernels mirror the reference (`rematrix_template.c`).

use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};

use super::error::AudioError;

// Channel ids follow the reference bit positions.
const FRONT_LEFT: usize = 0;
const FRONT_RIGHT: usize = 1;
const FRONT_CENTER: usize = 2;
const LOW_FREQUENCY: usize = 3;
const BACK_LEFT: usize = 4;
const BACK_RIGHT: usize = 5;
const FRONT_LEFT_OF_CENTER: usize = 6;
const FRONT_RIGHT_OF_CENTER: usize = 7;
const BACK_CENTER: usize = 8;
const SIDE_LEFT: usize = 9;
const SIDE_RIGHT: usize = 10;
const TOP_FRONT_LEFT: usize = 12;
const TOP_FRONT_CENTER: usize = 13;
const TOP_FRONT_RIGHT: usize = 14;
/// Channels with distribution rules live below this id; the rest can only
/// pass through identically.
const NUM_NAMED_CHANNELS: usize = 15;

/// Default center and surround mix levels (−3 dB).
const CENTER_MIX_LEVEL: f64 = FRAC_1_SQRT_2;
const SURROUND_MIX_LEVEL: f64 = FRAC_1_SQRT_2;
/// Default LFE mix level: the LFE channel is dropped.
const LFE_MIX_LEVEL: f64 = 0.0;

/// A native-order channel layout: bit `i` set means channel id `i` is
/// present; data channels are ordered by ascending id.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ChannelLayout(pub u64);

impl ChannelLayout {
    pub const MONO: Self = Self(1 << FRONT_CENTER);
    pub const STEREO: Self = Self(1 << FRONT_LEFT | 1 << FRONT_RIGHT);

    /// The default layout for a channel count, as the reference guesses for
    /// streams that do not declare one (`av_channel_layout_default`).
    pub fn default_for(channels: usize) -> Option<Self> {
        let mask: u64 = match channels {
            1 => 1 << FRONT_CENTER,
            2 => Self::STEREO.0,
            // 2.1
            3 => Self::STEREO.0 | 1 << LOW_FREQUENCY,
            // 4.0
            4 => Self::STEREO.0 | 1 << FRONT_CENTER | 1 << BACK_CENTER,
            // 5.0 (back)
            5 => {
                Self::STEREO.0 |
                    1 << FRONT_CENTER |
                    1 << BACK_LEFT |
                    1 << BACK_RIGHT
            },
            // 5.1 (back)
            6 => Self::default_for(5)?.0 | 1 << LOW_FREQUENCY,
            // 6.1
            7 => {
                Self::STEREO.0 |
                    1 << FRONT_CENTER |
                    1 << LOW_FREQUENCY |
                    1 << BACK_CENTER |
                    1 << SIDE_LEFT |
                    1 << SIDE_RIGHT
            },
            // 7.1
            8 => Self::default_for(6)?.0 | 1 << SIDE_LEFT | 1 << SIDE_RIGHT,
            _ => return None,
        };
        Some(Self(mask))
    }

    pub fn count(self) -> usize { self.0.count_ones() as usize }

    fn has(self, ch: usize) -> bool { self.0 & (1 << ch) != 0 }

    /// The data index of channel `ch` within this layout.
    fn index_of(self, ch: usize) -> Option<usize> {
        if !self.has(ch) {
            return None;
        }
        Some((self.0 & ((1 << ch) - 1)).count_ones() as usize)
    }

    fn channels(self) -> impl Iterator<Item = usize> {
        (0..64).filter(move |&ch| self.has(ch))
    }
}

/// A single mono channel that is not FRONT_CENTER is treated as mono
/// (`clean_layout`).
fn clean_layout(l: ChannelLayout) -> ChannelLayout {
    if l.count() == 1 && !l.has(FRONT_CENTER) {
        ChannelLayout::MONO
    } else {
        l
    }
}

/// The reference's `sane_layout` gate: at least one front speaker and no
/// asymmetric pairs.
fn sane_layout(l: ChannelLayout) -> bool {
    let pair = |a: usize, b: usize| (l.0 >> a & 1) == (l.0 >> b & 1);
    l.0 & 0b111 != 0 &&
        pair(FRONT_LEFT, FRONT_RIGHT) &&
        pair(SIDE_LEFT, SIDE_RIGHT) &&
        pair(BACK_LEFT, BACK_RIGHT) &&
        pair(FRONT_LEFT_OF_CENTER, FRONT_RIGHT_OF_CENTER) &&
        pair(TOP_FRONT_LEFT, TOP_FRONT_RIGHT)
}

/// Builds the mixing matrix `[out_index][in_index]`, ports `build_matrix`
/// (matrix encoding: none).
pub(crate) fn build_matrix(
    in_layout: ChannelLayout,
    out_layout: ChannelLayout,
) -> Result<Vec<Vec<f64>>, AudioError> {
    let unsupported = |what: &str| {
        AudioError::UnsupportedLayout(format!(
            "cannot mix {:#x} into {:#x}: {what}",
            in_layout.0, out_layout.0
        ))
    };
    let inl = clean_layout(in_layout);
    let outl = clean_layout(out_layout);
    if !sane_layout(inl) || !sane_layout(outl) {
        return Err(unsupported("layout out of the mixer's scope"));
    }

    // Named-channel working matrix, indexed by channel id.
    let mut m = [[0.0f64; NUM_NAMED_CHANNELS]; NUM_NAMED_CHANNELS];
    let unaccounted = inl.0 & !outl.0;

    for ch in 0..NUM_NAMED_CHANNELS {
        if inl.has(ch) && outl.has(ch) {
            m[ch][ch] = 1.0;
        }
    }

    if unaccounted & (1 << FRONT_CENTER) != 0 {
        if outl.0 & ChannelLayout::STEREO.0 == ChannelLayout::STEREO.0 {
            if inl.0 & ChannelLayout::STEREO.0 != 0 {
                m[FRONT_LEFT][FRONT_CENTER] += CENTER_MIX_LEVEL;
                m[FRONT_RIGHT][FRONT_CENTER] += CENTER_MIX_LEVEL;
            } else {
                m[FRONT_LEFT][FRONT_CENTER] += FRAC_1_SQRT_2;
                m[FRONT_RIGHT][FRONT_CENTER] += FRAC_1_SQRT_2;
            }
        } else {
            return Err(unsupported("stray front center"));
        }
    }
    if unaccounted & ChannelLayout::STEREO.0 != 0 {
        if outl.has(FRONT_CENTER) {
            m[FRONT_CENTER][FRONT_LEFT] += FRAC_1_SQRT_2;
            m[FRONT_CENTER][FRONT_RIGHT] += FRAC_1_SQRT_2;
            if inl.has(FRONT_CENTER) {
                m[FRONT_CENTER][FRONT_CENTER] = CENTER_MIX_LEVEL * SQRT_2;
            }
        } else {
            return Err(unsupported("stray front left/right"));
        }
    }

    if unaccounted & (1 << BACK_CENTER) != 0 {
        if outl.has(BACK_LEFT) {
            m[BACK_LEFT][BACK_CENTER] += FRAC_1_SQRT_2;
            m[BACK_RIGHT][BACK_CENTER] += FRAC_1_SQRT_2;
        } else if outl.has(SIDE_LEFT) {
            m[SIDE_LEFT][BACK_CENTER] += FRAC_1_SQRT_2;
            m[SIDE_RIGHT][BACK_CENTER] += FRAC_1_SQRT_2;
        } else if outl.has(FRONT_LEFT) {
            m[FRONT_LEFT][BACK_CENTER] += SURROUND_MIX_LEVEL * FRAC_1_SQRT_2;
            m[FRONT_RIGHT][BACK_CENTER] += SURROUND_MIX_LEVEL * FRAC_1_SQRT_2;
        } else if outl.has(FRONT_CENTER) {
            m[FRONT_CENTER][BACK_CENTER] += SURROUND_MIX_LEVEL * FRAC_1_SQRT_2;
        } else {
            return Err(unsupported("stray back center"));
        }
    }
    if unaccounted & (1 << BACK_LEFT) != 0 {
        if outl.has(BACK_CENTER) {
            m[BACK_CENTER][BACK_LEFT] += FRAC_1_SQRT_2;
            m[BACK_CENTER][BACK_RIGHT] += FRAC_1_SQRT_2;
        } else if outl.has(SIDE_LEFT) {
            let coef = if inl.has(SIDE_LEFT) {
                FRAC_1_SQRT_2
            } else {
                1.0
            };
            m[SIDE_LEFT][BACK_LEFT] += coef;
            m[SIDE_RIGHT][BACK_RIGHT] += coef;
        } else if outl.has(FRONT_LEFT) {
            m[FRONT_LEFT][BACK_LEFT] += SURROUND_MIX_LEVEL;
            m[FRONT_RIGHT][BACK_RIGHT] += SURROUND_MIX_LEVEL;
        } else if outl.has(FRONT_CENTER) {
            m[FRONT_CENTER][BACK_LEFT] += SURROUND_MIX_LEVEL * FRAC_1_SQRT_2;
            m[FRONT_CENTER][BACK_RIGHT] += SURROUND_MIX_LEVEL * FRAC_1_SQRT_2;
        } else {
            return Err(unsupported("stray back left/right"));
        }
    }
    if unaccounted & (1 << SIDE_LEFT) != 0 {
        if outl.has(BACK_LEFT) {
            let coef = if inl.has(BACK_LEFT) {
                FRAC_1_SQRT_2
            } else {
                1.0
            };
            m[BACK_LEFT][SIDE_LEFT] += coef;
            m[BACK_RIGHT][SIDE_RIGHT] += coef;
        } else if outl.has(BACK_CENTER) {
            m[BACK_CENTER][SIDE_LEFT] += FRAC_1_SQRT_2;
            m[BACK_CENTER][SIDE_RIGHT] += FRAC_1_SQRT_2;
        } else if outl.has(FRONT_LEFT) {
            m[FRONT_LEFT][SIDE_LEFT] += SURROUND_MIX_LEVEL;
            m[FRONT_RIGHT][SIDE_RIGHT] += SURROUND_MIX_LEVEL;
        } else if outl.has(FRONT_CENTER) {
            m[FRONT_CENTER][SIDE_LEFT] += SURROUND_MIX_LEVEL * FRAC_1_SQRT_2;
            m[FRONT_CENTER][SIDE_RIGHT] += SURROUND_MIX_LEVEL * FRAC_1_SQRT_2;
        } else {
            return Err(unsupported("stray side left/right"));
        }
    }
    if unaccounted & (1 << FRONT_LEFT_OF_CENTER) != 0 {
        if outl.has(FRONT_LEFT) {
            m[FRONT_LEFT][FRONT_LEFT_OF_CENTER] += 1.0;
            m[FRONT_RIGHT][FRONT_RIGHT_OF_CENTER] += 1.0;
        } else if outl.has(FRONT_CENTER) {
            m[FRONT_CENTER][FRONT_LEFT_OF_CENTER] += FRAC_1_SQRT_2;
            m[FRONT_CENTER][FRONT_RIGHT_OF_CENTER] += FRAC_1_SQRT_2;
        } else {
            return Err(unsupported("stray front left/right of center"));
        }
    }
    if unaccounted & (1 << TOP_FRONT_LEFT) != 0 {
        if outl.has(TOP_FRONT_CENTER) {
            m[TOP_FRONT_CENTER][TOP_FRONT_LEFT] += FRAC_1_SQRT_2;
            m[TOP_FRONT_CENTER][TOP_FRONT_RIGHT] += FRAC_1_SQRT_2;
            if inl.has(TOP_FRONT_CENTER) {
                m[TOP_FRONT_CENTER][TOP_FRONT_CENTER] =
                    CENTER_MIX_LEVEL * SQRT_2;
            }
        } else if outl.has(FRONT_LEFT) {
            let coef = if inl.has(FRONT_LEFT) {
                FRAC_1_SQRT_2
            } else {
                1.0
            };
            m[FRONT_LEFT][TOP_FRONT_LEFT] += coef;
            m[FRONT_RIGHT][TOP_FRONT_RIGHT] += coef;
        } else if outl.has(FRONT_CENTER) {
            m[FRONT_CENTER][TOP_FRONT_LEFT] += FRAC_1_SQRT_2;
            m[FRONT_CENTER][TOP_FRONT_RIGHT] += FRAC_1_SQRT_2;
        } else {
            return Err(unsupported("stray top front left/right"));
        }
    }
    if unaccounted & (1 << LOW_FREQUENCY) != 0 {
        if outl.has(FRONT_CENTER) {
            m[FRONT_CENTER][LOW_FREQUENCY] += LFE_MIX_LEVEL;
        } else if outl.has(FRONT_LEFT) {
            m[FRONT_LEFT][LOW_FREQUENCY] += LFE_MIX_LEVEL * FRAC_1_SQRT_2;
            m[FRONT_RIGHT][LOW_FREQUENCY] += LFE_MIX_LEVEL * FRAC_1_SQRT_2;
        } else {
            return Err(unsupported("stray LFE"));
        }
    }

    // Assemble [out][in] in data order; channels beyond the named set pass
    // through only identically. Track the largest row sum for normalization.
    let (n_in, n_out) = (inl.count(), outl.count());
    let mut matrix = vec![vec![0.0f64; n_in]; n_out];
    let mut maxcoef = 0.0f64;
    for out_ch in outl.channels() {
        let out_i = outl.index_of(out_ch).unwrap();
        let mut sum = 0.0;
        for in_ch in inl.channels() {
            let in_i = inl.index_of(in_ch).unwrap();
            let v = if out_ch < NUM_NAMED_CHANNELS && in_ch < NUM_NAMED_CHANNELS
            {
                m[out_ch][in_ch]
            } else {
                f64::from(u8::from(out_ch == in_ch))
            };
            matrix[out_i][in_i] = v;
            sum += v.abs();
        }
        maxcoef = maxcoef.max(sum);
    }

    // maxval is 1.0 whenever the output format is integer — always true for
    // this pipeline's s16 output.
    let maxval = 1.0;
    if maxcoef > maxval {
        for row in &mut matrix {
            for v in row {
                *v /= maxcoef;
            }
        }
    }
    Ok(matrix)
}

/// A working element of the mixing stage.
pub(crate) trait MixElement: Copy + Default + 'static {
    /// Native coefficient representation (float, double, or Q15).
    type Coef: Copy;
    /// Accumulator of the generic (>2 inputs) kernel.
    type Acc: Copy;

    /// Converts the double matrix into native coefficients.
    ///
    /// For the integer path this is the Q15 quantization with error feedback
    /// of the reference; the flag reports whether clipping variants must be
    /// used (a row of absolute Q15 sums exceeding unity).
    fn coefficients(matrix: &[Vec<f64>]) -> (Vec<Vec<Self::Coef>>, bool);

    fn mix1(x: Self, c: Self::Coef, clip: bool) -> Self;
    fn mix2(
        a: Self,
        ca: Self::Coef,
        b: Self,
        cb: Self::Coef,
        clip: bool,
    ) -> Self;
    fn acc_zero() -> Self::Acc;
    fn acc(acc: Self::Acc, x: Self, c: Self::Coef) -> Self::Acc;
    fn acc_end(acc: Self::Acc, clip: bool) -> Self;
}

impl MixElement for f32 {
    type Coef = f32;
    type Acc = f32;

    fn coefficients(matrix: &[Vec<f64>]) -> (Vec<Vec<f32>>, bool) {
        (
            matrix
                .iter()
                .map(|row| row.iter().map(|&v| v as f32).collect())
                .collect(),
            false,
        )
    }

    fn mix1(x: Self, c: f32, _clip: bool) -> Self { c * x }

    fn mix2(a: Self, ca: f32, b: Self, cb: f32, _clip: bool) -> Self {
        ca.mul_add(a, cb * b)
    }

    fn acc_zero() -> f32 { 0.0 }

    fn acc(acc: f32, x: Self, c: f32) -> f32 { x.mul_add(c, acc) }

    fn acc_end(acc: f32, _clip: bool) -> Self { acc }
}

impl MixElement for f64 {
    type Coef = f64;
    type Acc = f64;

    fn coefficients(matrix: &[Vec<f64>]) -> (Vec<Vec<f64>>, bool) {
        (matrix.to_vec(), false)
    }

    fn mix1(x: Self, c: f64, _clip: bool) -> Self { c * x }

    fn mix2(a: Self, ca: f64, b: Self, cb: f64, _clip: bool) -> Self {
        ca.mul_add(a, cb * b)
    }

    fn acc_zero() -> f64 { 0.0 }

    fn acc(acc: f64, x: Self, c: f64) -> f64 { x.mul_add(c, acc) }

    fn acc_end(acc: f64, _clip: bool) -> Self { acc }
}

impl MixElement for i16 {
    type Coef = i32;
    type Acc = i32;

    /// Q15 with error feedback: `target = m·32768 + rem; c = lrintf(target)`
    /// (the reference narrows the target to f32 before rounding).
    fn coefficients(matrix: &[Vec<f64>]) -> (Vec<Vec<i32>>, bool) {
        let mut maxsum = 0i64;
        let rows: Vec<Vec<i32>> = matrix
            .iter()
            .map(|row| {
                let mut rem = 0.0f64;
                let mut sum = 0i64;
                let out: Vec<i32> = row
                    .iter()
                    .map(|&v| {
                        let target = v * 32768.0 + rem;
                        let c = (target as f32).round_ties_even() as i32;
                        rem += target - f64::from(c);
                        sum += i64::from(c.abs());
                        c
                    })
                    .collect();
                maxsum = maxsum.max(sum);
                out
            })
            .collect();
        (rows, maxsum > 32768)
    }

    fn mix1(x: Self, c: i32, clip: bool) -> Self {
        finish_q15(i32::from(x).wrapping_mul(c), clip)
    }

    fn mix2(a: Self, ca: i32, b: Self, cb: i32, clip: bool) -> Self {
        let v = (i32::from(a).wrapping_mul(ca))
            .wrapping_add(i32::from(b).wrapping_mul(cb));
        finish_q15(v, clip)
    }

    fn acc_zero() -> i32 { 0 }

    fn acc(acc: i32, x: Self, c: i32) -> i32 {
        acc.wrapping_add(i32::from(x).wrapping_mul(c))
    }

    fn acc_end(acc: i32, clip: bool) -> Self { finish_q15(acc, clip) }
}

/// `R(x)` of the reference templates: a rounding Q15 shift, with saturation
/// in the clip variants.
fn finish_q15(v: i32, clip: bool) -> i16 {
    let shifted = v.wrapping_add(1 << 14) >> 15;
    if clip {
        shifted.clamp(-32768, 32767) as i16
    } else {
        shifted as i16
    }
}

/// Per-output-channel mixing plan (`matrix_ch` of the reference): the list
/// of inputs with nonzero coefficients.
pub(crate) struct Rematrix<T: MixElement> {
    coef: Vec<Vec<T::Coef>>,
    /// Exact-unity single-input rows copy verbatim (matrix value 1.0).
    unit: Vec<Vec<bool>>,
    plan: Vec<Vec<usize>>,
    clip: bool,
}

impl<T: MixElement> Rematrix<T> {
    pub fn new(matrix: &[Vec<f64>]) -> Self {
        let (coef, clip) = T::coefficients(matrix);
        let plan: Vec<Vec<usize>> = matrix
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|&(_, &v)| v != 0.0)
                    .map(|(j, _)| j)
                    .collect()
            })
            .collect();
        let unit = matrix
            .iter()
            .map(|row| row.iter().map(|&v| v == 1.0).collect())
            .collect();
        Self {
            coef,
            unit,
            plan,
            clip,
        }
    }

    /// Mixes planar input channels into planar output channels.
    pub fn apply(&self, input: &[&[T]]) -> Vec<Vec<T>> {
        let len = input.first().map_or(0, |c| c.len());
        self.plan
            .iter()
            .enumerate()
            .map(|(out_i, inputs)| match inputs.len() {
                0 => vec![T::default(); len],
                1 => {
                    let j = inputs[0];
                    if self.unit[out_i][j] {
                        input[j].to_vec()
                    } else {
                        let c = self.coef[out_i][j];
                        input[j]
                            .iter()
                            .map(|&x| T::mix1(x, c, self.clip))
                            .collect()
                    }
                },
                2 => {
                    let (j1, j2) = (inputs[0], inputs[1]);
                    let (c1, c2) = (self.coef[out_i][j1], self.coef[out_i][j2]);
                    input[j1]
                        .iter()
                        .zip(input[j2])
                        .map(|(&a, &b)| T::mix2(a, c1, b, c2, self.clip))
                        .collect()
                },
                _ => (0..len)
                    .map(|i| {
                        let mut acc = T::acc_zero();
                        for &j in inputs {
                            acc = T::acc(acc, input[j][i], self.coef[out_i][j]);
                        }
                        T::acc_end(acc, self.clip)
                    })
                    .collect(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
