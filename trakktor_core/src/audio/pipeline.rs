//! Stage selection and ordering of the PCM shaping pipeline.
//!
//! Ports the decisions of swresample's initialization and conversion core
//! (`swresample.c`): the internal working format, whether resampling runs
//! before or after channel mixing, and which stages exist at all. The v1
//! surface produces mono s16 — the shape of the ffmpeg reference command
//! `-f s16le -ac 1 -ar <rate>`.

use super::{
    convert,
    error::AudioError,
    rematrix::{ChannelLayout, MixElement, Rematrix, build_matrix},
    resample::{Element, Resampler},
};

/// Native sample formats a decoded stream can carry.
///
/// These are the representations the decoder produces; the reference's
/// signed-64-bit path has no equivalent (no supported codec decodes to it).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleFormat {
    U8,
    S16,
    /// Also carries 24-bit sources, shifted left by 8.
    S32,
    F32,
    F64,
}

impl SampleFormat {
    fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::S16 => 2,
            Self::S32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// One planar buffer of decoded audio in its native representation.
#[derive(Debug)]
pub(crate) enum NativeBuf {
    U8(Vec<Vec<u8>>),
    S16(Vec<Vec<i16>>),
    S32(Vec<Vec<i32>>),
    F32(Vec<Vec<f32>>),
    F64(Vec<Vec<f64>>),
}

impl NativeBuf {
    pub fn format(&self) -> SampleFormat {
        match self {
            Self::U8(_) => SampleFormat::U8,
            Self::S16(_) => SampleFormat::S16,
            Self::S32(_) => SampleFormat::S32,
            Self::F32(_) => SampleFormat::F32,
            Self::F64(_) => SampleFormat::F64,
        }
    }

    pub fn channels(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::S16(v) => v.len(),
            Self::S32(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    pub fn frames(&self) -> usize {
        match self {
            Self::U8(v) => v.first().map_or(0, Vec::len),
            Self::S16(v) => v.first().map_or(0, Vec::len),
            Self::S32(v) => v.first().map_or(0, Vec::len),
            Self::F32(v) => v.first().map_or(0, Vec::len),
            Self::F64(v) => v.first().map_or(0, Vec::len),
        }
    }

    /// Drops the first `n` frames (used for container-level start trims).
    pub fn drop_front(&mut self, n: usize) {
        fn cut<T>(v: &mut [Vec<T>], n: usize) {
            for ch in v {
                ch.drain(..n.min(ch.len()));
            }
        }
        match self {
            Self::U8(v) => cut(v, n),
            Self::S16(v) => cut(v, n),
            Self::S32(v) => cut(v, n),
            Self::F32(v) => cut(v, n),
            Self::F64(v) => cut(v, n),
        }
    }

    /// Appends another buffer of the same shape (format and channel count).
    pub fn append(&mut self, other: NativeBuf) -> Result<(), AudioError> {
        fn glue<T>(a: &mut [Vec<T>], b: Vec<Vec<T>>) -> bool {
            if a.len() != b.len() {
                return false;
            }
            for (dst, src) in a.iter_mut().zip(b) {
                dst.extend(src);
            }
            true
        }
        let ok = match (self, other) {
            (Self::U8(a), Self::U8(b)) => glue(a, b),
            (Self::S16(a), Self::S16(b)) => glue(a, b),
            (Self::S32(a), Self::S32(b)) => glue(a, b),
            (Self::F32(a), Self::F32(b)) => glue(a, b),
            (Self::F64(a), Self::F64(b)) => glue(a, b),
            _ => false,
        };
        if ok {
            Ok(())
        } else {
            Err(AudioError::Decode(
                "stream changed its sample format or channel count".into(),
            ))
        }
    }
}

/// Streaming shaper: native-format planar chunks in, mono s16 out.
pub(crate) struct Shaper {
    lane: Lane,
}

/// The internal working format (`swresample.c:227-253`), specialized for the
/// fixed s16 output: 16-bit-or-narrower integer inputs stay integer when no
/// resampling is needed (u8 even with it); anything up to four bytes works
/// in floats; doubles work in doubles.
enum Lane {
    S16(TypedLane<i16>),
    F32(TypedLane<f32>),
    F64(TypedLane<f64>),
}

struct TypedLane<T: Element + MixElement> {
    resample_first: bool,
    resampler: Option<Resampler<T>>,
    rematrix: Option<Rematrix<T>>,
}

fn refs<T>(planar: &[Vec<T>]) -> Vec<&[T]> {
    planar.iter().map(Vec::as_slice).collect()
}

impl<T: Element + MixElement> TypedLane<T> {
    fn new(
        in_rate: u32,
        in_layout: ChannelLayout,
        out_rate: u32,
        out_layout: ChannelLayout,
    ) -> Result<Self, AudioError> {
        let (in_ch, out_ch) = (in_layout.count(), out_layout.count());
        // The stage order of the reference (`swresample.c:337`, RSC = 1).
        // The left side divides in integers; the right side is a float
        // ratio. For mono output from ≥2 channels the left side is −1, so
        // resampling always runs first there.
        let lhs = (out_ch as i64 / in_ch as i64 - 1) as f64;
        let rhs = f64::from(out_rate as f32 / in_rate as f32) - 1.0;
        let resample_first = lhs < rhs;

        let resampler = if in_rate != out_rate {
            let channels = if resample_first { in_ch } else { out_ch };
            Some(Resampler::new(out_rate, in_rate, channels)?)
        } else {
            None
        };
        let rematrix = if in_layout != out_layout {
            Some(Rematrix::new(&build_matrix(in_layout, out_layout)?))
        } else {
            None
        };
        Ok(Self {
            resample_first,
            resampler,
            rematrix,
        })
    }

    /// Runs the stages in the configured order; returns the mono channel.
    fn push(&mut self, chunk: Vec<Vec<T>>) -> Vec<T> {
        let mut cur = chunk;
        if self.resample_first {
            if let Some(r) = &mut self.resampler {
                cur = r.feed(&refs(&cur));
            }
            if let Some(m) = &self.rematrix {
                cur = m.apply(&refs(&cur));
            }
        } else {
            if let Some(m) = &self.rematrix {
                cur = m.apply(&refs(&cur));
            }
            if let Some(r) = &mut self.resampler {
                cur = r.feed(&refs(&cur));
            }
        }
        cur.swap_remove(0)
    }

    fn drain(&mut self) -> Vec<T> {
        let Some(r) = &mut self.resampler else {
            return Vec::new();
        };
        let mut cur = r.finish();
        if self.resample_first {
            if let Some(m) = &self.rematrix {
                cur = m.apply(&refs(&cur));
            }
        }
        cur.swap_remove(0)
    }
}

impl Shaper {
    /// Configures the pipeline for one input stream shape.
    ///
    /// `in_layout` is the channel-position mask when the container declares
    /// one; otherwise the reference's per-count default layout is assumed.
    pub fn new(
        in_rate: u32,
        in_channels: usize,
        in_layout: Option<u64>,
        format: SampleFormat,
        out_rate: u32,
    ) -> Result<Self, AudioError> {
        if in_channels == 0 || in_rate == 0 {
            return Err(AudioError::Decode(
                "stream without channels or sample rate".into(),
            ));
        }
        let in_layout = match in_layout {
            Some(mask) if ChannelLayout(mask).count() == in_channels => {
                ChannelLayout(mask)
            },
            _ => ChannelLayout::default_for(in_channels).ok_or_else(|| {
                AudioError::UnsupportedLayout(format!(
                    "{in_channels} channels without a known layout"
                ))
            })?,
        };
        let out_layout = ChannelLayout::MONO;

        let same_rate = in_rate == out_rate;
        let bytes = format.bytes();
        // Internal format selection for the fixed s16 output.
        let lane = if (bytes <= 2 && same_rate) || bytes + 2 <= 3 {
            Lane::S16(TypedLane::new(in_rate, in_layout, out_rate, out_layout)?)
        } else if bytes <= 4 {
            Lane::F32(TypedLane::new(in_rate, in_layout, out_rate, out_layout)?)
        } else {
            Lane::F64(TypedLane::new(in_rate, in_layout, out_rate, out_layout)?)
        };
        Ok(Self { lane })
    }

    /// Feeds one native-format chunk, returning the mono s16 it unlocks.
    pub fn feed(&mut self, buf: NativeBuf) -> Vec<i16> {
        match &mut self.lane {
            Lane::S16(lane) => {
                let chunk: Vec<Vec<i16>> = match buf {
                    NativeBuf::U8(v) => v
                        .into_iter()
                        .map(|ch| {
                            ch.into_iter().map(convert::u8_to_s16).collect()
                        })
                        .collect(),
                    NativeBuf::S16(v) => v,
                    other => unreachable!(
                        "s16 lane cannot take {:?} input",
                        other.format()
                    ),
                };
                lane.push(chunk)
            },
            Lane::F32(lane) => {
                let chunk: Vec<Vec<f32>> = match buf {
                    NativeBuf::S16(v) => v
                        .into_iter()
                        .map(|ch| {
                            ch.into_iter().map(convert::s16_to_f32).collect()
                        })
                        .collect(),
                    NativeBuf::S32(v) => v
                        .into_iter()
                        .map(|ch| {
                            ch.into_iter().map(convert::s32_to_f32).collect()
                        })
                        .collect(),
                    NativeBuf::F32(v) => v,
                    other => unreachable!(
                        "float lane cannot take {:?} input",
                        other.format()
                    ),
                };
                lane.push(chunk)
                    .into_iter()
                    .map(convert::f32_to_s16)
                    .collect()
            },
            Lane::F64(lane) => {
                let chunk: Vec<Vec<f64>> = match buf {
                    NativeBuf::F64(v) => v,
                    other => unreachable!(
                        "double lane cannot take {:?} input",
                        other.format()
                    ),
                };
                lane.push(chunk)
                    .into_iter()
                    .map(convert::f64_to_s16)
                    .collect()
            },
        }
    }

    /// Drains the pipeline tail after the last chunk.
    pub fn finish(&mut self) -> Vec<i16> {
        match &mut self.lane {
            Lane::S16(lane) => lane.drain(),
            Lane::F32(lane) => {
                lane.drain().into_iter().map(convert::f32_to_s16).collect()
            },
            Lane::F64(lane) => {
                lane.drain().into_iter().map(convert::f64_to_s16).collect()
            },
        }
    }
}

#[cfg(test)]
mod tests;
