//! Decoding media files into native-format PCM via symphonia.
//!
//! The decoder preserves each stream's native sample representation (the
//! shaping pipeline chooses its working format from it, as the reference
//! does) and normalizes integer widths the way the reference pcm/flac
//! decoders present them: 24-bit samples ride in `s32 << 8`, low-justified
//! lossless payloads shift up to their container width. Gapless trimming is
//! on (the default), matching the reference's encoder-delay handling.

use std::{ffi::OsStr, fs::File, path::Path};

use symphonia::core::{
    audio::{Audio, Channels, GenericAudioBufferRef, sample::i24},
    codecs::{CodecParameters, audio::AudioDecoderOptions},
    errors::Error as SymError,
    formats::{FormatOptions, FormatReader, TrackType, probe::Hint},
    io::MediaSourceStream,
    meta::MetadataOptions,
};

use super::{
    error::{AudioError, SUPPORTED_FORMATS},
    pipeline::{NativeBuf, SampleFormat, Shaper},
};

mod edit_list;

/// A fully decoded audio stream in its native representation.
#[derive(Debug)]
pub struct DecodedAudio {
    rate: u32,
    channel_mask: Option<u64>,
    /// The source bit depth the container declared, when known. Used to tell a
    /// 24-bit stream (which rides in `s32 << 8`) from a true 32-bit one.
    source_bits: Option<u32>,
    samples: NativeBuf,
}

impl DecodedAudio {
    /// The stream sample rate, Hz.
    pub fn sample_rate(&self) -> u32 { self.rate }

    /// The number of channels.
    pub fn channels(&self) -> usize { self.samples.channels() }

    /// The number of frames (samples per channel).
    pub fn frames(&self) -> usize { self.samples.frames() }

    /// The native sample format the decoder produced.
    pub fn format(&self) -> SampleFormat { self.samples.format() }

    /// Whether the samples are floating point (lossy decoders decode to f32).
    pub fn is_float(&self) -> bool {
        matches!(self.samples.format(), SampleFormat::F32 | SampleFormat::F64)
    }

    /// The meaningful bit depth of a sample: 8/16/24/32 for integer PCM, 32/64
    /// for float. A 24-bit stream reports 24 even though it rides in `s32`.
    pub fn bits_per_sample(&self) -> u32 {
        match self.samples.format() {
            SampleFormat::U8 => 8,
            SampleFormat::S16 => 16,
            SampleFormat::S32 => match self.source_bits {
                Some(bits) if bits <= 24 => 24,
                _ => 32,
            },
            SampleFormat::F32 => 32,
            SampleFormat::F64 => 64,
        }
    }

    /// The stream duration in seconds (frames / sample rate).
    pub fn duration_seconds(&self) -> f64 {
        self.frames() as f64 / f64::from(self.rate)
    }

    /// The planar native samples, for in-crate editing and encoding.
    pub(crate) fn samples(&self) -> &NativeBuf { &self.samples }

    /// The channel-position mask, when the container declared one.
    pub(crate) fn channel_mask(&self) -> Option<u64> { self.channel_mask }

    /// The declared source bit depth, when known.
    pub(crate) fn source_bits(&self) -> Option<u32> { self.source_bits }

    /// Rebuilds a stream from parts (used by the editor to return a cut buffer
    /// with the same rate/layout/format as the original).
    pub(crate) fn from_parts(
        rate: u32,
        channel_mask: Option<u64>,
        source_bits: Option<u32>,
        samples: NativeBuf,
    ) -> Self {
        Self {
            rate,
            channel_mask,
            source_bits,
            samples,
        }
    }

    /// Shapes the audio into mono s16 at `out_rate` — the numeric equivalent
    /// of the classic `ffmpeg -f s16le -ac 1 -ar <rate>` chain.
    pub fn to_mono_s16(self, out_rate: u32) -> Result<Vec<i16>, AudioError> {
        let mut shaper = Shaper::new(
            self.rate,
            self.samples.channels(),
            self.channel_mask,
            self.samples.format(),
            out_rate,
        )?;
        let mut out = shaper.feed(self.samples);
        out.extend(shaper.finish());
        Ok(out)
    }
}

/// Decodes a whole media file, keeping the native sample format.
pub fn decode_file(path: &Path) -> Result<DecodedAudio, AudioError> {
    let mut stream = AudioStream::open(path)?;
    let mut samples: Option<NativeBuf> = None;
    while let Some(buf) = stream.next_buf()? {
        match &mut samples {
            None => samples = Some(buf),
            Some(all) => all.append(buf)?,
        }
    }
    Ok(DecodedAudio {
        rate: stream.rate,
        channel_mask: stream.mask,
        source_bits: stream.bits,
        samples: samples.unwrap_or_else(|| {
            NativeBuf::S16(vec![Vec::new(); stream.channels])
        }),
    })
}

/// Decodes and shapes a media file into mono s16 at `out_rate`, streaming
/// packet by packet (bounded memory even for long recordings).
pub fn decode_to_mono_s16(
    path: &Path,
    out_rate: u32,
) -> Result<Vec<i16>, AudioError> {
    let mut stream = MonoS16Stream::open(path, out_rate)?;
    let mut out = Vec::new();
    while let Some(block) = stream.next_block()? {
        out.extend(block);
    }
    Ok(out)
}

/// Decodes and shapes a media file into mono f32 in `[-1, 1)` at `out_rate` —
/// the shape voice-activity detection consumes. Streams packet by packet, then
/// scales the mono s16 the same way the reference `load_audio` does.
pub fn decode_to_mono_f32(
    path: &Path,
    out_rate: u32,
) -> Result<Vec<f32>, AudioError> {
    Ok(decode_to_mono_s16(path, out_rate)?
        .into_iter()
        .map(|sample| f32::from(sample) / 32768.0)
        .collect())
}

/// A streaming decode of a media file into mono s16 at a target rate: the
/// same packet-by-packet chain as [`decode_to_mono_s16`], exposed one shaped
/// block at a time so a consumer can process audio without ever holding the
/// whole file. The concatenation of all blocks is exactly the batch result.
pub struct MonoS16Stream {
    stream: AudioStream,
    shaper: Option<Shaper>,
    out_rate: u32,
    flushed: bool,
}

impl MonoS16Stream {
    /// Opens `path` for streaming decode to mono s16 at `out_rate`.
    pub fn open(path: &Path, out_rate: u32) -> Result<Self, AudioError> {
        Ok(Self {
            stream: AudioStream::open(path)?,
            shaper: None,
            out_rate,
            flushed: false,
        })
    }

    /// The track duration in seconds, when the container declares it. A hint
    /// for progress reporting only — streams may misdeclare or omit it.
    pub fn duration_hint(&self) -> Option<f64> { self.stream.duration_hint }

    /// The next non-empty block of shaped samples, or `None` at the end
    /// (after the final resampler flush).
    pub fn next_block(&mut self) -> Result<Option<Vec<i16>>, AudioError> {
        while !self.flushed {
            let Some(buf) = self.stream.next_buf()? else {
                self.flushed = true;
                if let Some(shaper) = &mut self.shaper {
                    let tail = shaper.finish();
                    if !tail.is_empty() {
                        return Ok(Some(tail));
                    }
                }
                return Ok(None);
            };
            let shaper = match &mut self.shaper {
                Some(s) => s,
                None => self.shaper.insert(Shaper::new(
                    self.stream.rate,
                    buf.channels(),
                    self.stream.mask,
                    buf.format(),
                    self.out_rate,
                )?),
            };
            let block = shaper.feed(buf);
            // The resampler may be priming; skip empty output.
            if !block.is_empty() {
                return Ok(Some(block));
            }
        }
        Ok(None)
    }
}

/// An opened media file: format reader plus the decoder of its default
/// audio track.
struct AudioStream {
    reader: Box<dyn FormatReader>,
    decoder: Box<dyn symphonia::core::codecs::audio::AudioDecoder>,
    track_id: u32,
    rate: u32,
    channels: usize,
    mask: Option<u64>,
    bits: Option<u32>,
    /// Leading frames still to drop (mp4 edit-list priming, §gapless).
    pending_skip: u64,
    /// Declared track duration in seconds, when the container knows it.
    duration_hint: Option<f64>,
}

impl AudioStream {
    fn open(path: &Path) -> Result<Self, AudioError> {
        let file = File::open(path).map_err(|source| AudioError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(OsStr::to_str) {
            hint.with_extension(ext);
        }
        let reader = symphonia::default::get_probe()
            .probe(
                &hint,
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(|e| AudioError::UnsupportedFormat {
                detail: format!("container not recognized: {e}"),
                supported: SUPPORTED_FORMATS,
            })?;

        let track = reader
            .default_track(TrackType::Audio)
            .ok_or(AudioError::NoAudioTrack)?;
        let track_id = track.id;
        let params = match &track.codec_params {
            Some(CodecParameters::Audio(p)) => p.clone(),
            _ => return Err(AudioError::NoAudioTrack),
        };
        // Declared duration, for progress hints: playable frames over the
        // source rate when known, else the container duration via the
        // timebase.
        let duration_hint = match (track.num_frames, params.sample_rate) {
            (Some(frames), Some(rate)) if rate > 0 => {
                Some(frames as f64 / f64::from(rate))
            },
            _ => match (track.duration, track.time_base) {
                (Some(duration), Some(tb)) => tb
                    .calc_time(symphonia::core::units::Timestamp::new(
                        duration.get() as i64,
                    ))
                    .map(|time| time.as_secs_f64()),
                _ => None,
            },
        };
        // Gapless trimming is on by default — encoder delay and padding are
        // removed the way the reference demuxer/decoder pair does it.
        let decoder = symphonia::default::get_codecs()
            .make_audio_decoder(&params, &AudioDecoderOptions::default())
            .map_err(|e| AudioError::UnsupportedFormat {
                detail: format!("codec not supported: {e}"),
                supported: SUPPORTED_FORMATS,
            })?;

        let rate = params.sample_rate.ok_or_else(|| {
            AudioError::Decode("stream declares no sample rate".into())
        })?;
        let (channels, mask) = match &params.channels {
            Some(Channels::Positioned(p)) => {
                let bits = p.bits() as u64;
                let count = bits.count_ones() as usize;
                // Positions beyond the named range have no equivalent in
                // the mixer; fall back to the per-count default layout.
                if bits >> 18 != 0 {
                    (count, None)
                } else {
                    (count, Some(bits))
                }
            },
            Some(Channels::Discrete(n)) => (usize::from(*n), None),
            Some(other) => {
                return Err(AudioError::UnsupportedLayout(format!(
                    "{other:?}"
                )));
            },
            None => {
                return Err(AudioError::Decode(
                    "stream declares no channels".into(),
                ));
            },
        };
        // The mp4 demuxer does not apply edit lists; reproduce the
        // reference's leading trim (AAC priming) ourselves.
        let pending_skip =
            edit_list::start_trim_frames(path, rate).unwrap_or(0);

        Ok(Self {
            reader,
            decoder,
            track_id,
            rate,
            channels,
            mask,
            bits: params.bits_per_sample,
            pending_skip,
            duration_hint,
        })
    }

    /// Decodes forward to the next non-empty audio buffer.
    ///
    /// Damaged packets are skipped (the reference decodes what it can);
    /// end of stream returns `None`.
    fn next_buf(&mut self) -> Result<Option<NativeBuf>, AudioError> {
        loop {
            let packet = match self.reader.next_packet() {
                Ok(Some(p)) => p,
                Ok(None) => return Ok(None),
                Err(e) => return Err(AudioError::Decode(e.to_string())),
            };
            if packet.track_id != self.track_id {
                continue;
            }
            match self.decoder.decode(&packet) {
                Ok(buf) => {
                    if buf.frames() == 0 {
                        continue;
                    }
                    let bits = self.bits;
                    let mut native = native_buf(&buf, bits)?;
                    if self.pending_skip > 0 {
                        let cut =
                            (self.pending_skip as usize).min(native.frames());
                        native.drop_front(cut);
                        self.pending_skip -= cut as u64;
                        if native.frames() == 0 {
                            continue;
                        }
                    }
                    return Ok(Some(native));
                },
                Err(SymError::DecodeError(_)) => continue,
                Err(e) => return Err(AudioError::Decode(e.to_string())),
            }
        }
    }
}

/// Extracts the buffer planes in the representation the reference decoders
/// would have produced.
fn native_buf(
    buf: &GenericAudioBufferRef<'_>,
    bits: Option<u32>,
) -> Result<NativeBuf, AudioError> {
    fn planes<S, D>(
        buf: &symphonia::core::audio::AudioBuffer<S>,
        map: impl Fn(S) -> D,
    ) -> Vec<Vec<D>>
    where
        S: symphonia::core::audio::sample::Sample
            + symphonia::core::audio::conv::FromSample<S>,
    {
        let mut planar: Vec<Vec<S>> = Vec::new();
        buf.copy_to_vecs_planar(&mut planar);
        planar
            .into_iter()
            .map(|ch| ch.into_iter().map(&map).collect())
            .collect()
    }

    Ok(match buf {
        GenericAudioBufferRef::U8(b) => NativeBuf::U8(planes(b, |x| x)),
        // The reference pcm_s8 decoder presents signed 8-bit as u8.
        GenericAudioBufferRef::S8(b) => {
            NativeBuf::U8(planes(b, |x| (i16::from(x) + 0x80) as u8))
        },
        GenericAudioBufferRef::S16(b) => NativeBuf::S16(planes(b, |x| x)),
        // 24-bit rides as s32 << 8, as the reference pcm decoders do.
        GenericAudioBufferRef::S24(b) => {
            NativeBuf::S32(planes(b, |x: i24| x.inner() << 8))
        },
        GenericAudioBufferRef::S32(b) => {
            // Lossless decoders (flac, alac) fill s32 buffers to full
            // scale (values shifted up to 32 bits). That already equals
            // the reference's s32 representation for depths above 16;
            // depths up to 16 narrow exactly to s16, reproducing the
            // reference decoders' native format (and therefore the
            // pipeline's format selection).
            let planes32 = planes(b, |x| x);
            match bits {
                Some(1..=16) => NativeBuf::S16(
                    planes32
                        .into_iter()
                        .map(|ch| {
                            ch.into_iter()
                                .map(|v: i32| (v >> 16) as i16)
                                .collect()
                        })
                        .collect(),
                ),
                _ => NativeBuf::S32(planes32),
            }
        },
        GenericAudioBufferRef::F32(b) => NativeBuf::F32(planes(b, |x| x)),
        GenericAudioBufferRef::F64(b) => NativeBuf::F64(planes(b, |x| x)),
        GenericAudioBufferRef::U16(_) |
        GenericAudioBufferRef::U24(_) |
        GenericAudioBufferRef::U32(_) => {
            return Err(AudioError::UnsupportedFormat {
                detail: "unsigned wide pcm has no reference equivalent".into(),
                supported: SUPPORTED_FORMATS,
            });
        },
    })
}

#[cfg(test)]
mod tests;
