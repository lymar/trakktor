//! Encoding decoded audio to WAV and FLAC, losslessly and in pure Rust.
//!
//! WAV is written with [hound] and can hold every native representation the
//! decoder produces (8/16/24/32-bit integer and 32-bit float; 64-bit float is
//! narrowed to 32-bit). FLAC is written with [flacenc] and is integer-only up
//! to 24 bits, so float sources (mp3/aac/vorbis decode to f32) and true 32-bit
//! integer sources are rejected with a clear message pointing at WAV.
//!
//! Both paths copy the sample values through unchanged, so the output is a
//! lossless snapshot of the decoded original — the point of the editing
//! feature is to cut at full quality rather than at the 16 kHz mono the
//! detector works on.
//!
//! [hound]: https://github.com/ruuda/hound
//! [flacenc]: https://github.com/yotarok/flacenc-rs

#[cfg(test)]
mod tests;

use std::{
    io::{BufWriter, Write},
    path::Path,
    process::{Command, Stdio},
    thread,
};

use flacenc::{component::BitRepr, error::Verify};

use super::{
    decode::DecodedAudio,
    error::AudioError,
    pipeline::{NativeBuf, SampleFormat},
};

/// The lossless container to write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// RIFF/WAVE, uncompressed. Holds any decoded representation.
    Wav,
    /// FLAC, lossless-compressed. Integer sources up to 24 bits only.
    Flac,
}

/// Writes `audio` to `path` in the requested lossless format.
///
/// # Errors
///
/// Returns [`AudioError::UnsupportedEncoding`] when the format cannot represent
/// the source losslessly (FLAC of a float or >24-bit source), or
/// [`AudioError::Encode`] when the encoder or the file write fails.
pub fn write(
    path: &Path,
    audio: &DecodedAudio,
    format: Format,
) -> Result<(), AudioError> {
    match format {
        Format::Wav => write_wav(path, audio),
        Format::Flac => write_flac(path, audio),
    }
}

/// Builds an [`AudioError::Encode`] carrying a rendered cause.
fn encode_err(path: &Path, cause: impl std::fmt::Display) -> AudioError {
    AudioError::Encode {
        path: path.to_path_buf(),
        message: cause.to_string(),
    }
}

fn write_wav(path: &Path, audio: &DecodedAudio) -> Result<(), AudioError> {
    // 24-bit rides in `s32 << 8`; write the true depth so the file is not
    // needlessly widened. f64 has no WAV representation here — narrow to f32.
    let (bits, sample_format) = match audio.format() {
        SampleFormat::U8 => (8, hound::SampleFormat::Int),
        SampleFormat::S16 => (16, hound::SampleFormat::Int),
        SampleFormat::S32 => {
            (audio.bits_per_sample() as u16, hound::SampleFormat::Int)
        },
        SampleFormat::F32 | SampleFormat::F64 => {
            (32, hound::SampleFormat::Float)
        },
    };
    let spec = hound::WavSpec {
        channels: audio.channels() as u16,
        sample_rate: audio.sample_rate(),
        bits_per_sample: bits,
        sample_format,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| encode_err(path, e))?;
    let ch = audio.channels();
    let frames = audio.frames();
    // Interleave frame by frame, mapping each native representation to what
    // hound expects for the chosen spec.
    match audio.samples() {
        NativeBuf::U8(planes) => {
            for i in 0..frames {
                for plane in planes.iter().take(ch) {
                    // WAV 8-bit is unsigned; hound takes a signed i8 and stores
                    // it offset back to unsigned, so shift the center to 0.
                    let sample = (i16::from(plane[i]) - 128) as i8;
                    writer
                        .write_sample(sample)
                        .map_err(|e| encode_err(path, e))?;
                }
            }
        },
        NativeBuf::S16(planes) => {
            for i in 0..frames {
                for plane in planes.iter().take(ch) {
                    writer
                        .write_sample(plane[i])
                        .map_err(|e| encode_err(path, e))?;
                }
            }
        },
        NativeBuf::S32(planes) => {
            // 24-bit data sits in the high 24 bits; hound wants the value in
            // the low bits of the i32, so shift down for a 24-bit
            // spec.
            let shift = if bits == 24 { 8 } else { 0 };
            for i in 0..frames {
                for plane in planes.iter().take(ch) {
                    writer
                        .write_sample(plane[i] >> shift)
                        .map_err(|e| encode_err(path, e))?;
                }
            }
        },
        NativeBuf::F32(planes) => {
            for i in 0..frames {
                for plane in planes.iter().take(ch) {
                    writer
                        .write_sample(plane[i])
                        .map_err(|e| encode_err(path, e))?;
                }
            }
        },
        NativeBuf::F64(planes) => {
            for i in 0..frames {
                for plane in planes.iter().take(ch) {
                    writer
                        .write_sample(plane[i] as f32)
                        .map_err(|e| encode_err(path, e))?;
                }
            }
        },
    }
    writer.finalize().map_err(|e| encode_err(path, e))
}

fn write_flac(path: &Path, audio: &DecodedAudio) -> Result<(), AudioError> {
    if audio.is_float() {
        return Err(AudioError::UnsupportedEncoding(
            "the source is float PCM (mp3/aac/vorbis decode to floats); FLAC \
             is integer-only — write WAV to stay lossless"
                .into(),
        ));
    }
    let bits = audio.bits_per_sample();
    if bits > 24 {
        return Err(AudioError::UnsupportedEncoding(format!(
            "the source is {bits}-bit; FLAC stores at most 24 bits — write \
             WAV to stay lossless"
        )));
    }

    let ch = audio.channels();
    let frames = audio.frames();
    // FLAC takes interleaved i32 with the sample value in the low `bits` bits.
    let mut interleaved: Vec<i32> = Vec::with_capacity(frames * ch);
    match audio.samples() {
        NativeBuf::U8(planes) => {
            for i in 0..frames {
                for plane in planes.iter().take(ch) {
                    interleaved.push(i32::from(plane[i]) - 128);
                }
            }
        },
        NativeBuf::S16(planes) => {
            for i in 0..frames {
                for plane in planes.iter().take(ch) {
                    interleaved.push(i32::from(plane[i]));
                }
            }
        },
        NativeBuf::S32(planes) => {
            // bits == 24 here (32-bit was rejected above): drop the low byte.
            for i in 0..frames {
                for plane in planes.iter().take(ch) {
                    interleaved.push(plane[i] >> 8);
                }
            }
        },
        NativeBuf::F32(_) | NativeBuf::F64(_) => {
            unreachable!("float sources are rejected above")
        },
    }

    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|(_, e)| encode_err(path, e))?;
    let source = flacenc::source::MemSource::from_samples(
        &interleaved,
        ch,
        bits as usize,
        audio.sample_rate() as usize,
    );
    let stream = flacenc::encode_with_fixed_block_size(
        &config,
        source,
        config.block_size,
    )
    .map_err(|e| encode_err(path, e))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| encode_err(path, format!("{e:?}")))?;
    std::fs::write(path, sink.as_slice()).map_err(|e| encode_err(path, e))
}

/// Encodes `audio` to `path` through an external `ffmpeg` process, which picks
/// the container and codec from the output extension. This is the opt-in path
/// for formats the built-in encoders do not cover (mp3, aac, opus, m4a, …).
///
/// The cut PCM is streamed to ffmpeg's stdin as raw interleaved samples in the
/// source's native format, and ffmpeg writes the file. `bitrate` (for example
/// `"192k"`) sets `-b:a` for lossy targets; `None` leaves ffmpeg's default.
///
/// Needs `ffmpeg` on `PATH`; a missing binary or a non-zero exit is reported as
/// [`AudioError::Ffmpeg`]. Since the samples are re-encoded (not
/// stream-copied), the cut stays sample-accurate for any target format.
///
/// # Errors
///
/// Returns [`AudioError::Ffmpeg`] if ffmpeg cannot be run, rejects the input,
/// or exits with an error.
pub fn write_ffmpeg(
    path: &Path,
    audio: &DecodedAudio,
    bitrate: Option<&str>,
) -> Result<(), AudioError> {
    let raw_format = match audio.format() {
        SampleFormat::U8 => "u8",
        SampleFormat::S16 => "s16le",
        SampleFormat::S32 => "s32le",
        SampleFormat::F32 => "f32le",
        SampleFormat::F64 => "f64le",
    };
    let mut command = Command::new("ffmpeg");
    command
        .args(["-hide_banner", "-loglevel", "error", "-y", "-f", raw_format])
        .args(["-ar"])
        .arg(audio.sample_rate().to_string())
        .args(["-ac"])
        .arg(audio.channels().to_string())
        .args(["-i", "-"]);
    if let Some(bitrate) = bitrate {
        command.args(["-b:a", bitrate]);
    }
    command
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    let mut child = command.spawn().map_err(|e| {
        AudioError::Ffmpeg(format!(
            "could not run ffmpeg ({e}); is it installed and on PATH?"
        ))
    })?;
    let stdin = child.stdin.take().expect("stdin was piped");

    // Stream the interleaved PCM on a scoped writer thread so a full stdin pipe
    // cannot deadlock against ffmpeg making progress, without copying the whole
    // buffer or requiring a `'static` bound.
    let output = thread::scope(|scope| {
        scope.spawn(|| {
            let mut writer = BufWriter::new(stdin);
            let _ = write_interleaved_le(&mut writer, audio);
            let _ = writer.flush();
            // `writer` (and the wrapped stdin) drops here, closing the pipe.
        });
        child.wait_with_output()
    });

    let output = output
        .map_err(|e| AudioError::Ffmpeg(format!("waiting for ffmpeg: {e}")))?;
    if output.status.success() {
        return Ok(());
    }
    let detail = String::from_utf8_lossy(&output.stderr);
    let detail = detail.trim();
    Err(AudioError::Ffmpeg(if detail.is_empty() {
        format!("ffmpeg exited with {}", output.status)
    } else {
        format!("ffmpeg exited with {}: {detail}", output.status)
    }))
}

/// Writes the samples interleaved, little-endian, in the native format — the
/// raw PCM ffmpeg reads from stdin.
fn write_interleaved_le<W: Write>(
    writer: &mut W,
    audio: &DecodedAudio,
) -> std::io::Result<()> {
    let ch = audio.channels();
    let frames = audio.frames();
    macro_rules! interleave {
        ($planes:expr, $to_bytes:expr) => {
            for i in 0..frames {
                for plane in $planes.iter().take(ch) {
                    writer.write_all(&$to_bytes(plane[i]))?;
                }
            }
        };
    }
    match audio.samples() {
        NativeBuf::U8(planes) => interleave!(planes, |x: u8| [x]),
        NativeBuf::S16(planes) => interleave!(planes, i16::to_le_bytes),
        NativeBuf::S32(planes) => interleave!(planes, i32::to_le_bytes),
        NativeBuf::F32(planes) => interleave!(planes, f32::to_le_bytes),
        NativeBuf::F64(planes) => interleave!(planes, f64::to_le_bytes),
    }
    Ok(())
}
