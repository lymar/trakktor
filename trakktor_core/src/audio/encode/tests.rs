use tempfile::tempdir;

use super::{Format, write};
use crate::audio::{AudioError, decode::DecodedAudio, pipeline::NativeBuf};

fn mono_s16(samples: Vec<i16>) -> DecodedAudio {
    DecodedAudio::from_parts(
        16_000,
        None,
        Some(16),
        NativeBuf::S16(vec![samples]),
    )
}

fn mono_f32(samples: Vec<f32>) -> DecodedAudio {
    DecodedAudio::from_parts(16_000, None, None, NativeBuf::F32(vec![samples]))
}

#[test]
fn wav_s16_round_trips_bit_exact() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("a.wav");
    let samples = vec![0i16, 1, -1, 32_767, -32_768, 100, -100, 5];
    write(&path, &mono_s16(samples.clone()), Format::Wav).unwrap();

    let mut reader = hound::WavReader::open(&path).unwrap();
    assert_eq!(reader.spec().bits_per_sample, 16);
    assert_eq!(reader.spec().channels, 1);
    assert_eq!(reader.spec().sample_rate, 16_000);
    let read: Vec<i16> = reader.samples::<i16>().map(Result::unwrap).collect();
    assert_eq!(read, samples);
}

#[test]
fn flac_s16_writes_a_valid_stream() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("a.flac");
    let samples: Vec<i16> =
        (0..4096).map(|i| ((i % 200) - 100) as i16).collect();
    write(&path, &mono_s16(samples), Format::Flac).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[0..4], b"fLaC", "FLAC stream marker");
    assert!(bytes.len() > 4);
}

#[test]
fn flac_rejects_float_sources_pointing_at_wav() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("a.flac");
    let err = write(&path, &mono_f32(vec![0.0, 0.5, -0.5]), Format::Flac)
        .unwrap_err();
    assert!(matches!(err, AudioError::UnsupportedEncoding(_)));
}

/// The encoder in use codes loud, noise-like 24-bit material as hundreds of
/// bytes per sample (its Rice parameter is capped at 14, and it prefers that
/// over storing the samples verbatim). The guard has to catch it.
#[test]
fn flac_refuses_output_larger_than_the_pcm_it_encodes() {
    let mut state = 1u64;
    let peak = f64::from((1u32 << 23) - 1);
    let samples: Vec<i32> = (0..200_000)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let noise = ((state >> 33) as f64 / f64::from(1u32 << 31)) - 1.0;
            ((0.3 * noise * peak).round() as i32) << 8
        })
        .collect();
    let audio = DecodedAudio::from_parts(
        16_000,
        None,
        Some(24),
        NativeBuf::S32(vec![samples]),
    );
    let dir = tempdir().unwrap();
    let path = dir.path().join("noise.flac");
    let err = write(&path, &audio, Format::Flac).unwrap_err();
    let message = err.to_string();
    assert!(
        matches!(err, AudioError::UnsupportedEncoding(_)),
        "{message}"
    );
    assert!(message.contains("Write WAV instead"), "{message}");
    // Nothing half-written is left behind.
    assert!(!path.exists());
}

/// The same guard must not stand in the way of audio that simply does not
/// compress much: sixteen-bit noise encodes correctly, at a shade under the
/// raw size, and has to be written.
#[test]
fn flac_still_writes_audio_that_barely_compresses() {
    let mut state = 7u64;
    let samples: Vec<i16> = (0..200_000)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let noise = ((state >> 33) as f64 / f64::from(1u32 << 31)) - 1.0;
            (0.3 * noise * 32_767.0) as i16
        })
        .collect();
    let dir = tempdir().unwrap();
    let path = dir.path().join("noise16.flac");
    write(&path, &mono_s16(samples), Format::Flac).unwrap();
    let size = std::fs::metadata(&path).unwrap().len();
    assert!(size > 0 && size < 200_000 * 2, "{size} bytes");
}
