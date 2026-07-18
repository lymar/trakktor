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
