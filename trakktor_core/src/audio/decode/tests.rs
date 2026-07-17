use std::{
    path::PathBuf,
    process::{Command, Stdio},
};

use super::*;

/// Builds a minimal PCM WAV in memory.
fn wav_s16(rate: u32, channels: u16, frames: &[i16]) -> Vec<u8> {
    let data_len = (frames.len() * 2) as u32;
    let byte_rate = rate * u32::from(channels) * 2;
    let block_align = channels * 2;
    let mut w = Vec::new();
    w.extend(b"RIFF");
    w.extend((36 + data_len).to_le_bytes());
    w.extend(b"WAVEfmt ");
    w.extend(16u32.to_le_bytes());
    w.extend(1u16.to_le_bytes()); // PCM
    w.extend(channels.to_le_bytes());
    w.extend(rate.to_le_bytes());
    w.extend(byte_rate.to_le_bytes());
    w.extend(block_align.to_le_bytes());
    w.extend(16u16.to_le_bytes());
    w.extend(b"data");
    w.extend(data_len.to_le_bytes());
    for s in frames {
        w.extend(s.to_le_bytes());
    }
    w
}

#[test]
fn wav_s16_decodes_natively() {
    let samples: Vec<i16> =
        (0..1000).map(|i| (i * 13 % 20011) as i16).collect();
    let wav = wav_s16(16_000, 1, &samples);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.wav");
    std::fs::write(&path, wav).unwrap();

    let decoded = decode_file(&path).unwrap();
    assert_eq!(decoded.sample_rate(), 16_000);
    assert_eq!(decoded.channels(), 1);
    assert_eq!(decoded.frames(), 1000);
    assert_eq!(decoded.format(), SampleFormat::S16);
    // Mono s16 at the same rate is the identity path.
    assert_eq!(decoded.to_mono_s16(16_000).unwrap(), samples);
}

#[test]
fn unknown_format_is_a_clean_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.bin");
    std::fs::write(&path, b"this is not audio at all").unwrap();
    match decode_file(&path) {
        Err(AudioError::UnsupportedFormat { .. }) => {},
        other => panic!("expected UnsupportedFormat, got {other:?}"),
    }
}

#[test]
fn missing_file_is_io() {
    match decode_file(Path::new("/nonexistent/nope.mp3")) {
        Err(AudioError::Io { .. }) => {},
        other => panic!("expected Io, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Differential acceptance vs the pinned local ffmpeg (dev-only).
// ---------------------------------------------------------------------------

fn repo_path(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel)
}

fn ffmpeg_file_to_mono_s16(path: &Path, out_rate: u32) -> Vec<i16> {
    let out = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(path)
        .args(["-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar"])
        .arg(out_rate.to_string())
        .arg("-")
        .stderr(Stdio::inherit())
        .output()
        .expect("running ffmpeg");
    assert!(out.status.success());
    out.stdout
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn assert_lossy_parity(ours: &[i16], theirs: &[i16], label: &str) {
    assert_eq!(ours.len(), theirs.len(), "{label}: length");
    let mut diff = 0usize;
    let mut max_d = 0i32;
    for (a, b) in ours.iter().zip(theirs) {
        let d = (i32::from(*a) - i32::from(*b)).abs();
        if d > 0 {
            diff += 1;
        }
        max_d = max_d.max(d);
    }
    let share = diff as f64 / ours.len() as f64;
    eprintln!(
        "{label}: {diff}/{} samples differ ({:.4}%), max |delta| = {max_d}",
        ours.len(),
        share * 100.0
    );
    assert!(max_d <= 1, "{label}: max |delta| {max_d} > 1 LSB");
    assert!(share <= 0.002, "{label}: {:.4}% > 0.2%", share * 100.0);
}

/// The acceptance gate for lossy inputs: the full mp3 sample end to end.
#[test]
#[ignore = "requires ffmpeg and the local sample; run with --ignored"]
fn mp3_sample_within_lossy_tolerance() {
    let sample = repo_path("tmp/sample.mp3");
    let ours = decode_to_mono_s16(&sample, 16_000).expect("decoding");
    let theirs = ffmpeg_file_to_mono_s16(&sample, 16_000);
    assert_lossy_parity(&ours, &theirs, "sample.mp3");
}

/// AAC in mp4: checks gapless/edit-list alignment against the reference on
/// a locally transcoded clip.
#[test]
#[ignore = "requires ffmpeg and the local sample; run with --ignored"]
fn m4a_alignment_and_tolerance() {
    let dir = tempfile::tempdir().unwrap();
    let m4a = dir.path().join("clip.m4a");
    let status = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(repo_path("tmp/sample.mp3"))
        .args(["-t", "30", "-c:a", "aac"])
        .arg(&m4a)
        .status()
        .expect("running ffmpeg");
    assert!(status.success());

    let ours = decode_to_mono_s16(&m4a, 16_000).expect("decoding");
    let theirs = ffmpeg_file_to_mono_s16(&m4a, 16_000);
    assert_lossy_parity(&ours, &theirs, "clip.m4a");
}

/// Lossless files must match bit for bit through the whole chain,
/// including a resampled flac.
#[test]
#[ignore = "requires ffmpeg on PATH"]
fn lossless_files_bit_for_bit() {
    let dir = tempfile::tempdir().unwrap();

    // A stereo 44.1k wav from deterministic noise, then a flac of it.
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut frames = Vec::with_capacity(44_100 * 4);
    for _ in 0..44_100 * 2 {
        for _ in 0..2 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            frames.push((state >> 48) as i16);
        }
    }
    let wav = dir.path().join("t.wav");
    std::fs::write(&wav, wav_s16(44_100, 2, &frames)).unwrap();
    let flac = dir.path().join("t.flac");
    let status = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(&wav)
        .arg(&flac)
        .status()
        .expect("running ffmpeg");
    assert!(status.success());

    for path in [&wav, &flac] {
        let ours = decode_to_mono_s16(path, 16_000).expect("decoding");
        let theirs = ffmpeg_file_to_mono_s16(path, 16_000);
        assert_eq!(ours.len(), theirs.len(), "{path:?}: length");
        let diff = ours.iter().zip(&theirs).filter(|(a, b)| a != b).count();
        assert_eq!(diff, 0, "{path:?}: {diff} samples differ");
    }
}
