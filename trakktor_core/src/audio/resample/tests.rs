use std::{
    io::{Read, Write},
    process::{Command, Stdio},
};

use super::*;

/// Deterministic full-entropy noise in (−0.5, 0.5).
///
/// Full-entropy mantissas keep quantization boundaries (and therefore any
/// tie-rounding differences) out of the comparison.
fn noise(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let v = ((state >> 40) as i64 - (1 << 23)) as f32 /
            f32::from(u16::MAX) /
            512.0;
        out.push(v);
    }
    out
}

fn run_all(r: &mut Resampler<f32>, x: &[f32], chunk: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for c in x.chunks(chunk) {
        out.extend(r.feed(&[c]).remove(0));
    }
    out.extend(r.finish().remove(0));
    out
}

#[test]
fn chunk_invariance() {
    let x = noise(44_100, 7);
    let mut one = Resampler::<f32>::new(16_000, 44_100, 1).unwrap();
    let reference = run_all(&mut one, &x, x.len());
    assert!(!reference.is_empty());
    for chunk in [1usize, 92, 93, 1152, 4096] {
        let mut r = Resampler::<f32>::new(16_000, 44_100, 1).unwrap();
        let out = run_all(&mut r, &x, chunk);
        assert_eq!(out.len(), reference.len(), "chunk {chunk}: length");
        let same = out
            .iter()
            .zip(&reference)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "chunk {chunk}: samples differ");
    }
}

#[test]
fn output_length_is_ceil_of_ideal() {
    for (in_rate, len) in [
        (44_100u32, 44_100usize * 2 + 123),
        (48_000, 48_000),
        (8_000, 8_003),
    ] {
        let x = noise(len, 3);
        let mut r = Resampler::<f32>::new(16_000, in_rate, 1).unwrap();
        let out = run_all(&mut r, &x, x.len());
        let expect = (len * 16_000).div_ceil(in_rate as usize);
        assert_eq!(out.len(), expect, "{in_rate} Hz, {len} samples");
    }
}

#[test]
fn too_short_input_yields_nothing() {
    // 44.1k → 16k has a 92-tap filter: fewer than filter_length + 1 samples
    // (even after the flush mirror) produce no output, as in the reference.
    let x = noise(40, 5);
    let mut r = Resampler::<f32>::new(16_000, 44_100, 1).unwrap();
    let out = run_all(&mut r, &x, x.len());
    assert!(out.is_empty());
}

#[test]
fn stereo_shares_the_state() {
    let l = noise(30_000, 11);
    let r_ch = noise(30_000, 12);
    let mut r = Resampler::<f32>::new(16_000, 44_100, 2).unwrap();
    let mut out = r.feed(&[&l, &r_ch]);
    let tail = r.finish();
    out[0].extend(tail[0].iter().copied());
    out[1].extend(tail[1].iter().copied());
    assert_eq!(out[0].len(), out[1].len());

    // Each channel equals a mono run of the same data.
    for (data, got) in [(&l, &out[0]), (&r_ch, &out[1])] {
        let mut mono = Resampler::<f32>::new(16_000, 44_100, 1).unwrap();
        let want = run_all(&mut mono, data, data.len());
        assert_eq!(&want, got);
    }
}

/// Feeds raw f32 mono through the local ffmpeg CLI: float in, float out
/// isolates the resampler stage (no rematrix, no quantization).
fn ffmpeg_resample_f32(x: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-f", "f32le", "-ac", "1", "-ar"])
        .arg(in_rate.to_string())
        .args(["-i", "-", "-f", "f32le", "-acodec", "pcm_f32le", "-ar"])
        .arg(out_rate.to_string())
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawning ffmpeg");
    let mut stdin = child.stdin.take().unwrap();
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let writer = std::thread::spawn(move || {
        stdin.write_all(&bytes).expect("writing to ffmpeg");
    });
    let mut raw = Vec::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_end(&mut raw)
        .expect("reading from ffmpeg");
    writer.join().unwrap();
    assert!(child.wait().unwrap().success());
    raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Bit-exact differential against the pinned local ffmpeg across common
/// rates, including 44 056 Hz (inexact phase step → the linear kernel).
#[test]
#[ignore = "requires ffmpeg on PATH; verifies against the local build"]
fn matches_ffmpeg_bit_for_bit() {
    for (in_rate, len) in [
        (44_100u32, 44_100usize * 3 + 17),
        (48_000, 48_000 * 2 + 1),
        (22_050, 22_050 * 2),
        (11_025, 11_025 + 7),
        (8_000, 8_000 * 2 + 5),
        (24_000, 24_000),
        (32_000, 32_000 + 9),
        (44_056, 44_056),
    ] {
        let x = noise(len, u64::from(in_rate));
        let mut r = Resampler::<f32>::new(16_000, in_rate, 1).unwrap();
        let ours = run_all(&mut r, &x, 1152);
        let theirs = ffmpeg_resample_f32(&x, in_rate, 16_000);
        assert_eq!(ours.len(), theirs.len(), "{in_rate} Hz: length");
        let diff = ours
            .iter()
            .zip(&theirs)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            diff,
            0,
            "{in_rate} Hz: {diff}/{} samples differ",
            ours.len()
        );
    }
}
