use super::*;

#[test]
fn pad_or_trim_pads_with_zeros() {
    let x = [1.0f32, 2.0, 3.0];
    assert_eq!(pad_or_trim(&x, 5), vec![1.0, 2.0, 3.0, 0.0, 0.0]);
}

#[test]
fn pad_or_trim_truncates() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    assert_eq!(pad_or_trim(&x, 2), vec![1.0, 2.0]);
}

#[test]
fn pcm_decode_scales_i16() {
    // -32768 -> -1.0, 0 -> 0.0, 32767 -> ~0.99997.
    let bytes = [0x00, 0x80, 0x00, 0x00, 0xff, 0x7f];
    let f = pcm_s16le_to_f32(&bytes);
    assert_eq!(f[0], -1.0);
    assert_eq!(f[1], 0.0);
    assert!((f[2] - 0.999_969).abs() < 1e-5);
}

/// Verifies the ffmpeg backend reproduces the committed PCM fixture.
/// Opt-in: it needs `ffmpeg` on `PATH` and the developer-local sample
/// file.
#[test]
#[ignore = "requires ffmpeg and the local sample; run with --ignored"]
fn ffmpeg_decode_matches_pcm_fixture() {
    let sample =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tmp/sample.mp3");
    let samples = FfmpegDecoder::default().decode(&sample).expect("decode");

    let want: Vec<f32> = include_bytes!("../testdata/sample_2s.pcm.bin")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    assert!(samples.len() >= want.len());
    assert_eq!(&samples[..want.len()], &want[..]);
}
