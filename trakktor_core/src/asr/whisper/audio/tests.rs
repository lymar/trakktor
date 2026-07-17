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

#[cfg(feature = "audio")]
mod builtin {
    use std::path::Path;

    use super::*;

    fn sample_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tmp/sample.mp3")
    }

    fn fixture_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/asr/whisper/testdata/sample_2s.pcm.bin")
    }

    /// Verifies the built-in decoder reproduces the committed PCM fixture
    /// bit for bit — the regression anchor of the whole audio pipeline.
    /// Opt-in: it needs the developer-local sample file.
    #[test]
    #[ignore = "requires the local sample; run with --ignored"]
    fn builtin_decode_matches_pcm_fixture() {
        let samples = BuiltinDecoder.decode(&sample_path()).expect("decode");

        let want: Vec<f32> = std::fs::read(fixture_path())
            .expect("reading the fixture")
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        assert!(samples.len() >= want.len());
        let same = samples[..want.len()]
            .iter()
            .zip(&want)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "decoded samples diverge from the fixture");
    }

    /// Regenerates the committed fixture from the current pipeline. Run
    /// explicitly (`--ignored write_sample_pcm_fixture`) only when the
    /// pipeline intentionally changes, and re-run the feature goldens after.
    #[test]
    #[ignore = "writes the fixture; run explicitly on intentional changes"]
    fn write_sample_pcm_fixture() {
        let samples = BuiltinDecoder.decode(&sample_path()).expect("decode");
        let two_seconds = &samples[..2 * SAMPLE_RATE];
        let bytes: Vec<u8> =
            two_seconds.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(fixture_path(), bytes).expect("writing the fixture");
    }
}
