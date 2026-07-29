//! End-to-end streaming over the real VAD and model. The chunk planner's
//! rolling-commit logic is engine-independent and tested hermetically in
//! [`crate::asr::segment`].

/// Real end-to-end streaming over the long-form path: pushing the same audio
/// in odd-sized blocks and in one shot must produce identical transcriptions
/// (commits are driven by the speech timeline, not block boundaries), and the
/// retained PCM must stay bounded far below the audio length.
#[test]
#[ignore = "needs ~/.cache/gigaam/v3_ctc.ckpt and tmp/sample.mp3"]
fn blockwise_equals_oneshot_and_memory_stays_bounded() {
    use crate::asr::gigaam::{
        config::config_for,
        runtime::{GigaamModel, Precision},
        transcribe::TranscribeOptions,
    };

    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let home = std::env::var("HOME").unwrap();
    let ckpt = std::path::PathBuf::from(home)
        .join(".cache/gigaam")
        .join("v3_ctc.ckpt");
    let config = config_for("v3_ctc").unwrap();
    let model = GigaamModel::load_cpu(&ckpt, &config, Precision::F32).unwrap();
    let tokenizer = config.tokenizer.build();

    // 300 s of the sample: enough to cross the commit horizon.
    let pcm =
        crate::audio::decode_to_mono_s16(&root.join("tmp/sample.mp3"), 16_000)
            .unwrap();
    let audio: Vec<f32> = pcm[..(300 * 16_000).min(pcm.len())]
        .iter()
        .map(|&s| f32::from(s) / 32768.0)
        .collect();
    let options = TranscribeOptions::default();

    let run = |blocks: &[usize]| {
        let mut session = super::StreamTranscriber::new(
            &model,
            &tokenizer,
            options.clone(),
            None,
        )
        .unwrap();
        let mut max_buffered = 0usize;
        let mut pos = 0usize;
        let mut sizes = blocks.iter().cycle();
        while pos < audio.len() {
            let step = *sizes.next().unwrap();
            let end = (pos + step).min(audio.len());
            session.push(&audio[pos..end], &mut |_| {}).unwrap();
            max_buffered = max_buffered.max(session.buffered_samples());
            pos = end;
        }
        let out = session.finish(&mut |_| {}).unwrap();
        (out, max_buffered)
    };

    let (oneshot, _) = run(&[usize::MAX]);
    let (blockwise, max_buffered) = run(&[16_000, 7_001, 512, 250_000]);

    assert_eq!(oneshot.text, blockwise.text, "text must be identical");
    assert_eq!(oneshot.segments.len(), blockwise.segments.len());
    for (a, b) in oneshot.segments.iter().zip(&blockwise.segments) {
        assert_eq!(a.start.to_bits(), b.start.to_bits());
        assert_eq!(a.end.to_bits(), b.end.to_bits());
        assert_eq!(a.text, b.text);
    }
    assert!(!oneshot.text.is_empty());

    // Memory bound: far below the 300 s of audio (horizon + margin + VAD lag).
    let bound = 260 * 16_000;
    assert!(
        max_buffered < bound,
        "retained {max_buffered} samples, bound {bound}"
    );
    println!(
        "segments: {}, max buffered: {:.1} s",
        oneshot.segments.len(),
        max_buffered as f64 / 16_000.0
    );
}
