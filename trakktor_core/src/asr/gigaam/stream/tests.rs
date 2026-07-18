//! Hermetic tests of the chunk planner's rolling-commit logic. End-to-end
//! streaming (real VAD + model) is exercised by the `#[ignore]` test at the
//! bottom.

use super::{COMMIT_HORIZON_S, COMMIT_MARGIN_S, ChunkPlanner};
use crate::asr::gigaam::segment::{Interval, MAX_DURATION_S};

fn iv(start: f64, end: f64) -> Interval { Interval { start, end } }

#[test]
fn no_commit_before_the_horizon() {
    let mut planner = ChunkPlanner::new();
    let mut t = 0.0;
    while t + 10.0 < COMMIT_HORIZON_S - 20.0 {
        assert!(planner.push(iv(t, t + 10.0)).is_empty());
        t += 12.0;
    }
    assert!(planner.earliest_pending_start().is_some());
}

#[test]
fn crossing_the_horizon_commits_a_stable_prefix() {
    let mut planner = ChunkPlanner::new();
    let mut committed = Vec::new();
    let mut t = 0.0;
    while committed.is_empty() {
        committed.extend(planner.push(iv(t, t + 10.0)));
        t += 12.0;
    }
    let frontier = t - 12.0 + 10.0; // end of the interval that triggered
    for &(start, end) in &committed {
        assert!(end - start <= MAX_DURATION_S + 1e-9);
        assert!(
            end <= frontier - COMMIT_MARGIN_S + 1e-9,
            "committed chunk {start}..{end} inside the margin"
        );
    }
    // The uncommitted tail is retained for the next round.
    let pending_start = planner.earliest_pending_start().unwrap();
    let last_end = committed.last().unwrap().1;
    assert!(pending_start >= last_end);
}

#[test]
fn poll_commits_pending_speech_after_long_silence() {
    let mut planner = ChunkPlanner::new();
    assert!(planner.push(iv(3.0, 9.0)).is_empty());
    // Silence rolls the frontier far past the speech: commit without new
    // intervals so the PCM behind it can be released.
    assert!(planner.poll(9.0 + COMMIT_MARGIN_S - 1.0).is_empty());
    let committed = planner.poll(3.0 + COMMIT_HORIZON_S + 1.0);
    assert_eq!(committed, vec![(3.0, 9.0)]);
    assert!(planner.earliest_pending_start().is_none());
}

#[test]
fn a_cut_inside_a_presplit_interval_keeps_the_tail() {
    let mut planner = ChunkPlanner::new();
    // One continuous 200 s speech interval: the DP presplits it; the commit
    // boundary falls inside the interval and the tail must stay pending.
    let committed = planner.push(iv(0.0, 200.0));
    assert!(!committed.is_empty());
    let boundary = committed.last().unwrap().1;
    assert!(boundary <= 200.0 - COMMIT_MARGIN_S + 1e-9);
    let pending = planner.earliest_pending_start().unwrap();
    assert!(
        (pending - boundary).abs() < 1e-9,
        "tail starts at the boundary"
    );
    // Finishing yields the rest, tiling up to 200 s.
    let rest = planner.finish(200.0);
    assert!((rest.last().unwrap().1 - 200.0).abs() < 1e-9);
    let mut prev = boundary;
    for &(s, e) in &rest {
        assert!((s - prev).abs() < 1e-9, "gap in presplit tiling");
        assert!(e - s <= MAX_DURATION_S + 1e-9);
        prev = e;
    }
}

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
    let model = GigaamModel::load_ctc_cpu(
        &ckpt,
        config.encoder,
        config.mel,
        config.num_classes,
        Precision::F32,
    )
    .unwrap();
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

#[test]
fn commits_then_finish_cover_all_speech_in_order() {
    let mut planner = ChunkPlanner::new();
    let mut chunks = Vec::new();
    let mut t = 0.0;
    for _ in 0..60 {
        chunks.extend(planner.push(iv(t, t + 8.0)));
        t += 9.5;
    }
    chunks.extend(planner.finish(t));
    // Ordered, non-overlapping, hard cap respected, all speech covered.
    for pair in chunks.windows(2) {
        assert!(pair[1].0 >= pair[0].1 - 1e-9);
    }
    for &(s, e) in &chunks {
        assert!(e - s <= MAX_DURATION_S + 1e-9);
    }
    assert!((chunks.first().unwrap().0 - 0.0).abs() < 1e-9);
    assert!((chunks.last().unwrap().1 - (t - 9.5 + 8.0)).abs() < 1e-9);
}
