use super::*;

/// A stand-in runtime that returns what it was given, so the windowing and the
/// stitching can be checked without loading two gigabytes of weights.
struct Echo {
    /// Every window it was handed, in order.
    seen: Vec<usize>,
}

impl EnhanceModel for Echo {
    fn enhance_window(
        &mut self,
        samples: &[f32],
        _lost: &[bool],
    ) -> Result<Vec<f32>, EnhanceError> {
        self.seen.push(samples.len());
        // A real runtime returns `frames × 320`, which drops the alignment
        // remainder; imitate that exactly.
        Ok(samples[..samples.len() / HOP * HOP].to_vec())
    }

    fn runtime(&self) -> crate::enhance::Runtime {
        crate::enhance::Runtime::Candle
    }
}

fn ramp(len: usize) -> Vec<f32> {
    (0..len).map(|index| index as f32 / len as f32).collect()
}

fn run(samples: &[f32]) -> (Enhanced, Vec<usize>) {
    let mut model = Echo { seen: Vec::new() };
    let enhanced = enhance_samples(
        samples,
        &mut model,
        &EnhanceOptions {
            sample_rate: Some(SAMPLE_RATE),
            plc: false,
        },
        SAMPLE_RATE,
        &mut |_| {},
    )
    .unwrap();
    (enhanced, model.seen)
}

const SECOND: usize = SAMPLE_RATE as usize;

#[test]
fn a_short_recording_is_one_window() {
    let (enhanced, seen) = run(&ramp(3 * SECOND));
    assert_eq!(seen.len(), 1);
    assert_eq!(enhanced.samples.len(), 3 * SECOND);
    assert_eq!(enhanced.windows, 1);
}

#[test]
fn the_result_is_as_long_as_the_recording() {
    for seconds in [1usize, 5, 8, 9, 12, 13, 20, 31] {
        let (enhanced, _) = run(&ramp(seconds * SECOND));
        assert_eq!(
            enhanced.samples.len(),
            seconds * SECOND,
            "{seconds} s came back {} samples",
            enhanced.samples.len()
        );
    }
}

#[test]
fn windows_start_every_four_seconds() {
    let spans = windows(20 * SECOND);
    let starts: Vec<usize> =
        spans.iter().map(|&(start, _)| start / SECOND).collect();
    assert_eq!(starts, vec![0, 4, 8, 12]);
    // The tail (16 s to 20 s) is only four seconds, so it is glued onto the
    // window that starts at 12 s rather than becoming a fifth.
    assert_eq!(spans.last().unwrap(), &(12 * SECOND, 20 * SECOND));
}

#[test]
fn a_tail_of_at_least_six_seconds_becomes_its_own_window() {
    // 12 s to 19 s is seven seconds, which is enough context to run.
    let spans = windows(19 * SECOND);
    assert_eq!(spans.len(), 4);
    assert_eq!(spans.last().unwrap(), &(12 * SECOND, 19 * SECOND));
}

#[test]
fn a_shorter_tail_lengthens_the_window_before_it() {
    // 16 s to 22 s would be six seconds short of a window, so the tail is
    // glued onto the window that starts at 12 s, which becomes ten long.
    let spans = windows(22 * SECOND);
    assert_eq!(spans.len(), 4);
    assert_eq!(spans.last().unwrap(), &(12 * SECOND, 22 * SECOND));
}

#[test]
fn a_recording_shorter_than_a_window_is_still_one_window() {
    // The reference indexes an empty list here and crashes; a two-second file
    // is a perfectly ordinary thing to be handed.
    assert_eq!(windows(2 * SECOND), vec![(0, 2 * SECOND)]);
}

#[test]
fn an_echoing_runtime_returns_the_recording_unchanged() {
    // Stitching must take each window's middle, so an identity runtime has to
    // reproduce the input sample for sample (up to the peak match, which is a
    // no-op here because the peak is unchanged).
    let original = ramp(25 * SECOND);
    let (enhanced, _) = run(&original);
    assert_eq!(enhanced.samples.len(), original.len());
    let worst = original
        .iter()
        .zip(&enhanced.samples)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(worst < 1e-6, "stitching moved samples by {worst}");
}

#[test]
fn every_window_reaches_the_runtime_aligned() {
    let (_, seen) = run(&ramp(25 * SECOND));
    for len in seen {
        assert_eq!(
            len % HOP,
            PAD_REMAINDER,
            "a window of {len} was not aligned"
        );
    }
}

#[test]
fn the_output_keeps_the_recordings_peak() {
    let mut original = ramp(10 * SECOND);
    for sample in &mut original {
        *sample *= 0.25;
    }
    let (enhanced, _) = run(&original);
    let want = original.iter().fold(0f32, |acc, &v| acc.max(v.abs()));
    let got = enhanced
        .samples
        .iter()
        .fold(0f32, |acc, &v| acc.max(v.abs()));
    assert!((want - got).abs() < 1e-5, "peak {got} against {want}");
}

#[test]
fn an_empty_recording_is_an_error() {
    let mut model = Echo { seen: Vec::new() };
    let error = enhance_samples(
        &[],
        &mut model,
        &EnhanceOptions::default(),
        SAMPLE_RATE,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(matches!(error, EnhanceError::Empty));
}

#[test]
fn progress_counts_every_window_once() {
    let mut model = Echo { seen: Vec::new() };
    let mut reported = Vec::new();
    let enhanced = enhance_samples(
        &ramp(25 * SECOND),
        &mut model,
        &EnhanceOptions::default(),
        SAMPLE_RATE,
        &mut |progress| reported.push(progress.window),
    )
    .unwrap();
    assert_eq!(reported, (1..=enhanced.windows).collect::<Vec<_>>());
}

#[test]
fn a_lower_output_rate_resamples() {
    let mut model = Echo { seen: Vec::new() };
    let enhanced = enhance_samples(
        &ramp(4 * SECOND),
        &mut model,
        &EnhanceOptions::default(),
        8000,
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(enhanced.sample_rate, 8000);
    let seconds = enhanced.samples.len() as f64 / 8000.0;
    assert!((seconds - 4.0).abs() < 0.05, "{seconds} s");
}
