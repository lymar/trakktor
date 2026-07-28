use super::{chunk_budget, duration_frames, noise, timesteps};

#[test]
fn the_plain_schedule_runs_from_zero_to_one() {
    let times = timesteps(32, 0.0);
    assert_eq!(times.len(), 33);
    assert_eq!(times[0], 0.0);
    assert!((times[32] - 1.0).abs() < 1e-6);
    assert!(times.windows(2).all(|pair| pair[1] > pair[0]));
}

#[test]
fn sway_crowds_the_steps_toward_the_start() {
    let plain = timesteps(32, 0.0);
    let swayed = timesteps(32, -1.0);
    // The ends are fixed points of the transform; everything between moves
    // down, which means the early interval covers less of the trajectory.
    assert_eq!(swayed[0], 0.0);
    assert!((swayed[32] - 1.0).abs() < 1e-6);
    assert!(swayed[16] < plain[16]);
    assert!(swayed.windows(2).all(|pair| pair[1] > pair[0]));
}

#[test]
fn low_step_counts_use_the_pruned_schedule() {
    // Sixteen steps is tabulated: the first steps are half the plain spacing.
    let pruned = timesteps(16, 0.0);
    assert_eq!(pruned.len(), 17);
    assert!((pruned[1] - 1.0 / 32.0).abs() < 1e-6);
    // Seventeen is not tabulated, so it divides the interval evenly.
    let plain = timesteps(17, 0.0);
    assert!((plain[1] - 1.0 / 17.0).abs() < 1e-6);
}

#[test]
fn duration_follows_the_reference_rate() {
    // A reference of 800 frames for 200 bytes of transcript reads at 4 frames
    // per byte, so 50 bytes of new text should add 200 frames.
    assert_eq!(duration_frames(800, 200, 50, 801, 250, 1.0), 1000);
    // Half the speed, twice the room.
    assert_eq!(duration_frames(800, 200, 50, 801, 250, 0.5), 1200);
}

#[test]
fn very_short_text_ignores_the_requested_speed() {
    // Under ten bytes the reference forces a slow rate, so the estimate does
    // not depend on `speed`.
    let fast = duration_frames(800, 200, 6, 801, 250, 2.0);
    let slow = duration_frames(800, 200, 6, 801, 250, 0.5);
    assert_eq!(fast, slow);
    // 800 / 200 × 6 / 0.3 is 79.999… in binary floating point, and truncating
    // it is what the reference does too — hence 79 added frames, not 80.
    assert_eq!(fast, 879);
}

#[test]
fn duration_never_falls_below_what_has_to_fit() {
    // Text longer than the estimate still gets a frame per character plus one.
    assert_eq!(duration_frames(10, 200, 1, 11, 500, 1.0), 501);
}

#[test]
fn the_budget_shrinks_as_the_reference_grows() {
    let short = chunk_budget(200, 5.0, 1.0);
    let long = chunk_budget(200, 10.0, 1.0);
    assert!(short > long, "{short} should exceed {long}");
    // 200 bytes over 10 s is 20 bytes a second, and 12 s are left of the
    // window.
    assert_eq!(long, 240);
}

#[test]
fn a_reference_past_the_window_still_leaves_a_budget() {
    assert!(chunk_budget(200, 25.0, 1.0) > 0);
    assert!(chunk_budget(200, 0.0, 1.0) > 0);
}

#[test]
fn noise_is_seeded_and_shaped() {
    let a = noise(7, 10, 100);
    let b = noise(7, 10, 100);
    let c = noise(8, 10, 100);
    assert_eq!(a.len(), 1000);
    assert_eq!(a, b);
    assert_ne!(a, c);
    let mean = a.iter().sum::<f32>() / a.len() as f32;
    let variance = a.iter().map(|value| (value - mean).powi(2)).sum::<f32>() /
        a.len() as f32;
    assert!(mean.abs() < 0.15, "mean {mean}");
    assert!((variance - 1.0).abs() < 0.2, "variance {variance}");
}
