use super::*;

#[test]
fn the_same_seed_gives_the_same_numbers() {
    let first = Noise::new(7).sample(1000);
    let second = Noise::new(7).sample(1000);
    assert_eq!(first, second);
}

#[test]
fn a_different_seed_gives_different_numbers() {
    let first = Noise::new(7).sample(64);
    let second = Noise::new(8).sample(64);
    assert_ne!(first, second);
}

#[test]
fn one_long_draw_is_the_same_as_several_short_ones() {
    // The vocoder asks for one block per chunk and the prior for another; the
    // stream must not depend on how it is cut up.
    let whole = Noise::new(3).sample(300);
    let mut stream = Noise::new(3);
    let mut pieces = stream.sample(101);
    pieces.extend(stream.sample(199));
    assert_eq!(whole, pieces);
}

#[test]
fn it_is_standard_normal_enough_for_a_vocoder() {
    let values = Noise::new(1).sample(200_000);
    let mean =
        values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>() /
        values.len() as f64;
    assert!(mean.abs() < 0.01, "mean {mean}");
    assert!((variance - 1.0).abs() < 0.02, "variance {variance}");
    // Nothing absurd in the tails: a hundred thousand draws should not reach
    // six standard deviations, and must certainly stay finite.
    let worst = values.iter().fold(0f32, |worst, &v| worst.max(v.abs()));
    assert!(worst.is_finite() && worst < 6.0, "worst {worst}");
}
