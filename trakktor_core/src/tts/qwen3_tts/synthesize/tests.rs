//! Tests for the parts of the driver that do not need weights.

use super::*;

fn config() -> ModelConfig {
    ModelConfig::parse(include_str!("../config/testdata/custom_voice_0b6.json"))
        .expect("parses")
}

#[test]
fn the_control_tail_of_the_vocabulary_is_forbidden_except_the_stop_code() {
    let cfg = config();
    let forbidden = forbidden_codes(&cfg);

    // Audio codes live below the predictor's vocabulary and stay reachable.
    assert!(!forbidden.contains(&0));
    assert!(!forbidden.contains(&2047));
    // The control tail is closed off …
    assert!(forbidden.contains(&2048));
    assert!(forbidden.contains(&3071));
    assert!(forbidden.contains(&cfg.talker.codec_bos_id));
    assert!(forbidden.contains(&cfg.talker.codec_pad_id));
    // … except the code that ends the speech, which the model needs to stop.
    assert!(!forbidden.contains(&cfg.talker.codec_eos_token_id));
}

#[test]
fn every_piece_samples_from_its_own_derived_seed() {
    let base = Sampling::TopK {
        top_k: 50,
        temperature: 0.9,
        repetition_penalty: 1.05,
        seed: 7,
    };

    let seed_of = |sampling: Sampling| match sampling {
        Sampling::TopK { seed, .. } => seed,
        Sampling::Greedy => panic!("expected sampling"),
    };

    // The first piece speaks with the run's own seed, so a single-piece run is
    // the same file it always was.
    assert_eq!(seed_of(piece_sampling(base, 0)), 7);
    // Later pieces get their own streams, derived from it — the run stays
    // reproducible, but no two pieces draw the same numbers.
    let seeds: Vec<u64> =
        (0..4).map(|i| seed_of(piece_sampling(base, i))).collect();
    assert_eq!(seeds.len(), 4);
    assert!(
        seeds.iter().collect::<std::collections::HashSet<_>>().len() == 4,
        "{seeds:?}"
    );
    // Deriving is a function of the index alone: reruns repeat it.
    assert_eq!(seed_of(piece_sampling(base, 3)), seeds[3]);

    // Greedy has no seed to derive.
    assert!(matches!(
        piece_sampling(Sampling::Greedy, 5),
        Sampling::Greedy
    ));
}
