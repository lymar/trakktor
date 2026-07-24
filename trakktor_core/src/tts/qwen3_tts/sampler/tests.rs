//! Sampling tests: the rules have to fire in the reference's order, and a seed
//! has to reproduce a run exactly.

use super::*;

#[test]
fn greedy_takes_the_most_likely_code() {
    let mut sampler = Sampler::new(0);
    let mut logits = vec![0.1, 0.9, 0.4, 0.2];

    let code = sampler.pick(&mut logits, Rule::Greedy, &[], 1.0, &[]);

    assert_eq!(code, 1);
}

#[test]
fn forbidden_codes_are_never_picked() {
    let mut sampler = Sampler::new(0);
    let mut logits = vec![0.1, 9.0, 0.4, 0.2];

    // The clear favourite is forbidden, so the runner-up wins.
    let code = sampler.pick(&mut logits, Rule::Greedy, &[], 1.0, &[1]);

    assert_eq!(code, 2);
}

#[test]
fn the_repetition_penalty_pushes_scores_toward_zero_from_both_sides() {
    let mut sampler = Sampler::new(0);
    // A positive score is divided, a negative one multiplied — both move away
    // from being picked.
    let mut logits = vec![2.0f32, -2.0];
    sampler.pick(&mut logits, Rule::Greedy, &[0, 1], 2.0, &[]);

    assert!((logits[0] - 1.0).abs() < 1e-6, "{:?}", logits);
    assert!((logits[1] + 4.0).abs() < 1e-6, "{:?}", logits);
}

#[test]
fn a_repeated_code_can_lose_its_lead_to_the_penalty() {
    let mut sampler = Sampler::new(0);
    let mut logits = vec![1.0f32, 0.9];

    // Code 0 leads, but having been said already it is penalized below code 1.
    let code = sampler.pick(&mut logits, Rule::Greedy, &[0], 2.0, &[]);

    assert_eq!(code, 1);
}

#[test]
fn top_k_restricts_the_draw_to_the_leaders() {
    let mut sampler = Sampler::new(7);
    // Only the two leaders may be drawn, however many times we try.
    for _ in 0..64 {
        let mut logits = vec![5.0, 4.9, -20.0, -21.0, -22.0];
        let code = sampler.pick(
            &mut logits,
            Rule::TopK {
                top_k: 2,
                temperature: 1.0,
            },
            &[],
            1.0,
            &[],
        );
        assert!(code == 0 || code == 1, "drew {code}");
    }
}

#[test]
fn the_same_seed_reproduces_a_run() {
    let draw = |seed: u64| -> Vec<u32> {
        let mut sampler = Sampler::new(seed);
        (0..16)
            .map(|_| {
                let mut logits = vec![1.0, 1.1, 0.9, 1.2, 0.8];
                sampler.pick(
                    &mut logits,
                    Rule::TopK {
                        top_k: 5,
                        temperature: 1.0,
                    },
                    &[],
                    1.0,
                    &[],
                )
            })
            .collect()
    };

    assert_eq!(draw(42), draw(42));
    // A different seed is (overwhelmingly likely to be) a different run; if it
    // were not, the seed would not be doing anything.
    assert_ne!(draw(42), draw(43));
}

#[test]
fn temperature_of_one_leaves_the_scores_alone() {
    let mut sampler = Sampler::new(0);
    let mut logits = vec![1.0f32, 2.0, 3.0];
    sampler.pick(
        &mut logits,
        Rule::TopK {
            top_k: 3,
            temperature: 1.0,
        },
        &[],
        1.0,
        &[],
    );
    assert_eq!(logits, vec![1.0, 2.0, 3.0]);
}
