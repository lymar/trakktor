//! Hermetic tests of the RNN-T head on hand-built weights. Reference parity
//! runs in `runtime::tests` / `runtime_burn::tests` against the golden dumps.

use super::{Dense, LstmLayer, LstmState, RnntHead};

/// A head with the given geometry and all-zero weights.
fn zero_head(
    d_model: usize,
    hidden: usize,
    joint: usize,
    classes: usize,
) -> RnntHead {
    let dense = |out_dim: usize, in_dim: usize| Dense {
        weight: vec![0.0; out_dim * in_dim],
        bias: vec![0.0; out_dim],
        in_dim,
        out_dim,
    };
    RnntHead {
        embed: vec![0.0; classes * hidden],
        lstm: vec![LstmLayer {
            w_ih: vec![0.0; 4 * hidden * hidden],
            w_hh: vec![0.0; 4 * hidden * hidden],
            b_ih: vec![0.0; 4 * hidden],
            b_hh: vec![0.0; 4 * hidden],
            in_dim: hidden,
            hidden,
        }],
        enc_proj: dense(joint, d_model),
        pred_proj: dense(joint, hidden),
        out: dense(classes, joint),
        pred_hidden: hidden,
        d_model,
        num_classes: classes,
        blank_id: classes as u32 - 1,
    }
}

#[test]
fn lstm_step_matches_manual_formulas() {
    // H = 1: gates are scalars, so the update is checked against the written-
    // out LSTM equations (PyTorch gate order i, f, g, o).
    let (w_ii, w_if, w_ig, w_io) = (0.5f32, -0.3, 0.8, 0.2);
    let (w_hi, w_hf, w_hg, w_ho) = (0.1f32, 0.4, -0.6, 0.7);
    let (b_i, b_f, b_g, b_o) = (0.05f32, -0.1, 0.2, 0.0);
    let layer = LstmLayer {
        w_ih: vec![w_ii, w_if, w_ig, w_io],
        w_hh: vec![w_hi, w_hf, w_hg, w_ho],
        b_ih: vec![b_i, b_f, b_g, b_o],
        b_hh: vec![0.0; 4],
        in_dim: 1,
        hidden: 1,
    };

    let (x, h0, c0) = (0.9f32, -0.4f32, 0.3f32);
    let mut h = vec![h0];
    let mut c = vec![c0];
    layer.step(&[x], &mut h, &mut c);

    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let i = sigmoid(w_ii * x + w_hi * h0 + b_i);
    let f = sigmoid(w_if * x + w_hf * h0 + b_f);
    let g = (w_ig * x + w_hg * h0 + b_g).tanh();
    let o = sigmoid(w_io * x + w_ho * h0 + b_o);
    let c1 = f * c0 + i * g;
    let h1 = o * c1.tanh();
    assert!((c[0] - c1).abs() < 1e-6, "c: {} vs {c1}", c[0]);
    assert!((h[0] - h1).abs() < 1e-6, "h: {} vs {h1}", h[0]);
}

#[test]
fn greedy_emits_nothing_when_blank_dominates() {
    let mut head = zero_head(4, 3, 5, 3);
    head.out.bias[2] = 10.0; // the blank always wins
    let encoded = vec![0.5f32; 6 * 4];
    let (ids, frames) = head.greedy(&encoded, 6);
    assert!(ids.is_empty());
    assert!(frames.is_empty());
}

#[test]
fn greedy_caps_symbols_per_frame() {
    let mut head = zero_head(4, 3, 5, 3);
    head.out.bias[1] = 10.0; // token 1 always wins: emit forever
    let encoded = vec![0.0f32; 2 * 4];
    let (ids, frames) = head.greedy(&encoded, 2);
    // Capped at MAX_SYMBOLS_PER_STEP per frame, then the loop moves on.
    assert_eq!(ids, vec![1; 2 * super::MAX_SYMBOLS_PER_STEP]);
    let want: Vec<usize> = std::iter::repeat_n(0, super::MAX_SYMBOLS_PER_STEP)
        .chain(std::iter::repeat_n(1, super::MAX_SYMBOLS_PER_STEP))
        .collect();
    assert_eq!(frames, want);
}

#[test]
fn greedy_advances_prediction_state_only_on_emission() {
    // H = J = 1, two classes (token 0, blank 1). The LSTM is wired so its
    // hidden state tracks tanh of the last input: i ≈ 1, f ≈ 0, o ≈ 1,
    // g = tanh(x). Fresh state (zero input) gives h = 0; after emitting
    // token 0 the embedding (5.0) drives h ≈ 0.76.
    let mut head = zero_head(1, 1, 1, 2);
    head.embed = vec![5.0, 0.0]; // token 0 -> 5.0, blank -> 0.0
    head.lstm[0].w_ih = vec![0.0, 0.0, 1.0, 0.0]; // g = tanh(x)
    head.lstm[0].b_ih = vec![10.0, -10.0, 0.0, 10.0]; // i ≈ 1, f ≈ 0, o ≈ 1
    // pred = 10 - 20·h: 10 at the fresh state, ≈ -5.2 after an emission.
    head.pred_proj.weight = vec![-20.0];
    head.pred_proj.bias = vec![10.0];
    // logits = [relu(pred), 5]: token 0 wins fresh, the blank afterwards.
    head.out.weight = vec![1.0, 0.0];
    head.out.bias = vec![0.0, 5.0];

    let encoded = vec![0.0f32; 3];
    let (ids, frames) = head.greedy(&encoded, 3);
    assert_eq!(ids, vec![0], "one emission, then the state blocks further");
    assert_eq!(frames, vec![0]);
}

#[test]
fn predict_reuse_matches_recompute() {
    // The greedy loop caches the prediction output across blank frames; a
    // recomputation from the same (label, state) must give the same values.
    let mut head = zero_head(2, 3, 4, 4);
    for (i, w) in head.lstm[0].w_ih.iter_mut().enumerate() {
        *w = ((i * 13 + 5) % 17) as f32 / 17.0 - 0.5;
    }
    for (i, w) in head.embed.iter_mut().enumerate() {
        *w = ((i * 7 + 3) % 11) as f32 / 11.0;
    }
    let state = LstmState::zeros(1, 3);
    let (p1, s1) = head.predict(Some(2), &state);
    let (p2, s2) = head.predict(Some(2), &state);
    assert_eq!(p1, p2);
    assert_eq!(s1.h, s2.h);
    assert_eq!(s1.c, s2.c);
}
