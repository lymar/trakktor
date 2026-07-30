//! The RNN-Transducer head and its greedy decoding, on the CPU.
//!
//! A faithful port of the reference `gigaam.decoder.RNNTHead`
//! (prediction network: embedding + LSTM; joint network: two linear
//! projections summed, ReLU, linear) and `gigaam.decoding.RNNTGreedyDecoding`.
//!
//! Unlike the CTC head — one matmul folded into the encoder's device pass —
//! transducer decoding is an inherently sequential per-token loop over tiny
//! matrices (the whole head is a few MB next to the encoder's hundreds), and
//! every step's argmax feeds the next step's input. Running it on a GPU would
//! mean hundreds of kernel launches and read-backs per chunk, so the head
//! always runs in `f32` on the CPU, shared by every runtime: the encoder
//! output is read back once per chunk (keeping the one-synchronization-per-
//! chunk property of the CTC path), and the same head code serves the candle
//! and burn runtimes identically. The reference computes its head in `f32`
//! as well, so this is also the parity-faithful precision.

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use super::{
    config::RnntConfig,
    error::GigaamError,
    runtime::{model_err, tensor_to_f32_parts},
};

/// The most non-blank tokens the greedy loop may emit on one encoder frame
/// before moving on (the reference `max_symbols_per_step`).
const MAX_SYMBOLS_PER_STEP: usize = 10;

/// A linear map `y = W·x + b` held as plain `f32` rows.
struct Dense {
    weight: Vec<f32>, // [out, in], row-major
    bias: Vec<f32>,   // [out]
    in_dim: usize,
    out_dim: usize,
}

impl Dense {
    fn load(
        map: &HashMap<String, candle_core::Tensor>,
        prefix: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self, GigaamError> {
        let weight =
            checked(map, &format!("{prefix}.weight"), &[out_dim, in_dim])?;
        let bias = checked(map, &format!("{prefix}.bias"), &[out_dim])?;
        Ok(Self {
            weight,
            bias,
            in_dim,
            out_dim,
        })
    }

    /// `y[o] = b[o] + Σ_i W[o,i]·x[i]`, written into `out`.
    fn forward_into(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_dim);
        debug_assert_eq!(out.len(), self.out_dim);
        for (o, out_val) in out.iter_mut().enumerate() {
            let row = &self.weight[o * self.in_dim..(o + 1) * self.in_dim];
            let dot: f32 = row.iter().zip(x).map(|(w, v)| w * v).sum();
            *out_val = self.bias[o] + dot;
        }
    }
}

/// One LSTM layer with PyTorch weight layout: the `[4H, in]` input and
/// `[4H, H]` recurrent matrices hold the input, forget, cell, and output
/// gates stacked in that order.
struct LstmLayer {
    w_ih: Vec<f32>, // [4H, in]
    w_hh: Vec<f32>, // [4H, H]
    b_ih: Vec<f32>, // [4H]
    b_hh: Vec<f32>, // [4H]
    in_dim: usize,
    hidden: usize,
}

impl LstmLayer {
    fn load(
        map: &HashMap<String, candle_core::Tensor>,
        prefix: &str,
        layer: usize,
        in_dim: usize,
        hidden: usize,
    ) -> Result<Self, GigaamError> {
        let gates = 4 * hidden;
        Ok(Self {
            w_ih: checked(
                map,
                &format!("{prefix}.weight_ih_l{layer}"),
                &[gates, in_dim],
            )?,
            w_hh: checked(
                map,
                &format!("{prefix}.weight_hh_l{layer}"),
                &[gates, hidden],
            )?,
            b_ih: checked(
                map,
                &format!("{prefix}.bias_ih_l{layer}"),
                &[gates],
            )?,
            b_hh: checked(
                map,
                &format!("{prefix}.bias_hh_l{layer}"),
                &[gates],
            )?,
            in_dim,
            hidden,
        })
    }

    /// One step: `(x, h, c)` -> new `(h, c)`, h returned via `h`/`c` in place.
    fn step(&self, x: &[f32], h: &mut [f32], c: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_dim);
        let hidden = self.hidden;
        let mut gates = vec![0.0f32; 4 * hidden];
        for (j, gate) in gates.iter_mut().enumerate() {
            let wi = &self.w_ih[j * self.in_dim..(j + 1) * self.in_dim];
            let wh = &self.w_hh[j * hidden..(j + 1) * hidden];
            let dot_i: f32 = wi.iter().zip(x).map(|(w, v)| w * v).sum();
            let dot_h: f32 = wh.iter().zip(h.iter()).map(|(w, v)| w * v).sum();
            *gate = self.b_ih[j] + self.b_hh[j] + dot_i + dot_h;
        }
        for k in 0..hidden {
            let i = sigmoid(gates[k]);
            let f = sigmoid(gates[hidden + k]);
            let g = gates[2 * hidden + k].tanh();
            let o = sigmoid(gates[3 * hidden + k]);
            c[k] = f * c[k] + i * g;
            h[k] = o * c[k].tanh();
        }
    }
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

/// The LSTM hidden state of the prediction network, one `(h, c)` pair per
/// layer. A fresh state is all zeros (the reference passes `None` to the
/// LSTM, which is the same thing).
#[derive(Clone)]
struct LstmState {
    h: Vec<Vec<f32>>,
    c: Vec<Vec<f32>>,
}

impl LstmState {
    fn zeros(layers: usize, hidden: usize) -> Self {
        Self {
            h: vec![vec![0.0; hidden]; layers],
            c: vec![vec![0.0; hidden]; layers],
        }
    }
}

/// The RNN-Transducer head: prediction network (embedding + LSTM) and joint
/// network, with the greedy decode loop over one chunk's encoder output.
pub struct RnntHead {
    /// Token embedding of the prediction network, `[num_classes, H]`.
    embed: Vec<f32>,
    lstm: Vec<LstmLayer>,
    /// Joint projection of the encoder frame, `d_model` -> `joint_hidden`.
    enc_proj: Dense,
    /// Joint projection of the prediction output, `H` -> `joint_hidden`.
    pred_proj: Dense,
    /// Joint output map, `joint_hidden` -> `num_classes`.
    out: Dense,
    pred_hidden: usize,
    d_model: usize,
    num_classes: usize,
    blank_id: u32,
}

impl RnntHead {
    /// Loads the head's weights from the checkpoint's `state_dict` tensors
    /// (under `head.*`), shape-checked against the given geometry.
    pub fn load(
        map: &HashMap<String, candle_core::Tensor>,
        d_model: usize,
        num_classes: usize,
        cfg: &RnntConfig,
    ) -> Result<Self, GigaamError> {
        let hidden = cfg.pred_hidden;
        let embed =
            checked(map, "head.decoder.embed.weight", &[num_classes, hidden])?;
        let lstm = (0..cfg.pred_rnn_layers)
            .map(|layer| {
                LstmLayer::load(map, "head.decoder.lstm", layer, hidden, hidden)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            embed,
            lstm,
            enc_proj: Dense::load(
                map,
                "head.joint.enc",
                cfg.joint_hidden,
                d_model,
            )?,
            pred_proj: Dense::load(
                map,
                "head.joint.pred",
                cfg.joint_hidden,
                hidden,
            )?,
            out: Dense::load(
                map,
                "head.joint.joint_net.1",
                num_classes,
                cfg.joint_hidden,
            )?,
            pred_hidden: hidden,
            d_model,
            num_classes,
            blank_id: num_classes as u32 - 1,
        })
    }

    /// One prediction-network step from the last emitted label (`None` before
    /// the first emission, which feeds a zero vector) over `state`, returning
    /// the joint-projected prediction output and the advanced state.
    fn predict(
        &self,
        last_label: Option<u32>,
        state: &LstmState,
    ) -> (Vec<f32>, LstmState) {
        let mut x = match last_label {
            Some(label) => {
                let row = label as usize * self.pred_hidden;
                self.embed[row..row + self.pred_hidden].to_vec()
            },
            None => vec![0.0; self.pred_hidden],
        };
        let mut state = state.clone();
        for (layer, lstm) in self.lstm.iter().enumerate() {
            lstm.step(&x, &mut state.h[layer], &mut state.c[layer]);
            x.copy_from_slice(&state.h[layer]);
        }
        let mut projected = vec![0.0f32; self.pred_proj.out_dim];
        self.pred_proj.forward_into(&x, &mut projected);
        (projected, state)
    }

    /// Projects every encoder frame through the joint's encoder projection,
    /// in parallel across frames.
    ///
    /// The greedy loop is sequential, but this input of its is not: each
    /// frame's projection is an independent matrix-vector product, and
    /// splitting the *frames* across threads leaves every row's accumulation
    /// order untouched — the results are bit-identical to the sequential
    /// loop, only computed sooner. This is the largest
    /// emission-independent share of the head's work, so it is the one part
    /// worth hoisting out of the loop.
    fn project_frames(&self, encoded: &[f32], enc_frames: usize) -> Vec<f32> {
        let out_dim = self.enc_proj.out_dim;
        let mut projected = vec![0.0f32; enc_frames * out_dim];
        let workers = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(enc_frames / 8);
        let project = |start: usize, rows: &mut [f32]| {
            for (j, row) in rows.chunks_mut(out_dim).enumerate() {
                let t = start + j;
                let frame = &encoded[t * self.d_model..(t + 1) * self.d_model];
                self.enc_proj.forward_into(frame, row);
            }
        };
        if workers <= 1 {
            project(0, &mut projected);
            return projected;
        }
        let per_worker = enc_frames.div_ceil(workers);
        std::thread::scope(|scope| {
            for (i, block) in
                projected.chunks_mut(per_worker * out_dim).enumerate()
            {
                scope.spawn(move || project(i * per_worker, block));
            }
        });
        projected
    }

    /// Greedy transducer decoding over one chunk's encoder output `[T', D]`
    /// (row-major `f32`), the port of the reference `RNNTGreedyDecoding`:
    /// per frame, emit tokens until the joint picks the blank (at most
    /// [`MAX_SYMBOLS_PER_STEP`]); the prediction network advances only on
    /// emission. Returns the emitted token ids and their frames.
    pub fn greedy(
        &self,
        encoded: &[f32],
        enc_frames: usize,
    ) -> (Vec<u32>, Vec<usize>) {
        debug_assert_eq!(encoded.len(), enc_frames * self.d_model);
        let mut ids = Vec::new();
        let mut frames = Vec::new();

        let mut last_label: Option<u32> = None;
        let mut state = LstmState::zeros(self.lstm.len(), self.pred_hidden);
        // The prediction output for (last_label, state); those advance only
        // on emission, so the value is reused across blank steps rather than
        // recomputed (the reference recomputes — same result, f32 is
        // deterministic).
        let mut predicted: Option<(Vec<f32>, LstmState)> = None;

        let enc_projected = self.project_frames(encoded, enc_frames);
        let out_dim = self.enc_proj.out_dim;
        let mut logits = vec![0.0f32; self.num_classes];
        for t in 0..enc_frames {
            let enc_frame = &enc_projected[t * out_dim..(t + 1) * out_dim];
            for _ in 0..MAX_SYMBOLS_PER_STEP {
                let (pred, next_state) = match predicted.take() {
                    Some(cached) => cached,
                    None => self.predict(last_label, &state),
                };
                // Joint: out(relu(enc + pred)); the log-softmax of the
                // reference cannot change the argmax and is skipped.
                let joint: Vec<f32> = enc_frame
                    .iter()
                    .zip(&pred)
                    .map(|(e, p)| (e + p).max(0.0))
                    .collect();
                self.out.forward_into(&joint, &mut logits);
                let label = argmax(&logits);

                if label == self.blank_id {
                    // Not consumed: the same prediction output feeds the
                    // next frame.
                    predicted = Some((pred, next_state));
                    break;
                }
                ids.push(label);
                frames.push(t);
                last_label = Some(label);
                state = next_state;
            }
        }
        (ids, frames)
    }
}

/// The index of the largest logit (first on ties, as the reference argmax).
fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best as u32
}

/// Reads a checkpoint tensor as flat `f32`, shape-checked.
fn checked(
    map: &HashMap<String, candle_core::Tensor>,
    key: &str,
    shape: &[usize],
) -> Result<Vec<f32>, GigaamError> {
    let (values, dims) = tensor_to_f32_parts(map, key)?;
    if dims != shape {
        return Err(model_err(
            key,
            format!("shape {dims:?}, expected {shape:?}"),
        ));
    }
    Ok(values)
}
