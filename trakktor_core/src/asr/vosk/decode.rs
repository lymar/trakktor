//! The transducer head (stateless decoder + joiner) and its search loops,
//! on the CPU.
//!
//! A faithful port of the reference sherpa-onnx transducer decoders:
//! **greedy search** (one argmax per encoder frame) and **modified beam
//! search** (per frame: batch joint over the active hypotheses, log-softmax,
//! top-k over hypothesis×token, duplicate hypotheses merged by log-add;
//! the final pick is length-normalized). The blank id is 0 and `<unk>` is
//! folded into the blank, as in the reference.
//!
//! Like the GigaAM RNN-T head, the whole head always runs in `f32` on the
//! CPU, shared by every runtime: the matrices are tiny next to the encoder
//! (the decoder is an embedding plus a two-token grouped convolution, the
//! joiner three small linears), the loop is sequential per token, and the
//! encoder output is read back once per chunk either way. The reference
//! runtime drives these same small graphs tensor-by-tensor, so CPU `f32` is
//! also the parity-faithful arithmetic.
//!
//! Decoding is deterministic: hypotheses live in insertion order and top-k
//! ties break by candidate index, so identical encoder output yields an
//! identical transcription on every runtime.

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use super::{error::VoskError, weights::ModelWeights};

/// The transducer blank id (fixed by the icefall recipe).
pub const BLANK_ID: u32 = 0;

/// Beam width of the reference decode scripts.
pub const DEFAULT_MAX_ACTIVE: usize = 10;

/// Cap on the memoized decoder outputs (~2 KB each). A streaming run keeps
/// one search alive across the whole file, and distinct contexts accumulate;
/// past the cap the memo is simply cleared — entries are pure functions of
/// the context, so dropping them costs recomputation, never correctness.
const MEMO_LIMIT: usize = 4096;

/// The search the transducer head runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decoding {
    /// One argmax per encoder frame.
    Greedy,
    /// Modified beam search with at most `max_active` hypotheses.
    Beam { max_active: usize },
}

/// Emitted tokens with the (global) encoder frame of each emission.
#[derive(Debug, Clone, Default)]
pub struct Emissions {
    pub token_ids: Vec<u32>,
    pub token_frames: Vec<usize>,
}

/// A linear map `y = W·x + b` held as plain `f32` rows.
struct Dense {
    weight: Vec<f32>, // [out, in], row-major
    bias: Vec<f32>,   // [out]
    in_dim: usize,
    out_dim: usize,
}

impl Dense {
    fn load(
        w: &ModelWeights,
        prefix: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self, VoskError> {
        let weight = w.get(&format!("{prefix}.weight"), &[out_dim, in_dim])?;
        let bias = w.get(&format!("{prefix}.bias"), &[out_dim])?;
        Ok(Self {
            weight: weight.data.clone(),
            bias: bias.data.clone(),
            in_dim,
            out_dim,
        })
    }

    /// `y[o] = b[o] + Σ_i W[o,i]·x[i]`, written into `out`.
    fn forward_into(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_dim);
        debug_assert_eq!(out.len(), self.out_dim);
        for (o, slot) in out.iter_mut().enumerate() {
            let row = &self.weight[o * self.in_dim..(o + 1) * self.in_dim];
            let dot: f32 = row.iter().zip(x).map(|(w, v)| w * v).sum();
            *slot = self.bias[o] + dot;
        }
    }
}

/// The transducer head: the stateless decoder (embedding + grouped
/// context convolution + ReLU + projection) and the joiner
/// (`output_linear(tanh(enc + dec))`).
pub struct TransducerHead {
    /// Token embedding, `[vocab, decoder_dim]`.
    embed: Vec<f32>,
    /// Context convolution, `[decoder_dim, group_width, context]`; absent
    /// for `context_size == 1`.
    conv: Option<Vec<f32>>,
    conv_group_width: usize,
    decoder_proj: Dense,
    out: Dense,
    vocab_size: usize,
    decoder_dim: usize,
    context_size: usize,
    joiner_dim: usize,
    /// `<unk>` id folded into the blank (never emitted).
    unk_id: Option<u32>,
}

impl TransducerHead {
    /// Loads the head's weights from the extracted tensor map.
    pub fn load(
        w: &ModelWeights,
        unk_id: Option<u32>,
    ) -> Result<Self, VoskError> {
        let c = &w.config;
        let embed = w
            .get("decoder.embedding.weight", &[c.vocab_size, c.decoder_dim])?
            .data
            .clone();
        let (conv, conv_group_width) = if c.context_size > 1 {
            let t = w.get_any("decoder.conv.weight")?;
            let [out, width, k] = t.dims[..] else {
                return Err(VoskError::InvalidModel(
                    "decoder.conv.weight is not 3-D".into(),
                ));
            };
            if out != c.decoder_dim ||
                k != c.context_size ||
                width == 0 ||
                !c.decoder_dim.is_multiple_of(width)
            {
                return Err(VoskError::InvalidModel(format!(
                    "decoder.conv.weight shape {:?}",
                    t.dims
                )));
            }
            (Some(t.data.clone()), width)
        } else {
            (None, 0)
        };
        Ok(Self {
            embed,
            conv,
            conv_group_width,
            decoder_proj: Dense::load(
                w,
                "decoder_proj",
                c.joiner_dim,
                c.decoder_dim,
            )?,
            out: Dense::load(w, "output_linear", c.vocab_size, c.joiner_dim)?,
            vocab_size: c.vocab_size,
            decoder_dim: c.decoder_dim,
            context_size: c.context_size,
            joiner_dim: c.joiner_dim,
            unk_id,
        })
    }

    /// Vocabulary size (logit width).
    pub fn vocab_size(&self) -> usize { self.vocab_size }

    /// Decoder context length.
    pub fn context_size(&self) -> usize { self.context_size }

    /// Whether a token id may be emitted (not blank, not `<unk>`).
    fn emits(&self, id: u32) -> bool {
        id != BLANK_ID && Some(id) != self.unk_id
    }

    /// The decoder output (already `decoder_proj`-projected) for a token
    /// context of `context_size` ids; negative ids embed as zeros (the
    /// initial context is `[-1, …, 0]`).
    fn decoder_out(&self, context: &[i64]) -> Vec<f32> {
        debug_assert_eq!(context.len(), self.context_size);
        let d = self.decoder_dim;
        // Embedding lookup, `[context, d]`; negative ids are zero vectors.
        let mut emb = vec![0.0f32; self.context_size * d];
        for (slot, &y) in emb.chunks_exact_mut(d).zip(context) {
            if y >= 0 {
                let row = y as usize * d;
                slot.copy_from_slice(&self.embed[row..row + d]);
            }
        }
        let hidden = match &self.conv {
            None => emb,
            Some(conv) => {
                // Grouped 1-D convolution over the context axis (kernel =
                // context, no padding, no bias), then ReLU.
                let width = self.conv_group_width;
                let k = self.context_size;
                let mut out = vec![0.0f32; d];
                for (c, slot) in out.iter_mut().enumerate() {
                    // Output channel c belongs to group g = c / (out/groups);
                    // with in == out channels, out/groups equals the group
                    // input width, so its inputs are
                    // [g·width, g·width + width).
                    let g = c / width;
                    let mut acc = 0.0f32;
                    for j in 0..width {
                        let in_ch = g * width + j;
                        for t in 0..k {
                            acc += conv[(c * width + j) * k + t] *
                                emb[t * d + in_ch];
                        }
                    }
                    *slot = acc.max(0.0);
                }
                out
            },
        };
        let mut projected = vec![0.0f32; self.joiner_dim];
        self.decoder_proj.forward_into(&hidden, &mut projected);
        projected
    }

    /// Joint logits for one (projected) encoder frame and decoder output.
    fn joint_into(&self, enc: &[f32], dec: &[f32], logits: &mut [f32]) {
        debug_assert_eq!(enc.len(), self.joiner_dim);
        let mut sum = vec![0.0f32; self.joiner_dim];
        for ((s, e), d) in sum.iter_mut().zip(enc).zip(dec) {
            *s = (e + d).tanh();
        }
        self.out.forward_into(&sum, logits);
    }
}

/// Log-add-exp of two natural-log values.
fn log_add(a: f64, b: f64) -> f64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    if lo == f64::NEG_INFINITY {
        return hi;
    }
    hi + (lo - hi).exp().ln_1p()
}

/// In-place log-softmax over a logits row (f32, as the reference).
fn log_softmax(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = row.iter().map(|&v| (v - max).exp()).sum();
    let log_sum = max + sum.ln();
    for v in row.iter_mut() {
        *v -= log_sum;
    }
}

/// One beam-search hypothesis.
#[derive(Debug, Clone)]
struct Hypothesis {
    /// Token ids, including the leading `[-1, …, 0]` context.
    ys: Vec<i64>,
    /// Encoder frame of each real emission.
    timestamps: Vec<usize>,
    /// Accumulated acoustic log-probability.
    log_prob: f64,
}

/// The search state carried across encoder-output blocks: one chunk for an
/// offline model, every chunk of the file for a streaming one.
pub struct DecodeState {
    decoding: Decoding,
    /// Greedy: the running context and its cached decoder output.
    context: Vec<i64>,
    dec_out: Vec<f32>,
    greedy: Emissions,
    /// Beam: active hypotheses in insertion order.
    hyps: Vec<Hypothesis>,
    /// Memoized decoder outputs per context (contexts repeat heavily).
    memo: HashMap<Vec<i64>, Vec<f32>>,
    /// Global frame offset of the next block.
    frame_offset: usize,
}

impl DecodeState {
    /// A fresh state: the blank context `[-1, …, 0]`.
    pub fn new(head: &TransducerHead, decoding: Decoding) -> Self {
        let mut context = vec![-1i64; head.context_size];
        *context.last_mut().expect("context_size >= 1") = i64::from(BLANK_ID);
        let dec_out = head.decoder_out(&context);
        let hyps = vec![Hypothesis {
            ys: context.clone(),
            timestamps: Vec::new(),
            log_prob: 0.0,
        }];
        Self {
            decoding,
            context,
            dec_out,
            greedy: Emissions::default(),
            hyps,
            memo: HashMap::new(),
            frame_offset: 0,
        }
    }

    /// Consumes one block of encoder output (`frames` rows of `joiner_dim`
    /// projected values), advancing the search.
    pub fn decode_block(
        &mut self,
        head: &TransducerHead,
        encoded: &[f32],
        frames: usize,
    ) {
        debug_assert_eq!(encoded.len(), frames * head.joiner_dim);
        match self.decoding {
            Decoding::Greedy => self.greedy_block(head, encoded, frames),
            Decoding::Beam { max_active } => {
                self.beam_block(head, encoded, frames, max_active)
            },
        }
        self.frame_offset += frames;
    }

    fn greedy_block(
        &mut self,
        head: &TransducerHead,
        encoded: &[f32],
        frames: usize,
    ) {
        let mut logits = vec![0.0f32; head.vocab_size];
        for t in 0..frames {
            let enc = &encoded[t * head.joiner_dim..(t + 1) * head.joiner_dim];
            head.joint_into(enc, &self.dec_out, &mut logits);
            let label = argmax(&logits);
            if head.emits(label) {
                self.greedy.token_ids.push(label);
                self.greedy.token_frames.push(self.frame_offset + t);
                self.context.rotate_left(1);
                *self.context.last_mut().expect("non-empty") = i64::from(label);
                self.dec_out = head.decoder_out(&self.context);
            }
        }
    }

    fn beam_block(
        &mut self,
        head: &TransducerHead,
        encoded: &[f32],
        frames: usize,
        max_active: usize,
    ) {
        let vocab = head.vocab_size;
        let context = head.context_size;
        for t in 0..frames {
            if self.memo.len() > MEMO_LIMIT {
                self.memo.clear();
            }
            let enc = &encoded[t * head.joiner_dim..(t + 1) * head.joiner_dim];
            // Scores of every (hypothesis, token) candidate: log-softmax of
            // the joint plus the hypothesis' running score.
            let mut scores: Vec<f64> =
                Vec::with_capacity(self.hyps.len() * vocab);
            let mut logits = vec![0.0f32; vocab];
            for hyp in &self.hyps {
                let ctx = hyp.ys[hyp.ys.len() - context..].to_vec();
                let dec_out = self
                    .memo
                    .entry(ctx)
                    .or_insert_with_key(|c| head.decoder_out(c));
                head.joint_into(enc, dec_out, &mut logits);
                log_softmax(&mut logits);
                scores.extend(
                    logits.iter().map(|&v| hyp.log_prob + f64::from(v)),
                );
            }
            // Top-k candidate indices, ties broken by index.
            let k = max_active.min(scores.len());
            let mut order: Vec<usize> = (0..scores.len()).collect();
            order.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
                scores[b]
                    .partial_cmp(&scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            order.truncate(k);
            order.sort_by(|&a, &b| {
                scores[b]
                    .partial_cmp(&scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });

            let prev = std::mem::take(&mut self.hyps);
            let mut index: HashMap<Vec<i64>, usize> = HashMap::new();
            for cand in order {
                let (h, token) = (cand / vocab, (cand % vocab) as u32);
                let mut hyp = prev[h].clone();
                if head.emits(token) {
                    hyp.ys.push(i64::from(token));
                    hyp.timestamps.push(self.frame_offset + t);
                }
                hyp.log_prob = scores[cand];
                match index.entry(hyp.ys.clone()) {
                    std::collections::hash_map::Entry::Occupied(slot) => {
                        let existing = &mut self.hyps[*slot.get()];
                        existing.log_prob =
                            log_add(existing.log_prob, hyp.log_prob);
                    },
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        slot.insert(self.hyps.len());
                        self.hyps.push(hyp);
                    },
                }
            }
        }
    }

    /// The best emissions so far (final once the last block was decoded).
    pub fn finish(self, head: &TransducerHead) -> Emissions {
        match self.decoding {
            Decoding::Greedy => self.greedy,
            Decoding::Beam { .. } => {
                let best = self
                    .hyps
                    .iter()
                    .max_by(|a, b| {
                        let na = a.log_prob / a.ys.len() as f64;
                        let nb = b.log_prob / b.ys.len() as f64;
                        na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .expect("at least one hypothesis");
                Emissions {
                    token_ids: best.ys[head.context_size..]
                        .iter()
                        .map(|&y| y as u32)
                        .collect(),
                    token_frames: best.timestamps.clone(),
                }
            },
        }
    }
}

/// The index of the largest value (first on ties, as the reference argmax).
fn argmax(values: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best as u32
}
