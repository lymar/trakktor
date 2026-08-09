//! The network, on candle.
//!
//! # The two shapes it works in
//!
//! Everything convolutional is `[1, channels, time, frequency]`, and
//! everything recurrent or attentive is `[batch, steps, channels]` — the
//! difference being which of time and frequency is the sequence and which is
//! the batch. The two-stage block does both in turn, and the whole of its
//! bookkeeping is two permutes.
//!
//! Worth knowing, because the names in the checkpoint say the opposite: what
//! upstream calls the *time* transformer attends over **frequency** (with time
//! as the batch), and the *frequency* one attends over **time**. That is not a
//! reading of the architecture but of the code — its attention is built with
//! torch's default `batch_first = false`, so the first axis of what it is
//! handed is the sequence. The names are kept as upstream has them, because
//! they are what the checkpoint's tensors are called.
//!
//! # Why the convolutions are written out
//!
//! Every convolution in the dense blocks is `(2, 3)` with a dilation that
//! applies to **time alone**, and its padding is asymmetric — the whole
//! dilation at the front of the time axis and nothing at the back, which is
//! what makes those blocks causal. candle's `conv2d` takes one padding and one
//! dilation for both axes, so none of them can be called directly. Each is
//! expressed instead as two convolutions over frequency — one per time tap,
//! read at the dilation's spacing and summed — which is also what makes the
//! causality obvious, since both taps look backwards.
//!
//! The convolutions that are `(1, k)` need no such trick: fold time into the
//! batch and a `(1, k)` kernel *is* a one-dimensional convolution over
//! frequency.
//!
//! # What is not here
//!
//! The instance norms are, unavoidably: they take their statistics from the
//! window being enhanced, so unlike a batch norm there was nothing to fold into
//! the convolution at conversion time.
//!
//! The last two operations are not: the mask multiply and the phase's
//! arctangent happen on the host, shared with the burn runtime
//! ([`stft::apply`](super::super::stft::apply)).
//!
//! Ported from MP-SENet (MIT).

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, Module, VarBuilder};

use crate::enhance::mpsenet::config::{
    ATTENTION_HEADS, BINS, CHANNELS, CORE_WIDTH, DENSE_DEPTH, GRU_HIDDEN,
    INSTANCE_NORM_EPS, LEAKY_SLOPE, MASK_BETA, NORM_EPS, TS_BLOCKS,
};

/// Largest attention score matrix computed at once, in elements.
///
/// The score matrix of the pass over time is `frames² × heads` for **every**
/// frequency, which at an eight-second window is two thirds of a billion
/// floats. The batch is cut into pieces small enough to keep the peak here;
/// the arithmetic is unchanged, because a softmax over one row never looks at
/// another.
const ATTENTION_BUDGET: usize = 32 << 20;

/// Folds the time axis into the batch so a `(1, k)` kernel becomes a plain
/// one-dimensional convolution over frequency: `[b, c, t, f] → [b·t, c, f]`.
fn frames_as_batch(xs: &Tensor) -> Result<(Tensor, usize, usize)> {
    let (batch, channels, time, freq) = xs.dims4()?;
    let folded = xs.permute((0, 2, 1, 3))?.contiguous()?.reshape((
        batch * time,
        channels,
        freq,
    ))?;
    Ok((folded, batch, time))
}

/// The inverse of [`frames_as_batch`].
fn batch_as_frames(xs: &Tensor, batch: usize, time: usize) -> Result<Tensor> {
    let (_, channels, freq) = xs.dims3()?;
    xs.reshape((batch, time, channels, freq))?
        .permute((0, 2, 1, 3))?
        .contiguous()
}

/// Reverses a tensor along an arbitrary axis.
fn reverse_axis(xs: &Tensor, axis: usize) -> Result<Tensor> {
    let len = xs.dim(axis)?;
    let index = Tensor::from_vec(
        (0..len).rev().map(|i| i as u32).collect::<Vec<_>>(),
        len,
        xs.device(),
    )?;
    xs.index_select(&index, axis)
}

/// A rectifier with a small slope below zero, as the recurrence's projection
/// wants it.
fn leaky_relu(xs: &Tensor) -> Result<Tensor> {
    let positive = xs.relu()?;
    let negative = (xs - &positive)?;
    positive + (negative * LEAKY_SLOPE)?
}

/// A parametric rectifier with one slope per channel.
#[derive(Debug)]
struct Prelu {
    slope: Tensor,
}

impl Prelu {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            slope: vb.get(channels, "weight")?.reshape((1, channels, 1, 1))?,
        })
    }

    /// `xs` is `[batch, channels, time, freq]`, and so is the result.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let positive = xs.relu()?;
        let negative = (xs - &positive)?;
        positive + negative.broadcast_mul(&self.slope)?
    }
}

/// Normalization over one channel of one window — every frame and every bin of
/// it — with a learned scale and shift.
///
/// This is what an `InstanceNorm2d` does at inference: unlike a batch norm it
/// keeps no running statistics, so the numbers it divides by come from the
/// audio in front of it. That is also why a long recording cut into windows
/// cannot come out exactly as one pass would have: each window is normalized
/// by its own contents.
#[derive(Debug)]
struct InstanceNorm {
    weight: Tensor,
    bias: Tensor,
}

impl InstanceNorm {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(channels, "weight")?.reshape((1, channels, 1, 1))?,
            bias: vb.get(channels, "bias")?.reshape((1, channels, 1, 1))?,
        })
    }

    /// The statistics are taken in `f32` even for an `f16` model — a mean over
    /// a quarter of a million values is not something half precision should be
    /// asked to accumulate — and the scale and shift are applied back in the
    /// model's own dtype.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let wide = xs.to_dtype(DType::F32)?;
        let mean = wide.mean_keepdim(3)?.mean_keepdim(2)?;
        let centered = wide.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim(3)?.mean_keepdim(2)?;
        let normed = centered
            .broadcast_div(&variance.affine(1.0, INSTANCE_NORM_EPS)?.sqrt()?)?
            .to_dtype(dtype)?;
        normed
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
    }
}

/// A convolution over frequency alone — every `(1, k)` kernel in the network.
#[derive(Debug)]
struct FreqConv {
    conv: Conv1d,
}

impl FreqConv {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb
            .get((out_channels, in_channels, 1, kernel), "weight")?
            .reshape((out_channels, in_channels, kernel))?;
        let cfg = Conv1dConfig {
            stride,
            padding,
            ..Default::default()
        };
        Ok(Self {
            conv: Conv1d::new(weight, Some(vb.get(out_channels, "bias")?), cfg),
        })
    }

    /// `xs` is `[batch, channels, time, freq]`, and so is the result.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (folded, batch, time) = frames_as_batch(xs)?;
        batch_as_frames(&self.conv.forward(&folded)?, batch, time)
    }
}

/// A dense block's `(2, 3)` convolution: two convolutions over frequency, one
/// per time tap, read `dilation` frames apart and summed.
///
/// The frequency axis is padded by one on each side, as the reference pads it;
/// the time axis is padded by the whole dilation at the **front** and not at
/// all at the back, which is what makes the block causal.
#[derive(Debug)]
struct TimeFreqConv {
    taps: [Conv1d; 2],
    dilation: usize,
}

impl TimeFreqConv {
    fn load(
        in_channels: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get((CHANNELS, in_channels, 2, 3), "weight")?;
        let bias = vb.get(CHANNELS, "bias")?;
        let cfg = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let tap = |index: usize| -> Result<Conv1d> {
            let slice = weight
                .narrow(2, index, 1)?
                .reshape((CHANNELS, in_channels, 3))?
                .contiguous()?;
            // The bias belongs to the sum, so only one tap carries it.
            let bias = if index == 0 { Some(bias.clone()) } else { None };
            Ok(Conv1d::new(slice, bias, cfg))
        };
        Ok(Self {
            taps: [tap(0)?, tap(1)?],
            dilation,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, channels, time, freq) = xs.dims4()?;
        let history = Tensor::zeros(
            (batch, channels, self.dilation, freq),
            xs.dtype(),
            xs.device(),
        )?;
        let padded = Tensor::cat(&[&history, xs], 2)?;
        let mut sum: Option<Tensor> = None;
        for (tap, conv) in self.taps.iter().enumerate() {
            let shifted = padded.narrow(2, tap * self.dilation, time)?;
            let (folded, batch, frames) = frames_as_batch(&shifted)?;
            let out = batch_as_frames(&conv.forward(&folded)?, batch, frames)?;
            sum = Some(match sum {
                Some(acc) => (acc + out)?,
                None => out,
            });
        }
        sum.ok_or_else(|| {
            candle_core::Error::Msg("a kernel has at least one tap".into())
        })
    }
}

/// Convolution, instance norm, parametric rectifier — the triple every stage
/// of this network is built from.
#[derive(Debug)]
struct DenseLayer {
    conv: TimeFreqConv,
    norm: InstanceNorm,
    prelu: Prelu,
}

impl DenseLayer {
    fn load(
        in_channels: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            conv: TimeFreqConv::load(in_channels, dilation, vb.pp("1"))?,
            norm: InstanceNorm::load(CHANNELS, vb.pp("2"))?,
            prelu: Prelu::load(CHANNELS, vb.pp("3"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.prelu
            .forward(&self.norm.forward(&self.conv.forward(xs)?)?)
    }
}

/// Four convolutions, each of which sees everything the ones before it
/// produced as well as the block's input.
///
/// The dilation doubles from layer to layer — 1, 2, 4, 8 — so the last one
/// reaches fifteen frames back for a kernel two frames wide.
#[derive(Debug)]
struct DenseBlock {
    layers: Vec<DenseLayer>,
}

impl DenseBlock {
    fn load(vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("dense_block");
        Ok(Self {
            layers: (0..DENSE_DEPTH)
                .map(|index| {
                    DenseLayer::load(
                        CHANNELS * (index + 1),
                        1 << index,
                        vb.pp(index.to_string()),
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut skip = xs.clone();
        let mut out = xs.clone();
        for layer in &self.layers {
            out = layer.forward(&skip)?;
            skip = Tensor::cat(&[&out, &skip], 1)?;
        }
        Ok(out)
    }
}

/// The encoder: a pointwise lift into 64 channels, a dense block, and one
/// stride-2 convolution that halves the frequency axis.
#[derive(Debug)]
struct Encoder {
    lift: FreqConv,
    lift_norm: InstanceNorm,
    lift_prelu: Prelu,
    dense: DenseBlock,
    down: FreqConv,
    down_norm: InstanceNorm,
    down_prelu: Prelu,
}

impl Encoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let first = vb.pp("dense_conv_1");
        let second = vb.pp("dense_conv_2");
        Ok(Self {
            lift: FreqConv::load(2, CHANNELS, 1, 1, 0, first.pp("0"))?,
            lift_norm: InstanceNorm::load(CHANNELS, first.pp("1"))?,
            lift_prelu: Prelu::load(CHANNELS, first.pp("2"))?,
            dense: DenseBlock::load(vb.pp("dense_block"))?,
            down: FreqConv::load(CHANNELS, CHANNELS, 3, 2, 1, second.pp("0"))?,
            down_norm: InstanceNorm::load(CHANNELS, second.pp("1"))?,
            down_prelu: Prelu::load(CHANNELS, second.pp("2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self
            .lift_prelu
            .forward(&self.lift_norm.forward(&self.lift.forward(xs)?)?)?;
        let xs = self.dense.forward(&xs)?;
        self.down_prelu
            .forward(&self.down_norm.forward(&self.down.forward(&xs)?)?)
    }
}

/// A gated recurrent unit that runs both ways at once.
///
/// The two directions are stacked into a leading axis and stepped together, so
/// each step of the sequence costs one batched matmul and one set of
/// elementwise operations rather than two of each. On a CPU that saves little;
/// on a GPU, where a step of this size is entirely the cost of launching the
/// kernels, it halves the whole recurrence.
#[derive(Debug)]
struct BiGru {
    /// `[2, input, 3·hidden]`, the directions stacked.
    weight_ih: Tensor,
    /// `[2, hidden, 3·hidden]`.
    weight_hh: Tensor,
    /// `[2, 1, 3·hidden]`.
    bias_ih: Tensor,
    bias_hh: Tensor,
    hidden: usize,
}

impl BiGru {
    fn load(input: usize, hidden: usize, vb: &VarBuilder) -> Result<Self> {
        let pair = |name: &str, rows: usize, cols: usize| -> Result<Tensor> {
            let forward = vb.get((rows, cols), &format!("{name}_l0"))?;
            let backward =
                vb.get((rows, cols), &format!("{name}_l0_reverse"))?;
            Tensor::stack(
                &[forward.t()?.contiguous()?, backward.t()?.contiguous()?],
                0,
            )
        };
        let biases = |name: &str| -> Result<Tensor> {
            let forward = vb.get(3 * hidden, &format!("{name}_l0"))?;
            let backward = vb.get(3 * hidden, &format!("{name}_l0_reverse"))?;
            Tensor::stack(&[forward, backward], 0)?.reshape((2, 1, 3 * hidden))
        };
        Ok(Self {
            weight_ih: pair("weight_ih", 3 * hidden, input)?,
            weight_hh: pair("weight_hh", 3 * hidden, hidden)?,
            bias_ih: biases("bias_ih")?,
            bias_hh: biases("bias_hh")?,
            hidden,
        })
    }

    /// Runs `[batch, steps, input]` and returns `[batch, steps, 2·hidden]` —
    /// the forward pass's output and the backward pass's, side by side, which
    /// is the layout torch concatenates them in.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, steps, input) = xs.dims3()?;
        // The backward direction reads the sequence reversed, so both
        // directions can then be stepped with the same index.
        let both = Tensor::stack(&[xs, &reverse_axis(xs, 1)?], 0)?;
        let projected = both
            .reshape((2, batch * steps, input))?
            .matmul(&self.weight_ih)?
            .broadcast_add(&self.bias_ih)?
            .reshape((2, batch, steps, 3 * self.hidden))?;

        let mut state =
            Tensor::zeros((2, batch, self.hidden), xs.dtype(), xs.device())?;
        let mut outputs = Vec::with_capacity(steps);
        for step in 0..steps {
            let gates_x = projected.i((.., .., step, ..))?;
            let gates_h = state
                .matmul(&self.weight_hh)?
                .broadcast_add(&self.bias_hh)?;
            let reset = candle_nn::ops::sigmoid(
                &(gates_x.narrow(2, 0, self.hidden)? +
                    gates_h.narrow(2, 0, self.hidden)?)?,
            )?;
            let update = candle_nn::ops::sigmoid(
                &(gates_x.narrow(2, self.hidden, self.hidden)? +
                    gates_h.narrow(2, self.hidden, self.hidden)?)?,
            )?;
            // The candidate applies the reset gate to the *biased* recurrent
            // projection, which is torch's convention and not every library's.
            let candidate =
                (gates_x.narrow(2, 2 * self.hidden, self.hidden)? +
                    reset.mul(&gates_h.narrow(
                        2,
                        2 * self.hidden,
                        self.hidden,
                    )?)?)?
                .tanh()?;
            state =
                ((1.0 - &update)?.mul(&candidate)? + update.mul(&state)?)?;
            outputs.push(state.clone());
        }

        let stacked = Tensor::stack(&outputs, 2)?;
        let forward = stacked.i(0)?;
        let backward = reverse_axis(&stacked.i(1)?, 1)?;
        Tensor::cat(&[&forward, &backward], 2)
    }
}

/// Multi-head self-attention over `[batch, steps, channels]`.
#[derive(Debug)]
struct Attention {
    in_proj: Tensor,
    in_bias: Tensor,
    out_proj: Linear,
}

impl Attention {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let out = vb.pp("out_proj");
        Ok(Self {
            in_proj: vb
                .get((3 * dim, dim), "in_proj_weight")?
                .t()?
                .contiguous()?,
            in_bias: vb.get(3 * dim, "in_proj_bias")?,
            out_proj: Linear::new(
                out.get((dim, dim), "weight")?,
                Some(out.get(dim, "bias")?),
            ),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, steps, dim) = xs.dims3()?;
        let head = dim / ATTENTION_HEADS;
        let projected = xs
            .reshape((batch * steps, dim))?
            .matmul(&self.in_proj)?
            .broadcast_add(&self.in_bias)?
            .reshape((batch, steps, 3, dim))?;
        // torch scales the query, not the scores.
        let scale = (head as f64).powf(-0.5);
        let split = |which: usize| -> Result<Tensor> {
            projected
                .i((.., .., which, ..))?
                .reshape((batch, steps, ATTENTION_HEADS, head))?
                .permute((0, 2, 1, 3))?
                .contiguous()
        };
        let query = (split(0)? * scale)?;
        let key = split(1)?;
        let value = split(2)?;

        // The score matrix is `steps² × heads` per batch entry, which the pass
        // over time makes very large; the batch is walked in pieces that keep
        // it bounded.
        let per_entry = ATTENTION_HEADS * steps * steps;
        let chunk = (ATTENTION_BUDGET / per_entry.max(1)).clamp(1, batch);
        let mut pieces = Vec::with_capacity(batch.div_ceil(chunk));
        for start in (0..batch).step_by(chunk) {
            let take = chunk.min(batch - start);
            let shape = (take * ATTENTION_HEADS, steps, head);
            let q = query.narrow(0, start, take)?.reshape(shape)?;
            let k = key.narrow(0, start, take)?.reshape(shape)?;
            let v = value.narrow(0, start, take)?.reshape(shape)?;
            let scores = q.matmul(&k.transpose(1, 2)?.contiguous()?)?;
            let weights = candle_nn::ops::softmax_last_dim(&scores)?;
            pieces.push(weights.matmul(&v)?.reshape((
                take,
                ATTENTION_HEADS,
                steps,
                head,
            ))?);
        }
        let context = if pieces.len() == 1 {
            pieces.remove(0)
        } else {
            Tensor::cat(&pieces, 0)?
        };
        self.out_proj.forward(
            &context
                .permute((0, 2, 1, 3))?
                .contiguous()?
                .reshape((batch, steps, dim))?,
        )
    }
}

/// One transformer over `[batch, steps, channels]`: attention, then a
/// recurrence where a feed-forward block would usually be.
#[derive(Debug)]
struct Transformer {
    norm1: LayerNorm,
    attention: Attention,
    norm2: LayerNorm,
    gru: BiGru,
    project: Linear,
    norm3: LayerNorm,
}

impl Transformer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let norm = |vb: VarBuilder| -> Result<LayerNorm> {
            Ok(LayerNorm::new(
                vb.get(CHANNELS, "weight")?,
                vb.get(CHANNELS, "bias")?,
                NORM_EPS,
            ))
        };
        let ffn = vb.pp("ffn");
        let project = ffn.pp("linear");
        Ok(Self {
            norm1: norm(vb.pp("norm1"))?,
            attention: Attention::load(CHANNELS, vb.pp("attention"))?,
            norm2: norm(vb.pp("norm2"))?,
            gru: BiGru::load(CHANNELS, GRU_HIDDEN, &ffn.pp("gru"))?,
            project: Linear::new(
                project.get((CHANNELS, 2 * GRU_HIDDEN), "weight")?,
                Some(project.get(CHANNELS, "bias")?),
            ),
            norm3: norm(vb.pp("norm3"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let attended = self.attention.forward(&self.norm1.forward(xs)?)?;
        let xs = (xs + attended)?;
        let recurrent = self.gru.forward(&self.norm2.forward(&xs)?)?;
        let projected = self.project.forward(&leaky_relu(&recurrent)?)?;
        self.norm3.forward(&(xs + projected)?)
    }
}

/// One two-stage block: a transformer across frequency and one across time,
/// each added back to what went into it.
#[derive(Debug)]
struct TsBlock {
    across_frequency: Transformer,
    across_time: Transformer,
}

impl TsBlock {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            across_frequency: Transformer::load(vb.pp("time_transformer"))?,
            across_time: Transformer::load(vb.pp("freq_transformer"))?,
        })
    }

    /// `xs` is `[1, channels, time, frequency]`, and so is the result.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // `[1, c, t, f]` → `[t, f, c]`: time batches, frequency is the
        // sequence.
        let over_freq = xs.i(0)?.permute((1, 2, 0))?.contiguous()?;
        let over_freq =
            (&over_freq + self.across_frequency.forward(&over_freq)?)?;
        // The same tensor read the other way: frequency batches, time is the
        // sequence.
        let over_time = over_freq.permute((1, 0, 2))?.contiguous()?;
        let over_time = (&over_time + self.across_time.forward(&over_time)?)?;
        over_time.permute((2, 1, 0))?.contiguous()?.unsqueeze(0)
    }
}

/// The upsampler both decoders end with: a convolution to twice the channels,
/// unwoven into twice the bins.
///
/// It is a sub-pixel convolution, which is a transposed convolution written so
/// that the arithmetic is an ordinary one: predict two output bins per input
/// bin as two channels, then interleave them.
#[derive(Debug)]
struct SubPixel {
    conv: FreqConv,
}

impl SubPixel {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: FreqConv::load(
                CHANNELS,
                2 * CHANNELS,
                3,
                1,
                1,
                vb.pp("conv"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.conv.forward(xs)?;
        let (batch, _, time, freq) = out.dims4()?;
        out.reshape((batch, 2, CHANNELS, time, freq))?
            .permute((0, 2, 3, 4, 1))?
            .contiguous()?
            .reshape((batch, CHANNELS, time, freq * 2))
    }
}

/// The head both decoders share: a dense block, the upsampler, and its
/// normalization and rectifier.
#[derive(Debug)]
struct DecoderStem {
    dense: DenseBlock,
    up: SubPixel,
    norm: InstanceNorm,
    prelu: Prelu,
}

impl DecoderStem {
    fn load(vb: VarBuilder, conv: &str) -> Result<Self> {
        let up = vb.pp(conv);
        Ok(Self {
            dense: DenseBlock::load(vb.pp("dense_block"))?,
            up: SubPixel::load(up.pp("0"))?,
            norm: InstanceNorm::load(CHANNELS, up.pp("1"))?,
            prelu: Prelu::load(CHANNELS, up.pp("2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.dense.forward(xs)?;
        self.prelu
            .forward(&self.norm.forward(&self.up.forward(&xs)?)?)
    }
}

/// The magnitude decoder: a gain per bin, between zero and `β`.
#[derive(Debug)]
struct MaskDecoder {
    stem: DecoderStem,
    out: FreqConv,
    slope: Tensor,
}

impl MaskDecoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            stem: DecoderStem::load(vb.clone(), "mask_conv")?,
            out: FreqConv::load(CHANNELS, 1, 2, 1, 0, vb.pp("mask_conv.3"))?,
            slope: vb.pp("lsigmoid").get((BINS, 1), "slope")?.reshape(BINS)?,
        })
    }

    /// The result is `[bins, frames]`.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let raw = self.out.forward(&self.stem.forward(xs)?)?;
        // `[1, 1, t, f]` → `[f, t]`.
        let raw = raw.i((0, 0))?.t()?.contiguous()?;
        let scaled = raw.broadcast_mul(&self.slope.reshape((BINS, 1))?)?;
        candle_nn::ops::sigmoid(&scaled)? * MASK_BETA
    }
}

/// The phase decoder: two components per bin, which the host turns into an
/// angle.
#[derive(Debug)]
struct PhaseDecoder {
    stem: DecoderStem,
    real: FreqConv,
    imag: FreqConv,
}

impl PhaseDecoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            stem: DecoderStem::load(vb.clone(), "phase_conv")?,
            real: FreqConv::load(CHANNELS, 1, 2, 1, 0, vb.pp("phase_conv_r"))?,
            imag: FreqConv::load(CHANNELS, 1, 2, 1, 0, vb.pp("phase_conv_i"))?,
        })
    }

    /// Both results are `[bins, frames]`.
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let hidden = self.stem.forward(xs)?;
        let plane =
            |xs: Tensor| -> Result<Tensor> { xs.i((0, 0))?.t()?.contiguous() };
        Ok((
            plane(self.real.forward(&hidden)?)?,
            plane(self.imag.forward(&hidden)?)?,
        ))
    }
}

/// The whole network.
#[derive(Debug)]
pub struct Mpsenet {
    encoder: Encoder,
    blocks: Vec<TsBlock>,
    mask: MaskDecoder,
    phase: PhaseDecoder,
    device: Device,
    dtype: DType,
}

impl Mpsenet {
    /// Loads the converted checkpoint.
    ///
    /// # Errors
    ///
    /// Returns whatever candle reports when a tensor is missing or has a shape
    /// this network does not have.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            encoder: Encoder::load(vb.pp("dense_encoder"))?,
            blocks: (0..TS_BLOCKS)
                .map(|index| {
                    TsBlock::load(vb.pp("TSTransformer").pp(index.to_string()))
                })
                .collect::<Result<Vec<_>>>()?,
            mask: MaskDecoder::load(vb.pp("mask_decoder"))?,
            phase: PhaseDecoder::load(vb.pp("phase_decoder"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// The device the network is on.
    #[must_use]
    pub fn device(&self) -> &Device { &self.device }

    /// The network's input: the two planes stacked as channels, `[1, 2, t, f]`.
    fn input(
        &self,
        magnitude: &[f32],
        phase: &[f32],
        frames: usize,
    ) -> Result<Tensor> {
        let plane = |values: &[f32]| -> Result<Tensor> {
            Tensor::from_slice(values, (BINS, frames), &self.device)?
                .to_dtype(self.dtype)?
                .t()?
                .contiguous()
        };
        Tensor::stack(&[plane(magnitude)?, plane(phase)?], 0)?.unsqueeze(0)
    }

    /// What the network predicts for one window: a mask and the two components
    /// of a phase, each `[bins, frames]` in row-major order.
    ///
    /// # Errors
    ///
    /// Returns whatever candle reports when the computation fails.
    pub fn predict(
        &self,
        magnitude: &[f32],
        phase: &[f32],
        frames: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let mut hidden = self
            .encoder
            .forward(&self.input(magnitude, phase, frames)?)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        let mask = self.mask.forward(&hidden)?;
        let (real, imag) = self.phase.forward(&hidden)?;
        let host = |xs: Tensor| -> Result<Vec<f32>> {
            xs.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
        };
        Ok((host(mask)?, host(real)?, host(imag)?))
    }

    /// Every stage of one pass, for the parity tests.
    #[cfg(test)]
    pub(crate) fn stages(
        &self,
        magnitude: &[f32],
        phase: &[f32],
        frames: usize,
    ) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::new();
        let xs = self.input(magnitude, phase, frames)?;
        out.push(("net_input".to_owned(), xs.clone()));

        let encoder = &self.encoder;
        let lifted = encoder.lift_prelu.forward(
            &encoder.lift_norm.forward(&encoder.lift.forward(&xs)?)?,
        )?;
        out.push(("enc_conv1".to_owned(), lifted.clone()));
        let mut skip = lifted.clone();
        let mut dense = lifted;
        for (index, layer) in encoder.dense.layers.iter().enumerate() {
            dense = layer.forward(&skip)?;
            out.push((format!("enc_dense_layer{index}"), dense.clone()));
            skip = Tensor::cat(&[&dense, &skip], 1)?;
        }
        let mut hidden = encoder.down_prelu.forward(
            &encoder.down_norm.forward(&encoder.down.forward(&dense)?)?,
        )?;
        out.push(("enc_conv2".to_owned(), hidden.clone()));

        for (index, block) in self.blocks.iter().enumerate() {
            let over_freq = hidden.i(0)?.permute((1, 2, 0))?.contiguous()?;
            let over_freq =
                (&over_freq + block.across_frequency.forward(&over_freq)?)?;
            out.push((
                format!("ts{index}_time"),
                over_freq.permute((2, 0, 1))?.contiguous()?.unsqueeze(0)?,
            ));
            let over_time = over_freq.permute((1, 0, 2))?.contiguous()?;
            let over_time =
                (&over_time + block.across_time.forward(&over_time)?)?;
            hidden =
                over_time.permute((2, 1, 0))?.contiguous()?.unsqueeze(0)?;
            out.push((format!("ts{index}"), hidden.clone()));
        }

        out.push(("mask".to_owned(), self.mask.forward(&hidden)?));
        // The reference dumps the phase heads before it transposes them, so
        // these two go back the other way to be comparable.
        let (real, imag) = self.phase.forward(&hidden)?;
        out.push(("phase_r".to_owned(), real.t()?.contiguous()?));
        out.push(("phase_i".to_owned(), imag.t()?.contiguous()?));
        Ok(out)
    }
}

/// The width the transformers see, asserted where it is easy to check.
const _: () = assert!(CORE_WIDTH == 101);
