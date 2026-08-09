//! The network, on candle.
//!
//! # Why the convolutions are written out
//!
//! Every convolution in GTCRN is **anisotropic** — `(1, 5)` over frequency
//! alone, or `(3, 3)` with a dilation that applies to time only — and half of
//! them are grouped, transposed, or both. candle's `conv2d` takes one padding,
//! one stride and one dilation for both axes, and its `conv_transpose2d` takes
//! no groups at all, so none of them can be called directly. They are expressed
//! instead in terms of `conv1d`, which does take a stride, a padding, a
//! dilation and groups:
//!
//! - a `(1, k)` convolution is a one-dimensional convolution over frequency,
//!   applied to every frame — fold time into the batch and it is exactly that;
//! - a transposed one of the same shape is zero-stuffing by the stride followed
//!   by an ordinary convolution with the kernel reversed;
//! - the depthwise `(3, 3)` is a sum of three frequency convolutions, one per
//!   time tap, read at the dilation's spacing — which also makes its causality
//!   obvious, since every tap looks backwards;
//! - a transposed convolution of stride one is an ordinary one with the kernel
//!   reversed along both axes, so the decoder reuses the encoder's code and
//!   reverses its kernels when they are loaded.
//!
//! Batch norms are gone: the conversion folded each into the convolution before
//! it, so what is left is a weight and a bias.
//!
//! Ported from GTCRN (MIT).

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, VarBuilder};

use crate::enhance::gtcrn::config::{
    BANDED, BINS, CHANNELS, CORE_WIDTH, DP_HIDDEN, DP_NORM_EPS, ERB_BANDS,
    ERB_LOW_BINS, GT_DILATIONS, MAG_EPS, SFE_KERNEL,
};

/// A parametric ReLU with one slope for all channels, as the checkpoint has it.
#[derive(Debug)]
struct Prelu {
    slope: Tensor,
}

impl Prelu {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            slope: vb.get(1, "weight")?.reshape(())?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let positive = xs.relu()?;
        let negative = (xs - &positive)?;
        positive + negative.broadcast_mul(&self.slope)?
    }
}

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

/// A convolution over frequency alone — the `(1, k)` kernels of the encoder.
#[derive(Debug)]
struct FreqConv {
    conv: Conv1d,
    /// Set when the checkpoint's kernel is a transposed one, which this
    /// implements as zero-stuffing plus a reversed ordinary convolution.
    transposed: Option<usize>,
}

impl FreqConv {
    /// Loads a forward convolution of `[out, in/groups, 1, kernel]`.
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb
            .get((out_channels, in_channels / groups, 1, kernel), "weight")?
            .reshape((out_channels, in_channels / groups, kernel))?;
        let cfg = Conv1dConfig {
            stride,
            padding,
            groups,
            ..Default::default()
        };
        Ok(Self {
            conv: Conv1d::new(weight, Some(vb.get(out_channels, "bias")?), cfg),
            transposed: None,
        })
    }

    /// Loads a transposed convolution, whose kernel is stored `[in, out/groups,
    /// 1, kernel]` and which is run as an ordinary convolution over a
    /// zero-stuffed input, with the kernel reversed and the padding turned
    /// inside out.
    fn load_transposed(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let stored =
            vb.get((in_channels, out_channels / groups, 1, kernel), "weight")?;
        // Grouped or not, the stored layout is input-major; an ordinary
        // convolution wants it output-major, with the taps reversed.
        let per_group_in = in_channels / groups;
        let per_group_out = out_channels / groups;
        let reshaped =
            stored.reshape((groups, per_group_in, per_group_out, kernel))?;
        let weight = reshaped
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((out_channels, per_group_in, kernel))?;
        let weight = reverse_last(&weight)?;
        let cfg = Conv1dConfig {
            stride: 1,
            padding: kernel - 1 - padding,
            groups,
            ..Default::default()
        };
        Ok(Self {
            conv: Conv1d::new(weight, Some(vb.get(out_channels, "bias")?), cfg),
            transposed: Some(stride),
        })
    }

    /// `xs` is `[batch, channels, time, freq]`, and so is the result.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (folded, batch, time) = frames_as_batch(xs)?;
        let folded = match self.transposed {
            Some(stride) if stride > 1 => zero_stuff(&folded, stride)?,
            _ => folded,
        };
        let out = self.conv.forward(&folded)?;
        batch_as_frames(&out, batch, time)
    }
}

/// Reverses a tensor along its last axis.
fn reverse_last(xs: &Tensor) -> Result<Tensor> {
    let last = xs.dim(D::Minus1)?;
    let index = Tensor::from_vec(
        (0..last).rev().map(|i| i as u32).collect::<Vec<_>>(),
        last,
        xs.device(),
    )?;
    xs.index_select(&index, D::Minus1)
}

/// Inserts `stride - 1` zeros between neighbouring samples of the last axis —
/// what a transposed convolution does before it convolves.
fn zero_stuff(xs: &Tensor, stride: usize) -> Result<Tensor> {
    let (batch, channels, len) = xs.dims3()?;
    let spread = xs.reshape((batch, channels, len, 1))?;
    let zeros = Tensor::zeros(
        (batch, channels, len, stride - 1),
        xs.dtype(),
        xs.device(),
    )?;
    // Interleave, then drop the trailing zeros: a transposed convolution of
    // `n` samples covers `(n - 1) · stride + 1` positions, not `n · stride`.
    Tensor::cat(&[&spread, &zeros], 3)?
        .reshape((batch, channels, len * stride))?
        .narrow(2, 0, (len - 1) * stride + 1)
}

/// The depthwise `(3, 3)` convolution, as three frequency convolutions read at
/// the dilation's spacing and summed.
#[derive(Debug)]
struct DepthwiseTimeFreq {
    /// One depthwise frequency convolution per time tap.
    taps: Vec<Conv1d>,
    dilation: usize,
}

impl DepthwiseTimeFreq {
    fn load(
        channels: usize,
        dilation: usize,
        reversed: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get((channels, 1, 3, 3), "weight")?;
        // A transposed convolution of stride one is an ordinary one with the
        // kernel reversed along both spatial axes.
        let weight = if reversed {
            reverse_last(&reverse_axis(&weight, 2)?)?
        } else {
            weight
        };
        let bias = vb.get(channels, "bias")?;
        let cfg = Conv1dConfig {
            padding: 1,
            groups: channels,
            ..Default::default()
        };
        let mut taps = Vec::with_capacity(3);
        for tap in 0..3 {
            let slice = weight
                .narrow(2, tap, 1)?
                .reshape((channels, 1, 3))?
                .contiguous()?;
            // The bias belongs to the sum, so only one tap carries it.
            let bias = if tap == 0 { Some(bias.clone()) } else { None };
            taps.push(Conv1d::new(slice, bias, cfg));
        }
        Ok(Self { taps, dilation })
    }

    /// `xs` is `[batch, channels, time + 2·dilation, freq]` — already padded at
    /// the front, which is what makes the block causal — and the result is
    /// `[batch, channels, time, freq]`.
    fn forward(&self, xs: &Tensor, time: usize) -> Result<Tensor> {
        let mut sum: Option<Tensor> = None;
        for (tap, conv) in self.taps.iter().enumerate() {
            let shifted = xs.narrow(2, tap * self.dilation, time)?;
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

/// A gated recurrent unit, in torch's gate order and bias convention.
#[derive(Debug)]
struct Gru {
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    hidden: usize,
}

impl Gru {
    fn load(
        input: usize,
        hidden: usize,
        suffix: &str,
        vb: &VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            weight_ih: vb
                .get((3 * hidden, input), &format!("weight_ih_l0{suffix}"))?
                .t()?
                .contiguous()?,
            weight_hh: vb
                .get((3 * hidden, hidden), &format!("weight_hh_l0{suffix}"))?
                .t()?
                .contiguous()?,
            bias_ih: vb.get(3 * hidden, &format!("bias_ih_l0{suffix}"))?,
            bias_hh: vb.get(3 * hidden, &format!("bias_hh_l0{suffix}"))?,
            hidden,
        })
    }

    /// Runs the sequence `[batch, steps, input]` and returns
    /// `[batch, steps, hidden]` together with the state it ends in.
    ///
    /// `carried` is where the previous chunk left off. This network's memory is
    /// far longer than any overlap worth paying for — measured, a thirty-second
    /// warm-up still leaves the output visibly different — so a chunk continues
    /// its predecessor rather than restarting near it.
    fn forward(
        &self,
        xs: &Tensor,
        backwards: bool,
        carried: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, steps, _) = xs.dims3()?;
        // Every step's input projection is one matmul over the whole sequence.
        let projected = xs
            .reshape((batch * steps, xs.dim(2)?))?
            .matmul(&self.weight_ih)?
            .broadcast_add(&self.bias_ih)?
            .reshape((batch, steps, 3 * self.hidden))?;
        let mut state = match carried {
            Some(state) => state.clone(),
            None => {
                Tensor::zeros((batch, self.hidden), xs.dtype(), xs.device())?
            },
        };
        let mut outputs = vec![state.clone(); steps];
        for index in 0..steps {
            let step = if backwards { steps - 1 - index } else { index };
            let gates_x = projected.i((.., step, ..))?;
            let gates_h = state
                .matmul(&self.weight_hh)?
                .broadcast_add(&self.bias_hh)?;
            let reset = candle_nn::ops::sigmoid(
                &(gates_x.narrow(1, 0, self.hidden)? +
                    gates_h.narrow(1, 0, self.hidden)?)?,
            )?;
            let update = candle_nn::ops::sigmoid(
                &(gates_x.narrow(1, self.hidden, self.hidden)? +
                    gates_h.narrow(1, self.hidden, self.hidden)?)?,
            )?;
            // The candidate applies the reset gate to the *biased* recurrent
            // projection, which is torch's convention and not every library's.
            let candidate =
                (gates_x.narrow(1, 2 * self.hidden, self.hidden)? +
                    reset.mul(&gates_h.narrow(
                        1,
                        2 * self.hidden,
                        self.hidden,
                    )?)?)?
                .tanh()?;
            state =
                ((1.0 - &update)?.mul(&candidate)? + update.mul(&state)?)?;
            outputs[step] = state.clone();
        }
        Ok((Tensor::stack(&outputs, 1)?, state))
    }
}

/// A grouped recurrent unit: the features are split in two and each half gets
/// its own recurrence, which is where most of the parameter saving comes from.
#[derive(Debug)]
struct GroupedGru {
    first: Vec<Gru>,
    second: Vec<Gru>,
    bidirectional: bool,
}

impl GroupedGru {
    fn load(
        input: usize,
        hidden: usize,
        bidirectional: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let load = |name: &str| -> Result<Vec<Gru>> {
            let vb = vb.pp(name);
            let mut out = vec![Gru::load(input / 2, hidden / 2, "", &vb)?];
            if bidirectional {
                out.push(Gru::load(input / 2, hidden / 2, "_reverse", &vb)?);
            }
            Ok(out)
        };
        Ok(Self {
            first: load("rnn1")?,
            second: load("rnn2")?,
            bidirectional,
        })
    }

    /// `xs` is `[batch, steps, input]`; the result is `[batch, steps, hidden]`
    /// for a one-way unit and twice that for a two-way one, plus the state each
    /// half ends in. A two-way unit is only ever used across frequency inside a
    /// frame, where there is nothing to carry, so only the one-way state is
    /// meaningful.
    fn forward(
        &self,
        xs: &Tensor,
        carried: Option<&[Tensor]>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let half = xs.dim(2)? / 2;
        let mut parts = Vec::new();
        let mut states = Vec::with_capacity(2);
        for (index, (group, xs)) in [
            (&self.first, xs.narrow(2, 0, half)?),
            (&self.second, xs.narrow(2, half, half)?),
        ]
        .into_iter()
        .enumerate()
        {
            let carried = carried.and_then(|states| states.get(index));
            let (forward, state) = group[0].forward(&xs, false, carried)?;
            states.push(state);
            parts.push(if self.bidirectional {
                Tensor::cat(
                    &[forward, group[1].forward(&xs, true, None)?.0],
                    2,
                )?
            } else {
                forward
            });
        }
        Ok((Tensor::cat(&parts, 2)?, states))
    }
}

/// A layer norm whose statistics run over the last **two** axes at once, with
/// an affine of that same shape — the reference normalizes a whole
/// frequency-by-channel plane, not each channel vector.
#[derive(Debug)]
struct PlaneNorm {
    weight: Tensor,
    bias: Tensor,
}

impl PlaneNorm {
    fn load(width: usize, channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get((width, channels), "weight")?,
            bias: vb.get((width, channels), "bias")?,
        })
    }

    /// `xs` is `[batch, steps, width, channels]`.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, steps, width, channels) = xs.dims4()?;
        let flat = xs.reshape((batch, steps, width * channels))?;
        let mean = flat.mean_keepdim(2)?;
        let centred = flat.broadcast_sub(&mean)?;
        let variance = centred.sqr()?.mean_keepdim(2)?;
        let normed =
            centred.broadcast_div(&(variance + DP_NORM_EPS)?.sqrt()?)?;
        normed
            .reshape((batch, steps, width, channels))?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
    }
}

/// Temporal recurrent attention: one scalar per channel and frame, worked out
/// from how much energy that channel carries, and multiplied back in.
#[derive(Debug)]
struct Tra {
    gru: Gru,
    fc: Linear,
}

impl Tra {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gru: Gru::load(channels, channels * 2, "", &vb.pp("att_gru"))?,
            fc: candle_nn::linear(channels * 2, channels, vb.pp("att_fc"))?,
        })
    }

    /// `xs` is `[batch, channels, time, freq]`, and so is the result.
    fn forward(
        &self,
        xs: &Tensor,
        carried: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let energy = xs.sqr()?.mean(D::Minus1)?;
        let (hidden, state) = self.gru.forward(
            &energy.transpose(1, 2)?.contiguous()?,
            false,
            carried,
        )?;
        let gate = candle_nn::ops::sigmoid(&self.fc.forward(&hidden)?)?;
        let out = xs.broadcast_mul(
            &gate.transpose(1, 2)?.unsqueeze(3)?.contiguous()?,
        )?;
        Ok((out, state))
    }
}

/// Subband feature extraction: every band is concatenated with its two
/// neighbours, so a convolution one band wide still sees a neighbourhood.
fn spread_subbands(xs: &Tensor) -> Result<Tensor> {
    let (batch, channels, time, freq) = xs.dims4()?;
    let padded = xs.pad_with_zeros(3, 1, 1)?;
    let mut parts = Vec::with_capacity(channels * SFE_KERNEL);
    for channel in 0..channels {
        for tap in 0..SFE_KERNEL {
            parts.push(padded.i((.., channel, .., tap..tap + freq))?);
        }
    }
    Tensor::stack(&parts, 1)?.reshape((
        batch,
        channels * SFE_KERNEL,
        time,
        freq,
    ))
}

/// One plain convolution block of the encoder or decoder.
#[derive(Debug)]
struct ConvBlock {
    conv: FreqConv,
    act: Option<Prelu>,
}

impl ConvBlock {
    fn load(
        in_channels: usize,
        out_channels: usize,
        groups: usize,
        transposed: bool,
        tanh: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = if transposed {
            FreqConv::load_transposed(
                in_channels,
                out_channels,
                5,
                2,
                2,
                groups,
                vb.pp("conv"),
            )?
        } else {
            FreqConv::load(
                in_channels,
                out_channels,
                5,
                2,
                2,
                groups,
                vb.pp("conv"),
            )?
        };
        Ok(Self {
            conv,
            act: if tanh {
                None
            } else {
                Some(Prelu::load(vb.pp("act"))?)
            },
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.conv.forward(xs)?;
        match &self.act {
            Some(prelu) => prelu.forward(&out),
            None => out.tanh(),
        }
    }
}

/// A grouped temporal convolution block: half the channels go through a
/// bottleneck with a causal dilated convolution in it, the other half is
/// carried across untouched, and the two are interleaved at the end.
#[derive(Debug)]
struct GtConvBlock {
    point1: FreqConv,
    point_act: Prelu,
    depth: DepthwiseTimeFreq,
    depth_act: Prelu,
    point2: FreqConv,
    tra: Tra,
    dilation: usize,
}

impl GtConvBlock {
    fn load(
        channels: usize,
        dilation: usize,
        transposed: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let half = channels / 2;
        let point = |in_c, out_c, name: &str| -> Result<FreqConv> {
            if transposed {
                FreqConv::load_transposed(in_c, out_c, 1, 1, 0, 1, vb.pp(name))
            } else {
                FreqConv::load(in_c, out_c, 1, 1, 0, 1, vb.pp(name))
            }
        };
        Ok(Self {
            point1: point(half * 3, channels, "point_conv1")?,
            point_act: Prelu::load(vb.pp("point_act"))?,
            depth: DepthwiseTimeFreq::load(
                channels,
                dilation,
                transposed,
                vb.pp("depth_conv"),
            )?,
            depth_act: Prelu::load(vb.pp("depth_act"))?,
            point2: point(channels, half, "point_conv2")?,
            tra: Tra::load(half, vb.pp("tra"))?,
            dilation,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        carried: Option<&BlockState>,
    ) -> Result<(Tensor, BlockState)> {
        let (_, channels, time, _) = xs.dims4()?;
        let half = channels / 2;
        let kept = xs.narrow(1, half, half)?;
        let worked = xs.narrow(1, 0, half)?.contiguous()?;

        let spread = spread_subbands(&worked)?;
        let hidden = self.point_act.forward(&self.point1.forward(&spread)?)?;
        // The only padding on the time axis, and it is entirely at the front:
        // that is what makes the block causal. Across a chunk boundary the
        // padding is not silence but the frames the previous chunk ended on.
        let history = 2 * self.dilation;
        let padded = match carried.map(|state| &state.history) {
            Some(tail) => Tensor::cat(&[tail, &hidden], 2)?,
            None => hidden.pad_with_zeros(2, history, 0)?,
        };
        let tail = padded.narrow(2, time, history)?.contiguous()?;
        let hidden = self
            .depth_act
            .forward(&self.depth.forward(&padded, time)?)?;
        let hidden = self.point2.forward(&hidden)?;
        let (hidden, attention) =
            self.tra.forward(&hidden, carried.map(|s| &s.attention))?;

        // Interleave the two halves, which is the shuffle that lets the next
        // block work on channels this one carried across.
        let mut parts = Vec::with_capacity(channels);
        for index in 0..half {
            parts.push(hidden.i((.., index, .., ..))?);
            parts.push(kept.i((.., index, .., ..))?);
        }
        Ok((
            Tensor::stack(&parts, 1)?,
            BlockState {
                history: tail,
                attention,
            },
        ))
    }
}

/// What one grouped temporal block carries from a chunk to the next: the frames
/// its causal convolution would otherwise pad with silence, and where its
/// attention's recurrence left off.
#[derive(Clone)]
pub(crate) struct BlockState {
    history: Tensor,
    attention: Tensor,
}

/// One dual-path block: a two-way recurrence across frequency inside a frame,
/// then a one-way recurrence across time within a band.
#[derive(Debug)]
struct DualPath {
    intra_rnn: GroupedGru,
    intra_fc: Linear,
    intra_norm: PlaneNorm,
    inter_rnn: GroupedGru,
    inter_fc: Linear,
    inter_norm: PlaneNorm,
}

impl DualPath {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            intra_rnn: GroupedGru::load(
                CHANNELS,
                DP_HIDDEN / 2,
                true,
                vb.pp("intra_rnn"),
            )?,
            intra_fc: candle_nn::linear(
                DP_HIDDEN,
                DP_HIDDEN,
                vb.pp("intra_fc"),
            )?,
            intra_norm: PlaneNorm::load(
                CORE_WIDTH,
                DP_HIDDEN,
                vb.pp("intra_ln"),
            )?,
            inter_rnn: GroupedGru::load(
                CHANNELS,
                DP_HIDDEN,
                false,
                vb.pp("inter_rnn"),
            )?,
            inter_fc: candle_nn::linear(
                DP_HIDDEN,
                DP_HIDDEN,
                vb.pp("inter_fc"),
            )?,
            inter_norm: PlaneNorm::load(
                CORE_WIDTH,
                DP_HIDDEN,
                vb.pp("inter_ln"),
            )?,
        })
    }

    /// `xs` is `[batch, channels, time, freq]`, and so is the result.
    ///
    /// Only the time-axis recurrence carries: the other runs across frequency
    /// inside a single frame and has nothing to remember.
    fn forward(
        &self,
        xs: &Tensor,
        carried: Option<&Vec<Tensor>>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let (batch, _, time, freq) = xs.dims4()?;
        // Frequency first, one sequence per frame.
        let plane = xs.permute((0, 2, 3, 1))?.contiguous()?;
        let sequence = plane.reshape((batch * time, freq, CHANNELS))?;
        let (intra, _) = self.intra_rnn.forward(&sequence, None)?;
        let intra = self.intra_fc.forward(&intra)?;
        let intra = self
            .intra_norm
            .forward(&intra.reshape((batch, time, freq, DP_HIDDEN))?)?;
        let intra = (plane + intra)?;

        // Then time, one sequence per band. The result comes back to the
        // frame-major layout before the norm, because the norm's plane is a
        // frequency-by-channel one either way.
        let swapped = intra.permute((0, 2, 1, 3))?.contiguous()?;
        let sequence = swapped.reshape((batch * freq, time, CHANNELS))?;
        let (inter, states) = self
            .inter_rnn
            .forward(&sequence, carried.map(Vec::as_slice))?;
        let inter = self.inter_fc.forward(&inter)?;
        let inter = inter
            .reshape((batch, freq, time, DP_HIDDEN))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let inter = self.inter_norm.forward(&inter)?;

        Ok((
            (intra + inter)?.permute((0, 3, 1, 2))?.contiguous()?,
            states,
        ))
    }
}

/// The whole network.
/// Everything one recording carries from a chunk to the next.
///
/// Small — a few hundred floats — which is why chunking with carried state
/// costs nothing and chunking with an overlap costs a third of the run and is
/// still wrong.
#[derive(Default, Clone)]
pub struct State {
    encoder: Vec<Option<BlockState>>,
    decoder: Vec<Option<BlockState>>,
    dual: Vec<Option<Vec<Tensor>>>,
}

pub struct Gtcrn {
    erb_forward: Tensor,
    erb_inverse: Tensor,
    encoder: Vec<Stage>,
    dual: Vec<DualPath>,
    decoder: Vec<Stage>,
    device: Device,
    dtype: DType,
}

/// A stage of the encoder or decoder: either a plain block or a grouped
/// temporal one.
#[derive(Debug)]
enum Stage {
    Plain(ConvBlock),
    Temporal(GtConvBlock),
}

impl Stage {
    fn forward(
        &self,
        xs: &Tensor,
        carried: Option<&BlockState>,
    ) -> Result<(Tensor, Option<BlockState>)> {
        match self {
            Self::Plain(block) => Ok((block.forward(xs)?, None)),
            Self::Temporal(block) => {
                let (out, state) = block.forward(xs, carried)?;
                Ok((out, Some(state)))
            },
        }
    }
}

impl Gtcrn {
    /// Loads the network from the converted checkpoint.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let erb = vb.pp("erb");
        let encoder_vb = vb.pp("encoder").pp("en_convs");
        let decoder_vb = vb.pp("decoder").pp("de_convs");

        let mut encoder = vec![
            Stage::Plain(ConvBlock::load(
                3 * SFE_KERNEL,
                CHANNELS,
                1,
                false,
                false,
                encoder_vb.pp("0"),
            )?),
            Stage::Plain(ConvBlock::load(
                CHANNELS,
                CHANNELS,
                2,
                false,
                false,
                encoder_vb.pp("1"),
            )?),
        ];
        for (index, dilation) in GT_DILATIONS.iter().enumerate() {
            encoder.push(Stage::Temporal(GtConvBlock::load(
                CHANNELS,
                *dilation,
                false,
                encoder_vb.pp((index + 2).to_string()),
            )?));
        }

        let mut decoder = Vec::with_capacity(5);
        for (index, dilation) in GT_DILATIONS.iter().rev().enumerate() {
            decoder.push(Stage::Temporal(GtConvBlock::load(
                CHANNELS,
                *dilation,
                true,
                decoder_vb.pp(index.to_string()),
            )?));
        }
        decoder.push(Stage::Plain(ConvBlock::load(
            CHANNELS,
            CHANNELS,
            2,
            true,
            false,
            decoder_vb.pp("3"),
        )?));
        decoder.push(Stage::Plain(ConvBlock::load(
            CHANNELS,
            2,
            1,
            true,
            true,
            decoder_vb.pp("4"),
        )?));

        Ok(Self {
            // The band matrix maps 192 bins to 64 bands; stored as the linear
            // layer's `[out, in]`, it is used transposed on the right.
            erb_forward: erb
                .pp("erb_fc")
                .get((ERB_BANDS, BINS - ERB_LOW_BINS), "weight")?
                .t()?
                .contiguous()?,
            erb_inverse: erb
                .pp("ierb_fc")
                .get((BINS - ERB_LOW_BINS, ERB_BANDS), "weight")?
                .t()?
                .contiguous()?,
            encoder,
            dual: (0..super::super::config::DP_BLOCKS)
                .map(|index| {
                    DualPath::load(vb.pp(format!("dpgrnn{}", index + 1)))
                })
                .collect::<Result<Vec<_>>>()?,
            decoder,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// The device the network is on.
    pub fn device(&self) -> &Device { &self.device }

    /// Every stage of one pass, for the parity tests.
    #[cfg(test)]
    pub(crate) fn stages(
        &self,
        real: &[f32],
        imag: &[f32],
        frames: usize,
    ) -> Result<Vec<(&'static str, Tensor)>> {
        let mut out = Vec::new();
        let spec = |values: &[f32]| -> Result<Tensor> {
            Tensor::from_slice(values, (1, BINS, frames), &self.device)?
                .to_dtype(self.dtype)?
                .transpose(1, 2)?
                .contiguous()
        };
        let real_t = spec(real)?;
        let imag_t = spec(imag)?;
        let magnitude = (real_t.sqr()? + imag_t.sqr()?)?
            .affine(1.0, MAG_EPS)?
            .sqrt()?;
        let feat = Tensor::stack(&[&magnitude, &real_t, &imag_t], 1)?;
        out.push(("feat", feat.clone()));
        let banded = self.to_bands(&feat)?;
        out.push(("banded", banded.clone()));
        let mut hidden = spread_subbands(&banded)?;
        out.push(("spread", hidden.clone()));

        let names = ["en0", "en1", "en2", "en3", "en4"];
        let mut skips = Vec::new();
        for (index, stage) in self.encoder.iter().enumerate() {
            hidden = stage.forward(&hidden, None)?.0;
            skips.push(hidden.clone());
            out.push((names[index], hidden.clone()));
        }
        for (index, block) in self.dual.iter().enumerate() {
            hidden = block.forward(&hidden, None)?.0;
            out.push((["dpgrnn1", "dpgrnn2"][index], hidden.clone()));
        }
        let names = ["de0", "de1", "de2", "de3", "de4"];
        for (index, stage) in self.decoder.iter().enumerate() {
            let skip = &skips[self.decoder.len() - 1 - index];
            hidden = stage.forward(&(hidden + skip)?, None)?.0;
            out.push((names[index], hidden.clone()));
        }
        out.push(("mask_banded", hidden.clone()));
        out.push(("mask", self.from_bands(&hidden)?));
        Ok(out)
    }

    /// Turns a spectrum into the complex ratio mask that cleans it.
    ///
    /// `real` and `imag` are `[bins, frames]` in row-major order; the result is
    /// the enhanced spectrum in the same layout. `state` is where the previous
    /// chunk of this recording left off, and is updated in place — a fresh
    /// [`State::default`] starts a new recording.
    pub fn enhance(
        &self,
        real: &[f32],
        imag: &[f32],
        frames: usize,
        state: &mut State,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        state.encoder.resize(self.encoder.len(), None);
        state.decoder.resize(self.decoder.len(), None);
        state.dual.resize(self.dual.len(), None);
        let spec = |values: &[f32]| -> Result<Tensor> {
            Tensor::from_slice(values, (1, BINS, frames), &self.device)?
                .to_dtype(self.dtype)?
                .transpose(1, 2)?
                .contiguous()
        };
        let real_t = spec(real)?;
        let imag_t = spec(imag)?;
        let magnitude = (real_t.sqr()? + imag_t.sqr()?)?
            .affine(1.0, MAG_EPS)?
            .sqrt()?;
        let feat = Tensor::stack(&[&magnitude, &real_t, &imag_t], 1)?;

        let banded = self.to_bands(&feat)?;
        let mut hidden = spread_subbands(&banded)?;

        let mut skips = Vec::with_capacity(self.encoder.len());
        for (index, stage) in self.encoder.iter().enumerate() {
            let (out, carried) =
                stage.forward(&hidden, state.encoder[index].as_ref())?;
            if carried.is_some() {
                state.encoder[index] = carried;
            }
            hidden = out;
            skips.push(hidden.clone());
        }
        for (index, block) in self.dual.iter().enumerate() {
            let (out, carried) =
                block.forward(&hidden, state.dual[index].as_ref())?;
            state.dual[index] = Some(carried);
            hidden = out;
        }
        for (index, stage) in self.decoder.iter().enumerate() {
            let skip = &skips[self.decoder.len() - 1 - index];
            let (out, carried) = stage
                .forward(&(hidden + skip)?, state.decoder[index].as_ref())?;
            if carried.is_some() {
                state.decoder[index] = carried;
            }
            hidden = out;
        }
        let mask = self.from_bands(&hidden)?;

        let mask_real = mask.i((0, 0))?;
        let mask_imag = mask.i((0, 1))?;
        let real_out =
            (real_t.i(0)?.mul(&mask_real)? - imag_t.i(0)?.mul(&mask_imag)?)?;
        let imag_out =
            (imag_t.i(0)?.mul(&mask_real)? + real_t.i(0)?.mul(&mask_imag)?)?;
        // Back to the bin-major layout the transform speaks.
        let out = |xs: Tensor| -> Result<Vec<f32>> {
            xs.transpose(0, 1)?
                .contiguous()?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()
        };
        Ok((out(real_out)?, out(imag_out)?))
    }

    /// Folds the bins above the split into equivalent-rectangular bands.
    fn to_bands(&self, feat: &Tensor) -> Result<Tensor> {
        let low = feat.narrow(3, 0, ERB_LOW_BINS)?;
        let high = feat.narrow(3, ERB_LOW_BINS, BINS - ERB_LOW_BINS)?;
        let (batch, channels, time, _) = high.dims4()?;
        let banded = high
            .reshape((batch * channels * time, BINS - ERB_LOW_BINS))?
            .matmul(&self.erb_forward)?
            .reshape((batch, channels, time, ERB_BANDS))?;
        Tensor::cat(&[low, banded], 3)
    }

    /// Spreads the bands back out over the bins they came from.
    fn from_bands(&self, mask: &Tensor) -> Result<Tensor> {
        let low = mask.narrow(3, 0, ERB_LOW_BINS)?;
        let high = mask.narrow(3, ERB_LOW_BINS, ERB_BANDS)?;
        let (batch, channels, time, _) = high.dims4()?;
        let spread = high
            .contiguous()?
            .reshape((batch * channels * time, ERB_BANDS))?
            .matmul(&self.erb_inverse)?
            .reshape((batch, channels, time, BINS - ERB_LOW_BINS))?;
        Tensor::cat(&[low, spread], 3)
    }
}

/// The width the banded spectrum has, asserted where it is easy to see.
const _: () = assert!(BANDED == ERB_LOW_BINS + ERB_BANDS);
