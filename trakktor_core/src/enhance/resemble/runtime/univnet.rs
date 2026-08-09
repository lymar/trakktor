//! The waveform generator on candle: noise in, speech out, conditioned on what
//! the decoder produced.
//!
//! # It starts from noise, and that is not a detail
//!
//! There is no waveform on this path at all. The vocoder is handed a hundred
//! and twenty-eight channels of Gaussian noise and four upsampling blocks turn
//! it into sound, steered at every step by the conditioning. That is why this
//! engine can put back a band the recording never had — and why two runs of it
//! differ unless the noise is fixed ([`noise`](super::super::noise)).
//!
//! # Location-variable convolutions
//!
//! The blocks do not hold their convolution kernels. A **kernel predictor**
//! reads the conditioning and emits a fresh kernel for every frame of it: two
//! hundred and twenty-one thousand numbers per frame, which is where three
//! quarters of this checkpoint's weight sits. Each output sample is then
//! convolved with the kernel belonging to the conditioning frame it falls
//! under.
//!
//! Written out, that is one small matrix multiply per conditioning frame, and
//! there are as many of those as there are mel frames — three thousand on a
//! thirty-second chunk. Issuing three thousand kernel launches four times per
//! block would spend all the time in the driver, so the frames are batched: the
//! shifted taps are gathered into one `[frames, hop, taps · channels]` tensor
//! and multiplied against the predicted kernels in a single batched matmul. The
//! gather is what costs memory, so it is done a slice of frames at a time.
//!
//! # The anti-aliased activation
//!
//! Each block's activation is not applied where it is. The signal is upsampled
//! two-to-one through a Kaiser-windowed sinc, the activation is applied there,
//! and it is filtered back down — which is what keeps a periodic nonlinearity
//! from folding harmonics back into the band. Both filters are the **same
//! twelve numbers for every channel**, so they are read out of the checkpoint
//! as twelve scalars and the filtering is written as twelve shifted multiplies.
//! A grouped convolution would be the literal translation and, in candle, one
//! kernel launch per channel.
//!
//! Ported from resemble-enhance (MIT), whose vocoder follows UnivNet and
//! LVCNet, with the anti-aliased activation from BigVGAN.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module,
    VarBuilder,
};

use super::super::config::{
    AMP_DILATIONS, KPNET_HIDDEN, KPNET_KERNEL, LEAKY_SLOPE, LVC_DILATIONS,
    NOISE_CHANNELS, RESAMPLE_RATIO, RESAMPLE_TAPS, SNAKE_CLAMP, STRIDES,
    UNIVNET_CHANNELS, VOCODER_HALO, VOCODER_INPUT, VOCODER_PAD, vocoder_slice,
};

/// Every stage of one vocoder run, and the waveform it ended in.
pub type Staged = (Vec<(String, Tensor)>, Vec<f32>);

/// Largest gathered location-variable batch, in elements.
///
/// The gather is `frames × hop × taps × channels`, and on the last block that
/// is one and a half gigabytes if it is built in one go. The frames are cut
/// into slices under this budget; the arithmetic does not change, because no
/// output position reads across a conditioning frame's kernel.
const LVC_BUDGET: usize = 16 << 20;

/// A rectifier with a small slope below zero.
fn leaky(xs: &Tensor, slope: f64) -> Result<Tensor> {
    let positive = xs.relu()?;
    let negative = (xs - &positive)?;
    positive + (negative * slope)?
}

/// Reverses a `[batch, channels, length]` tensor along its length.
fn reverse(xs: &Tensor) -> Result<Tensor> {
    let len = xs.dim(2)?;
    let index = Tensor::from_vec(
        (0..len).rev().map(|i| i as u32).collect::<Vec<_>>(),
        len,
        xs.device(),
    )?;
    // The callers hand in slices of a larger tensor, and a gather needs its
    // source laid out.
    xs.contiguous()?.index_select(&index, 2)
}

/// Mirrors `pad` samples in at each end, the way a convolution with
/// `padding_mode="reflect"` does — the edge sample itself is not repeated.
fn reflect_pad(xs: &Tensor, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(xs.clone());
    }
    let len = xs.dim(2)?;
    let left = reverse(&xs.narrow(2, 1, pad)?)?;
    let right = reverse(&xs.narrow(2, len - 1 - pad, pad)?)?;
    Tensor::cat(&[&left, xs, &right], 2)
}

/// Repeats the edge sample outwards, the way `pad(..., mode="replicate")` does.
fn replicate_pad(xs: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let len = xs.dim(2)?;
    let mut parts = Vec::with_capacity(3);
    let head;
    let tail;
    if left > 0 {
        head = xs.narrow(2, 0, 1)?.repeat((1, 1, left))?;
        parts.push(&head);
    }
    parts.push(xs);
    if right > 0 {
        tail = xs.narrow(2, len - 1, 1)?.repeat((1, 1, right))?;
        parts.push(&tail);
    }
    Tensor::cat(&parts, 2)
}

/// The Snake activation, with a learned frequency and a learned magnitude per
/// channel: `x + sin²(αx) / β`.
#[derive(Debug)]
struct Snake {
    alpha: Tensor,
    beta: Tensor,
}

impl Snake {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        let read = |name: &str| -> Result<Tensor> {
            vb.get(channels, name)?
                .exp()?
                .clamp(SNAKE_CLAMP.0, SNAKE_CLAMP.1)?
                .reshape((1, channels, 1))
        };
        Ok(Self {
            alpha: read("log_alpha")?,
            beta: read("log_beta")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let wave = xs.broadcast_mul(&self.alpha)?.sin()?.sqr()?;
        xs + wave.broadcast_div(&self.beta)?
    }
}

/// A half-band filter, held as its twelve taps rather than as a kernel.
#[derive(Debug)]
struct HalfBand {
    taps: Vec<f64>,
}

impl HalfBand {
    /// Reads the filter the checkpoint carries. It is a buffer rather than a
    /// parameter — the same Kaiser-windowed sinc for every channel — so twelve
    /// numbers is all of it.
    fn load(vb: VarBuilder, name: &str) -> Result<Self> {
        let taps = vb
            .get((1, 1, RESAMPLE_TAPS), name)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(Self {
            taps: taps.into_iter().map(f64::from).collect(),
        })
    }

    /// Two-to-one interpolation: the reference's transposed convolution,
    /// written as its two polyphase branches so that no grouped convolution is
    /// needed.
    fn up(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, channels, len) = xs.dims3()?;
        let half = RESAMPLE_TAPS / RESAMPLE_RATIO;
        let margin = half - 1;
        // The reference pads by `kernel/ratio − 1` before the transposed
        // convolution and trims the transient afterwards.
        let padded = replicate_pad(xs, margin, margin)?;
        let span = len + 2 * margin;
        // A transposed convolution of stride two and `2h` taps lengthens its
        // input by `h − 1` at each end, and each of the two polyphase branches
        // produces one output per input plus that margin.
        let reach = half - 1;
        let zeroed = padded.pad_with_zeros(2, reach, reach)?;
        let branch = |parity: usize| -> Result<Tensor> {
            let mut sum: Option<Tensor> = None;
            for step in 0..half {
                let weight = self.taps[RESAMPLE_RATIO * step + parity];
                let slice = zeroed.narrow(2, reach - step, span + reach)?;
                let term = (slice * weight)?;
                sum = Some(match sum {
                    Some(total) => (total + term)?,
                    None => term,
                });
            }
            sum.ok_or_else(|| candle_core::Error::Msg("an empty filter".into()))
        };
        let woven = Tensor::stack(&[branch(0)?, branch(1)?], 3)?.reshape((
            batch,
            channels,
            RESAMPLE_RATIO * (span + reach),
        ))?;
        // The transposed convolution is scaled by the ratio, and the two
        // transients are the same width at each end.
        let trim =
            margin * RESAMPLE_RATIO + (RESAMPLE_TAPS - RESAMPLE_RATIO) / 2;
        woven.narrow(2, trim, len * RESAMPLE_RATIO)? * RESAMPLE_RATIO as f64
    }

    /// One-to-two decimation through the same filter, again by polyphase, so
    /// that only the samples that survive are computed.
    fn down(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, channels, len) = xs.dims3()?;
        let half = RESAMPLE_TAPS / RESAMPLE_RATIO;
        // The reference's low pass pads asymmetrically: one short of half the
        // kernel on the left, half of it on the right.
        let padded = replicate_pad(xs, half - 1, half)?;
        // One more zero makes the length even so that it splits into two
        // phases; nothing reads it.
        let padded = padded.pad_with_zeros(2, 0, 1)?;
        let span = padded.dim(2)?;
        let phases = padded.reshape((batch, channels, span / 2, 2))?;
        let out = len / RESAMPLE_RATIO;
        let mut sum: Option<Tensor> = None;
        for step in 0..half {
            for parity in 0..RESAMPLE_RATIO {
                let weight = self.taps[RESAMPLE_RATIO * step + parity];
                let phase = phases
                    .narrow(3, parity, 1)?
                    .squeeze(3)?
                    .narrow(2, step, out)?;
                let term = (phase * weight)?;
                sum = Some(match sum {
                    Some(total) => (total + term)?,
                    None => term,
                });
            }
        }
        sum.ok_or_else(|| candle_core::Error::Msg("an empty filter".into()))
    }
}

/// One anti-aliased layer: a dilated convolution, the activation applied at
/// twice the rate, and a convolution back.
#[derive(Debug)]
struct AmpLayer {
    first: Conv1d,
    snake: Snake,
    filter: HalfBand,
    down_filter: HalfBand,
    second: Conv1d,
}

impl AmpLayer {
    fn load(channels: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            first: same_conv(channels, channels, 3, dilation, vb.pp("0"))?,
            snake: Snake::load(channels, vb.pp("1.act"))?,
            filter: HalfBand::load(vb.pp("1.upsample"), "filter")?,
            down_filter: HalfBand::load(
                vb.pp("1.downsample.lowpass"),
                "filter",
            )?,
            second: same_conv(channels, channels, 3, 1, vb.pp("2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = self.first.forward(xs)?;
        let hidden = self.filter.up(&hidden)?;
        let hidden = self.snake.forward(&hidden)?;
        let hidden = self.down_filter.down(&hidden)?;
        self.second.forward(&hidden)
    }
}

/// Three of those, added back onto the input.
#[derive(Debug)]
struct AmpBlock {
    layers: Vec<AmpLayer>,
}

impl AmpBlock {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(AMP_DILATIONS.len());
        for (index, &dilation) in AMP_DILATIONS.iter().enumerate() {
            layers.push(AmpLayer::load(
                channels,
                dilation,
                vb.pp(format!("{index}")),
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut hidden = xs.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        xs + hidden
    }
}

/// A convolution whose padding keeps the length.
fn same_conv(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    dilation: usize,
    vb: VarBuilder,
) -> Result<Conv1d> {
    candle_nn::conv1d(
        in_channels,
        out_channels,
        kernel,
        Conv1dConfig {
            padding: dilation * (kernel / 2),
            dilation,
            ..Default::default()
        },
        vb,
    )
}

/// The network that reads the conditioning and emits a convolution kernel for
/// every frame of it.
///
/// **The kernels are never produced whole**, and that is what makes this engine
/// usable on a long recording. Upstream predicts all four layers' kernels with
/// one convolution of two hundred and twenty-one thousand output channels; over
/// a thirty-second chunk that single tensor is 2.8 GB, and the copies the
/// matmul wants are another 2.8 GB on top. So the convolution is **split into
/// its four layers at load time** — the layers are its slowest axis, so a slice
/// of its output channels is exactly one layer — and each layer's kernels are
/// produced a slice of frames at a time, inside the loop that consumes them.
/// The arithmetic is unchanged: a width-three convolution restricted to a range
/// of frames is the same numbers, given one frame of context at each end.
#[derive(Debug)]
struct KernelPredictor {
    input: Conv1d,
    residual: Vec<(Conv1d, Conv1d)>,
    kernels: Vec<Conv1d>,
    biases: Conv1d,
}

impl KernelPredictor {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        let taps = KPNET_KERNEL;
        let layers = LVC_DILATIONS.len();
        let per_layer = channels * 2 * channels * KPNET_KERNEL;
        let bias_channels = 2 * channels * layers;
        let mut residual = Vec::with_capacity(3);
        for index in 0..3 {
            let vb = vb.pp(format!("residual_convs.{index}"));
            residual.push((
                same_conv(KPNET_HIDDEN, KPNET_HIDDEN, taps, 1, vb.pp("1"))?,
                same_conv(KPNET_HIDDEN, KPNET_HIDDEN, taps, 1, vb.pp("3"))?,
            ));
        }
        // The one convolution, read once and cut into four. Its padding is
        // taken over by the caller, which pads the frames it hands in.
        let vb_kernels = vb.pp("kernel_conv");
        let weight = vb_kernels
            .get((per_layer * layers, KPNET_HIDDEN, KPNET_KERNEL), "weight")?;
        let bias = vb_kernels.get(per_layer * layers, "bias")?;
        let mut kernels = Vec::with_capacity(layers);
        for index in 0..layers {
            kernels.push(Conv1d::new(
                weight
                    .narrow(0, index * per_layer, per_layer)?
                    .contiguous()?,
                Some(
                    bias.narrow(0, index * per_layer, per_layer)?
                        .contiguous()?,
                ),
                Conv1dConfig::default(),
            ));
        }
        Ok(Self {
            input: same_conv(
                VOCODER_INPUT,
                KPNET_HIDDEN,
                5,
                1,
                vb.pp("input_conv.0"),
            )?,
            residual,
            kernels,
            biases: same_conv(
                KPNET_HIDDEN,
                bias_channels,
                taps,
                1,
                vb.pp("bias_conv"),
            )?,
        })
    }

    /// Reads the conditioning into the representation the kernels are made
    /// from, padded by the one frame of context the width-three convolution
    /// needs at each end, and the biases, which are small enough to keep whole.
    fn forward(
        &self,
        cond: &Tensor,
        channels: usize,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let frames = cond.dim(2)?;
        let mut hidden = leaky(&self.input.forward(cond)?, LEAKY_SLOPE)?;
        for (first, second) in &self.residual {
            let branch = leaky(&first.forward(&hidden)?, LEAKY_SLOPE)?;
            let branch = leaky(&second.forward(&branch)?, LEAKY_SLOPE)?;
            hidden = (hidden + branch)?;
        }
        let layers = LVC_DILATIONS.len();
        let out_channels = 2 * channels;
        let biases = self.biases.forward(&hidden)?.reshape((
            layers,
            out_channels,
            frames,
        ))?;
        let per_layer = (0..layers)
            .map(|index| {
                biases
                    .narrow(0, index, 1)?
                    .squeeze(0)?
                    .t()?
                    .contiguous()?
                    .reshape((frames, 1, out_channels))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((hidden.pad_with_zeros(2, 1, 1)?, per_layer))
    }

    /// One layer's kernels for a range of frames, as `[frames, taps · in,
    /// out]`.
    fn slice(
        &self,
        layer: usize,
        hidden: &Tensor,
        start: usize,
        count: usize,
        channels: usize,
    ) -> Result<Tensor> {
        let out_channels = 2 * channels;
        let raw = self.kernels[layer]
            .forward(&hidden.narrow(2, start, count + 2)?.contiguous()?)?;
        raw.reshape((channels, out_channels, KPNET_KERNEL, count))?
            .permute((3, 2, 0, 1))?
            .contiguous()?
            .reshape((count, KPNET_KERNEL * channels, out_channels))
    }
}

/// One upsampling block.
#[derive(Debug)]
struct LvcBlock {
    up: ConvTranspose1d,
    amp: AmpBlock,
    predictor: KernelPredictor,
    convs: Vec<Conv1d>,
    hop: usize,
    channels: usize,
}

impl LvcBlock {
    fn load(
        channels: usize,
        stride: usize,
        hop: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let up = candle_nn::conv_transpose1d(
            channels,
            channels,
            2 * stride,
            ConvTranspose1dConfig {
                stride,
                padding: stride / 2 + stride % 2,
                output_padding: stride % 2,
                ..Default::default()
            },
            vb.pp("convt_pre.1"),
        )?;
        let mut convs = Vec::with_capacity(LVC_DILATIONS.len());
        for (index, &dilation) in LVC_DILATIONS.iter().enumerate() {
            convs.push(same_conv(
                channels,
                channels,
                3,
                dilation,
                vb.pp(format!("conv_blocks.{index}.1")),
            )?);
        }
        Ok(Self {
            up,
            amp: AmpBlock::load(channels, vb.pp("amp_block"))?,
            predictor: KernelPredictor::load(
                channels,
                vb.pp("kernel_predictor"),
            )?,
            convs,
            hop,
            channels,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cond: &Tensor,
        label: usize,
        mut stages: Option<&mut Vec<(String, Tensor)>>,
    ) -> Result<Tensor> {
        let record =
            |name: &str,
             tensor: &Tensor,
             stages: &mut Option<&mut Vec<(String, Tensor)>>| {
                if let Some(stages) = stages.as_deref_mut() {
                    stages.push((format!("voc{label}_{name}"), tensor.clone()));
                }
            };
        let mut hidden = self.up.forward(&leaky(xs, LEAKY_SLOPE)?)?;
        record("convt", &hidden, &mut stages);
        hidden = self.amp.forward(&hidden)?;
        record("amp", &hidden, &mut stages);
        let (conditioning, biases) =
            self.predictor.forward(cond, self.channels)?;
        for (index, (conv, bias)) in self.convs.iter().zip(&biases).enumerate()
        {
            let branch = leaky(
                &conv.forward(&leaky(&hidden, LEAKY_SLOPE)?)?,
                LEAKY_SLOPE,
            )?;
            let out =
                self.variable_conv(&branch, index, &conditioning, bias)?;
            if index == 0 {
                record("lvc", &out, &mut stages);
            }
            let gate =
                candle_nn::ops::sigmoid(&out.narrow(1, 0, self.channels)?)?;
            let value = out.narrow(1, self.channels, self.channels)?.tanh()?;
            hidden = (hidden + (gate * value)?)?;
        }
        Ok(hidden)
    }

    /// The convolution whose kernel changes with the conditioning frame.
    ///
    /// The input is `frames × hop` samples long, and the sample at
    /// `frame · hop + s` is convolved with that frame's kernel. Both the
    /// kernels and the gathered taps are produced a slice of frames at a
    /// time, under one budget: whole, on the last block of a thirty-second
    /// chunk, they would be several gigabytes apiece.
    fn variable_conv(
        &self,
        xs: &Tensor,
        layer: usize,
        conditioning: &Tensor,
        bias: &Tensor,
    ) -> Result<Tensor> {
        let (_, channels, len) = xs.dims3()?;
        let frames = len / self.hop;
        let out_channels = 2 * channels;
        let taps = KPNET_KERNEL;
        let padded = xs.pad_with_zeros(2, taps / 2, taps / 2)?;

        // Both halves of the slice cost, per frame: the taps gathered out of
        // the signal, and the kernel predicted for it.
        let per_frame =
            self.hop * taps * channels + taps * channels * out_channels;
        let slice = (LVC_BUDGET / per_frame.max(1)).clamp(1, frames);
        let mut pieces = Vec::new();
        let mut start = 0;
        while start < frames {
            let count = slice.min(frames - start);
            let span = count * self.hop;
            let mut gathered = Vec::with_capacity(taps);
            for tap in 0..taps {
                gathered.push(
                    padded
                        .narrow(2, start * self.hop + tap, span)?
                        .reshape((channels, count, self.hop))?
                        .permute((1, 2, 0))?,
                );
            }
            let features = Tensor::stack(&gathered, 2)?
                .contiguous()?
                .reshape((count, self.hop, taps * channels))?;
            let kernel = self.predictor.slice(
                layer,
                conditioning,
                start,
                count,
                channels,
            )?;
            let piece = features
                .matmul(&kernel)?
                .broadcast_add(&bias.narrow(0, start, count)?)?;
            pieces.push(
                piece
                    .permute((2, 0, 1))?
                    .contiguous()?
                    .reshape((out_channels, span))?,
            );
            start += count;
        }
        Tensor::cat(&pieces, 1)?.reshape((1, out_channels, len))
    }
}

/// The vocoder.
#[derive(Debug)]
pub struct UnivNet {
    pre: Conv1d,
    blocks: Vec<LvcBlock>,
    post: Conv1d,
    device: Device,
    dtype: DType,
}

impl UnivNet {
    /// Loads it from a checkpoint rooted at `vb`.
    ///
    /// # Errors
    ///
    /// Returns whatever candle reports for a missing or misshapen tensor.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::with_capacity(STRIDES.len());
        let mut hop = 1;
        for (index, &stride) in STRIDES.iter().enumerate() {
            hop *= stride;
            blocks.push(LvcBlock::load(
                UNIVNET_CHANNELS,
                stride,
                hop,
                vb.pp(format!("blocks.{index}")),
            )?);
        }
        Ok(Self {
            pre: candle_nn::conv1d(
                NOISE_CHANNELS,
                UNIVNET_CHANNELS,
                7,
                Conv1dConfig::default(),
                vb.pp("conv_pre"),
            )?,
            blocks,
            post: candle_nn::conv1d(
                UNIVNET_CHANNELS,
                1,
                7,
                Conv1dConfig::default(),
                vb.pp("conv_post.1"),
            )?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// How many noise values one run over `frames` of conditioning needs.
    #[must_use]
    pub fn noise_len(frames: usize) -> usize {
        NOISE_CHANNELS * (frames + VOCODER_PAD)
    }

    /// Turns conditioning into a waveform.
    ///
    /// `cond` is `[1, VOCODER_INPUT, frames]` and `noise` is
    /// [`noise_len`](Self::noise_len) standard normal values, channel-major.
    /// The result is `frames × HOP` samples.
    ///
    /// # Errors
    ///
    /// Returns whatever candle reports for a failed operation.
    pub fn forward(&self, cond: &Tensor, noise: &[f32]) -> Result<Vec<f32>> {
        self.run(cond, noise, None)
    }

    /// The same, collecting every stage on the way, which is what the parity
    /// tests compare against the reference.
    ///
    /// # Errors
    ///
    /// Returns whatever candle reports for a failed operation.
    pub fn stages(&self, cond: &Tensor, noise: &[f32]) -> Result<Staged> {
        let mut stages = Vec::new();
        let wave = self.run(cond, noise, Some(&mut stages))?;
        Ok((stages, wave))
    }

    /// Runs the whole vocoder, a slice of conditioning frames at a time.
    ///
    /// Every operation in it is local — convolutions of a few taps, an
    /// upsampling with a fixed ratio, a convolution whose kernel belongs to the
    /// frame it sits under — so a slice with
    /// [`VOCODER_HALO`] frames of context on each side produces, in its
    /// interior, what a single pass would. What that buys is a peak that does
    /// not grow with the recording: in one pass a thirty-second chunk is half a
    /// gigabyte per activation and several gigabytes live at once.
    ///
    /// Collecting stages forces a single slice, because a stage is only
    /// meaningful whole. The sliced path is what everything else uses, and it
    /// is what the parity test on the vocoder's own waveform measures.
    fn run(
        &self,
        cond: &Tensor,
        noise: &[f32],
        mut stages: Option<&mut Vec<(String, Tensor)>>,
    ) -> Result<Vec<f32>> {
        let frames = cond.dim(2)?;
        let padded_frames = frames + VOCODER_PAD;
        let cond = cond.pad_with_zeros(2, 0, VOCODER_PAD)?;
        let source = Tensor::from_slice(
            noise,
            (1, NOISE_CHANNELS, padded_frames),
            &self.device,
        )?
        .to_dtype(self.dtype)?;
        let hop: usize = STRIDES.iter().product();

        let slice = if stages.is_some() {
            padded_frames
        } else {
            vocoder_slice()
        };
        let mut out: Vec<f32> = Vec::with_capacity(frames * hop);
        let mut start = 0;
        while start < padded_frames {
            let count = slice.min(padded_frames - start);
            let low = start.saturating_sub(VOCODER_HALO);
            let high = (start + count + VOCODER_HALO).min(padded_frames);
            let wave = self.blocks_over(
                &cond.narrow(2, low, high - low)?,
                &source.narrow(2, low, high - low)?,
                stages.as_deref_mut(),
            )?;
            let begin = (start - low) * hop;
            out.extend_from_slice(&wave[begin..begin + count * hop]);
            start += count;
        }
        out.truncate(frames * hop);
        Ok(out)
    }

    /// One slice through the block stack, as samples.
    fn blocks_over(
        &self,
        cond: &Tensor,
        noise: &Tensor,
        mut stages: Option<&mut Vec<(String, Tensor)>>,
    ) -> Result<Vec<f32>> {
        let mut hidden = self.pre.forward(&reflect_pad(noise, 3)?)?;
        if let Some(stages) = stages.as_deref_mut() {
            stages.push(("voc_pre".to_owned(), hidden.clone()));
        }
        for (index, block) in self.blocks.iter().enumerate() {
            hidden =
                block.forward(&hidden, cond, index, stages.as_deref_mut())?;
            if let Some(stages) = stages.as_deref_mut() {
                stages.push((format!("voc_block{index}"), hidden.clone()));
            }
        }
        hidden = leaky(&hidden, LEAKY_SLOPE)?;
        hidden = self.post.forward(&reflect_pad(&hidden, 3)?)?.tanh()?;
        if let Some(stages) = stages {
            stages.push(("voc_post".to_owned(), hidden.clone()));
        }
        hidden.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
    }
}
