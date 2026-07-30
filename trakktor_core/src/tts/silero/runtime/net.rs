//! The networks on candle.
//!
//! Two things shape this file, and both come from the same fact: the decoder
//! and the vocoder run over **mel frames**, of which a full-length utterance
//! has thousands, while the encoder runs over symbols, of which it has
//! hundreds.
//!
//! - Everything is laid out **time-major**, `[frames, channels]`, and the batch
//!   axis is gone: the frontend only ever produces one utterance, and carrying
//!   a singleton axis through every reshape buys nothing.
//! - The three convolutions are written as matrix multiplications rather than
//!   handed to a convolution kernel. A same-padded convolution over time is an
//!   `im2col` followed by one large `gemm`, and a depthwise one is `k` shifted
//!   multiply-adds; both keep every thread busy, which a convolution over a
//!   batch of one does not.
//!
//! Ported from Silero TTS (the FastPitch-family acoustic model) and Vocos (the
//! vocoder), both MIT.

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, ops::softmax_last_dim};

use crate::tts::silero::config::{
    ATTENTION_TILE, Config, FFN_KERNEL, HEADS, NORM_EPS, PITCH_KERNEL,
    SHORTEN_FACTOR, VOCODER_KERNEL,
};

/// A linear layer whose weight is stored the way the multiplication wants it.
///
/// torch stores `[out, in]` and every forward pass then multiplies by its
/// transpose. Transposing once at load instead hands the multiplication a
/// contiguous, row-major operand every time — worth doing where nearly all of
/// this engine's time goes, even though the gain measures in the low single
/// digits of a percent.
#[derive(Debug)]
pub struct Dense {
    /// `[in, out]`, contiguous.
    weight: Tensor,
    bias: Tensor,
}

impl Dense {
    /// Loads a layer stored as torch stores it.
    pub fn load(
        in_features: usize,
        out_features: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            weight: vb
                .get((out_features, in_features), "weight")?
                .t()?
                .contiguous()?,
            bias: vb.get(out_features, "bias")?,
        })
    }

    /// Applies it to `[time, in]`, giving `[time, out]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.matmul(&self.weight)?.broadcast_add(&self.bias)
    }
}

/// A same-padded convolution over time, as one matrix multiplication.
///
/// The weights are folded once, at load, into the `[kernel · in, out]` matrix
/// the `im2col` form multiplies by, so nothing is rearranged per call.
#[derive(Debug)]
pub struct TimeConv {
    /// `[kernel · in, out]`.
    weight: Tensor,
    bias: Tensor,
    kernel: usize,
    channels: usize,
}

impl TimeConv {
    /// Loads a convolution stored the way torch stores it, `[out, in, kernel]`.
    pub fn load(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get((out_channels, in_channels, kernel), "weight")?;
        Ok(Self {
            // `[out, in, k]` → `[k, in, out]` → `[k · in, out]`, which is the
            // order the stacked windows below arrive in.
            weight: weight
                .permute((2, 1, 0))?
                .contiguous()?
                .reshape((kernel * in_channels, out_channels))?,
            bias: vb.get(out_channels, "bias")?,
            kernel,
            channels: in_channels,
        })
    }

    /// Applies it to `[time, in]`, giving `[time, out]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (time, _) = xs.dims2()?;
        let padded = pad_time(xs, self.kernel / 2)?;
        let windows: Vec<Tensor> = (0..self.kernel)
            .map(|tap| padded.narrow(0, tap, time))
            .collect::<Result<_>>()?;
        let stacked = if self.kernel == 1 {
            xs.clone()
        } else {
            Tensor::cat(&windows, 1)?
        };
        debug_assert_eq!(stacked.dim(1)?, self.kernel * self.channels);
        stacked.matmul(&self.weight)?.broadcast_add(&self.bias)
    }
}

/// A depthwise convolution over time: every channel has its own `kernel` taps
/// and mixes with no other.
///
/// Written as `kernel` shifted multiply-adds over the whole `[time, channels]`
/// block, which is both the least memory traffic available and free of the
/// transposes a channels-first kernel would need.
#[derive(Debug)]
pub struct DepthwiseConv {
    /// `[kernel, channels]`.
    taps: Tensor,
    bias: Tensor,
    kernel: usize,
}

impl DepthwiseConv {
    /// Loads a depthwise convolution stored as `[channels, 1, kernel]`.
    pub fn load(
        channels: usize,
        kernel: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get((channels, 1, kernel), "weight")?;
        Ok(Self {
            taps: weight.reshape((channels, kernel))?.t()?.contiguous()?,
            bias: vb.get(channels, "bias")?,
            kernel,
        })
    }

    /// Applies it to `[time, channels]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (time, _) = xs.dims2()?;
        let padded = pad_time(xs, self.kernel / 2)?;
        let mut out: Option<Tensor> = None;
        for tap in 0..self.kernel {
            let shifted = padded.narrow(0, tap, time)?;
            let term = shifted.broadcast_mul(&self.taps.narrow(0, tap, 1)?)?;
            out = Some(match out {
                None => term,
                Some(sum) => (sum + term)?,
            });
        }
        out.expect("a convolution has at least one tap")
            .broadcast_add(&self.bias)
    }
}

/// Pads a `[time, channels]` block with `pad` zero frames at each end.
fn pad_time(xs: &Tensor, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(xs.clone());
    }
    let (_, channels) = xs.dims2()?;
    let edge = Tensor::zeros((pad, channels), xs.dtype(), xs.device())?;
    Tensor::cat(&[&edge, xs, &edge], 0)
}

/// The learned positional table: a slice of it, scaled, is added to the input.
#[derive(Debug)]
pub struct Positional {
    /// `[positions, dim]`.
    table: Tensor,
    scale: Tensor,
}

impl Positional {
    pub fn load(positions: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            table: vb.get((positions, dim), "pe")?,
            scale: vb.get(1, "scale")?,
        })
    }

    /// Adds the first `time` positions to `[time, dim]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (time, _) = xs.dims2()?;
        let table = self.table.narrow(0, 0, time)?;
        xs + table.broadcast_mul(&self.scale)?
    }
}

/// Multi-head self-attention over time, in tiles.
#[derive(Debug)]
pub struct Attention {
    in_proj: Linear,
    out_proj: Dense,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            in_proj: Linear::new(
                vb.get((3 * dim, dim), "in_proj_weight")?,
                Some(vb.get(3 * dim, "in_proj_bias")?),
            ),
            out_proj: Dense::load(dim, dim, vb.pp("out_proj"))?,
            heads: HEADS,
            head_dim: dim / HEADS,
        })
    }

    /// Attends `[time, dim]` over itself.
    ///
    /// The queries are walked in tiles so the score matrix is never held whole:
    /// over a long utterance it would be hundreds of megabytes, and it is
    /// consumed a row at a time anyway. Each softmax still runs over a complete
    /// row, so the result is what one large matrix would give.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (time, dim) = xs.dims2()?;
        let projected = self.in_proj.forward(xs)?;
        let split = |offset: usize| -> Result<Tensor> {
            projected
                .narrow(1, offset * dim, dim)?
                .reshape((time, self.heads, self.head_dim))?
                .transpose(0, 1)?
                .contiguous()
        };
        // torch scales the queries before the product rather than the scores
        // after it; the difference is invisible in exact arithmetic and not in
        // this one.
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let queries = (split(0)? * scale)?;
        let keys = split(1)?.transpose(1, 2)?.contiguous()?;
        let values = split(2)?;

        let mut tiles = Vec::with_capacity(time.div_ceil(ATTENTION_TILE));
        for start in (0..time).step_by(ATTENTION_TILE) {
            let rows = ATTENTION_TILE.min(time - start);
            let scores = queries.narrow(1, start, rows)?.matmul(&keys)?;
            tiles.push(softmax_last_dim(&scores)?.matmul(&values)?);
        }
        let joined = if tiles.len() == 1 {
            tiles.remove(0)
        } else {
            Tensor::cat(&tiles, 1)?
        };
        let merged = joined.transpose(0, 1)?.reshape((time, dim))?;
        self.out_proj.forward(&merged)
    }
}

/// One block of the acoustic model: self-attention, then a **convolutional**
/// feed-forward — a wide same-padded convolution and a pointwise projection
/// back, which is what makes this a FastSpeech block rather than a plain
/// transformer one.
#[derive(Debug)]
pub struct FftBlock {
    attention: Attention,
    conv: TimeConv,
    project: Dense,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl FftBlock {
    pub fn load(dim: usize, inner: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(dim, vb.pp("self_attn"))?,
            conv: TimeConv::load(dim, inner, FFN_KERNEL, vb.pp("conv1"))?,
            // The second convolution has a kernel of one, which is a matrix
            // multiplication written the long way; the conversion stores it as
            // one, and the transposes it would have needed disappear with it.
            project: Dense::load(inner, dim, vb.pp("conv2"))?,
            norm1: norm(dim, vb.pp("norm1"))?,
            norm2: norm(dim, vb.pp("norm2"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let attended =
            self.norm1.forward(&(xs + self.attention.forward(xs)?)?)?;
        let hidden = self.conv.forward(&attended)?.relu()?;
        let hidden = self.project.forward(&hidden)?;
        self.norm2.forward(&(attended + hidden)?)
    }
}

/// A stack of [`FftBlock`]s over a positional table — the encoder, and the body
/// of both predictors.
#[derive(Debug)]
pub struct ForwardTransformer {
    positional: Positional,
    layers: Vec<FftBlock>,
    norm: LayerNorm,
}

impl ForwardTransformer {
    pub fn load(
        positions: usize,
        dim: usize,
        inner: usize,
        layers: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            positional: Positional::load(positions, dim, vb.pp("pos_encoder"))?,
            layers: (0..layers)
                .map(|index| {
                    FftBlock::load(
                        dim,
                        inner,
                        vb.pp("layers").pp(index.to_string()),
                    )
                })
                .collect::<Result<_>>()?,
            norm: norm(dim, vb.pp("norm"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut hidden = self.positional.forward(xs)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        self.norm.forward(&hidden)
    }
}

/// A head that predicts one number per symbol: duration, or pitch.
#[derive(Debug)]
pub struct SeriesPredictor {
    embedding: Tensor,
    speakers: Tensor,
    /// The intonation table, on the one model that has one.
    types: Option<Tensor>,
    transformer: ForwardTransformer,
    out: Dense,
}

impl SeriesPredictor {
    pub fn load(
        config: &Config,
        with_types: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dim = config.predictor_dim;
        Ok(Self {
            embedding: vb.get((config.symbols, dim), "embedding.weight")?,
            speakers: vb
                .get((config.speaker_slots, dim), "speaker_embedding.weight")?,
            types: with_types
                .then(|| {
                    vb.get(
                        (config.utterance_types, dim),
                        "type_embedding.weight",
                    )
                })
                .transpose()?,
            transformer: ForwardTransformer::load(
                config.positions,
                dim,
                config.predictor_ff_inner,
                config.predictor_layers,
                vb.pp("transformer"),
            )?,
            out: Dense::load(dim, 1, vb.pp("lin"))?,
        })
    }

    /// Predicts one value per symbol.
    pub fn forward(
        &self,
        ids: &Tensor,
        speaker: usize,
        types: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden = self.embedding.index_select(ids, 0)?;
        hidden =
            hidden.broadcast_add(&speaker_row(&self.speakers, speaker)?)?;
        if let (Some(table), Some(types)) = (&self.types, types) {
            hidden = (hidden + table.index_select(types, 0)?)?;
        }
        let hidden = self.transformer.forward(&hidden)?;
        self.out.forward(&hidden)?.squeeze(D::Minus1)
    }
}

/// The decoder: one block at full resolution, two on a sequence shortened
/// threefold, one more at full resolution again.
///
/// The first of the two shortened blocks is **not** run. Its output is
/// discarded upstream — both are applied to the same input there, and only the
/// second is kept — so computing it would cost a third of the decoder for a
/// result that is thrown away. Its weights are not even loaded.
#[derive(Debug)]
pub struct HourGlass {
    positional: Positional,
    pre: FftBlock,
    shortened: FftBlock,
    post: FftBlock,
    upsample: Dense,
    norm: LayerNorm,
}

impl HourGlass {
    pub fn load(config: &Config, vb: VarBuilder) -> Result<Self> {
        let dim = config.dim;
        let inner = config.ff_inner;
        Ok(Self {
            positional: Positional::load(
                config.positions,
                dim,
                vb.pp("pos_encoder"),
            )?,
            pre: FftBlock::load(
                dim,
                inner,
                vb.pp("pre_vanilla_layers").pp("0"),
            )?,
            shortened: FftBlock::load(
                dim,
                inner,
                vb.pp("shorten_layers").pp("1"),
            )?,
            post: FftBlock::load(
                dim,
                inner,
                vb.pp("post_vanilla_layers").pp("0"),
            )?,
            upsample: Dense::load(
                dim,
                dim * SHORTEN_FACTOR,
                vb.pp("upsample").pp("proj"),
            )?,
            norm: norm(dim, vb.pp("norm"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (time, dim) = xs.dims2()?;
        let hidden = self.pre.forward(&self.positional.forward(xs)?)?;
        // The sequence is padded out to a multiple of the shortening factor,
        // and the padding is cut off again after the residual.
        let padded_len = time.div_ceil(SHORTEN_FACTOR) * SHORTEN_FACTOR;
        let residual = if padded_len == time {
            hidden
        } else {
            let tail = Tensor::zeros(
                (padded_len - time, dim),
                xs.dtype(),
                xs.device(),
            )?;
            Tensor::cat(&[&hidden, &tail], 0)?
        };
        // Average pooling with kernel and stride both equal to the factor is a
        // regrouping and a mean, which is what it is written as.
        let pooled = residual
            .reshape((padded_len / SHORTEN_FACTOR, SHORTEN_FACTOR, dim))?
            .mean(1)?;
        let shortened = self.shortened.forward(&pooled)?;
        let upsampled = self
            .upsample
            .forward(&shortened)?
            .reshape((padded_len, dim))?;
        let joined = (upsampled + residual)?.narrow(0, 0, time)?;
        self.norm.forward(&self.post.forward(&joined)?)
    }
}

/// The acoustic model: symbols in, a mel spectrogram out.
#[derive(Debug)]
pub struct Acoustic {
    embedding: Tensor,
    speakers: Tensor,
    encoder: ForwardTransformer,
    pitch_proj: TimeConv,
    decoder: HourGlass,
    out: Dense,
}

impl Acoustic {
    pub fn load(config: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            embedding: vb
                .get((config.symbols, config.dim), "embedding.weight")?,
            speakers: vb.get(
                (config.speaker_slots, config.dim),
                "speaker_embedding.weight",
            )?,
            encoder: ForwardTransformer::load(
                config.positions,
                config.dim,
                config.ff_inner,
                config.encoder_layers,
                vb.pp("encoder"),
            )?,
            pitch_proj: TimeConv::load(
                1,
                config.dim,
                PITCH_KERNEL,
                vb.pp("pitch_proj"),
            )?,
            decoder: HourGlass::load(config, vb.pp("decoder"))?,
            out: Dense::load(config.dim, config.mel_channels, vb.pp("lin"))?,
        })
    }

    /// Encodes the symbols, expands them to frames, and decodes a mel.
    ///
    /// `expansion` is one symbol index per frame — the length regulator,
    /// prepared on the host because neither backend has `repeat_interleave`.
    pub fn forward(
        &self,
        ids: &Tensor,
        speaker: usize,
        pitch: &Tensor,
        expansion: &Tensor,
    ) -> Result<Tensor> {
        let embedded = self.embedding.index_select(ids, 0)?;
        let encoded = self.encoder.forward(&embedded)?;
        let encoded =
            encoded.broadcast_add(&speaker_row(&self.speakers, speaker)?)?;
        let projected = self.pitch_proj.forward(&pitch.unsqueeze(1)?)?;
        let encoded = (encoded + projected)?;
        let expanded = encoded.index_select(expansion, 0)?;
        let decoded = self.decoder.forward(&expanded)?;
        self.out.forward(&decoded)
    }
}

/// One ConvNeXt block of the vocoder.
#[derive(Debug)]
pub struct ConvNeXt {
    dwconv: DepthwiseConv,
    norm: LayerNorm,
    pwconv1: Dense,
    pwconv2: Dense,
    gamma: Tensor,
}

impl ConvNeXt {
    pub fn load(dim: usize, inner: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            dwconv: DepthwiseConv::load(dim, VOCODER_KERNEL, vb.pp("dwconv"))?,
            norm: norm(dim, vb.pp("norm"))?,
            pwconv1: Dense::load(dim, inner, vb.pp("pwconv1"))?,
            pwconv2: Dense::load(inner, dim, vb.pp("pwconv2"))?,
            gamma: vb.get(dim, "gamma")?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = self.norm.forward(&self.dwconv.forward(xs)?)?;
        let hidden = self.pwconv1.forward(&hidden)?.gelu_erf()?;
        let hidden = self.pwconv2.forward(&hidden)?;
        xs + hidden.broadcast_mul(&self.gamma)?
    }
}

/// The vocoder: a mel spectrogram in, a complex spectrum out.
#[derive(Debug)]
pub struct Vocoder {
    embed: TimeConv,
    norm: LayerNorm,
    blocks: Vec<ConvNeXt>,
    final_norm: LayerNorm,
    out: Dense,
}

impl Vocoder {
    pub fn load(config: &Config, vb: VarBuilder) -> Result<Self> {
        let backbone = vb.pp("backbone");
        Ok(Self {
            embed: TimeConv::load(
                config.mel_channels,
                config.vocoder_dim,
                VOCODER_KERNEL,
                backbone.pp("embed"),
            )?,
            norm: norm(config.vocoder_dim, backbone.pp("norm"))?,
            blocks: (0..config.vocoder_layers)
                .map(|index| {
                    ConvNeXt::load(
                        config.vocoder_dim,
                        config.vocoder_ff_inner,
                        backbone.pp("convnext").pp(index.to_string()),
                    )
                })
                .collect::<Result<_>>()?,
            final_norm: norm(
                config.vocoder_dim,
                backbone.pp("final_layer_norm"),
            )?,
            out: Dense::load(
                config.vocoder_dim,
                config.n_fft + 2,
                vb.pp("head").pp("out"),
            )?,
        })
    }

    /// Runs the backbone and the head over `[frames, mel]`, returning
    /// `[frames, n_fft + 2]` — a log magnitude and a phase per bin.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let mut hidden = self.norm.forward(&self.embed.forward(mel)?)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        self.out.forward(&self.final_norm.forward(&hidden)?)
    }
}

/// One speaker's embedding row, shaped to broadcast over time.
fn speaker_row(table: &Tensor, speaker: usize) -> Result<Tensor> {
    table.narrow(0, speaker, 1)
}

/// A normalization with the model's epsilon.
fn norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        vb.get(size, "weight")?,
        vb.get(size, "bias")?,
        NORM_EPS,
    ))
}

/// Builds a `[len]` index tensor for `index_select`.
pub fn indices(values: &[u32], device: &Device) -> Result<Tensor> {
    Tensor::from_slice(values, values.len(), device)
}

/// Builds a `[len]` f32 tensor.
pub fn values(values: &[f32], device: &Device) -> Result<Tensor> {
    Tensor::from_slice(values, values.len(), device)?.to_dtype(DType::F32)
}
