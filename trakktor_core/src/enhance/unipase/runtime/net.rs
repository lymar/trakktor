//! The networks, on candle: the encoder, the adapter and the vocoder.
//!
//! Ported from UniPASE (MIT), which builds on WavLM by Microsoft (MIT) and on
//! the Vocos backbone by way of WavTokenizer (MIT).

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, GroupNorm, LayerNorm, Linear, Module, VarBuilder,
};

use crate::enhance::unipase::config::{
    ATTENTION_HEADS, BACKBONE_DIM, BACKBONE_FF, BACKBONE_LAYERS,
    BACKBONE_NORM_EPS, CONV_LAYERS, CONV_POS, CONV_POS_GROUPS, ENCODER_DIM,
    ENCODER_LAYERS, ENCODER_NORM_EPS, FFN_DIM, GREP_DIM, MAX_DISTANCE,
    NUM_BUCKETS, POS_NET_ATTN, POS_NET_GROUPS, POS_NET_RES, TAP_ACOUSTIC,
    TAP_NORM_EPS, TAP_PHONETIC, bins, head_dim,
};

/// A plain convolution over time, with the padding the reference gives it.
fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    cfg: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight =
        vb.get((out_channels, in_channels / cfg.groups, kernel), "weight")?;
    let bias = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}

/// The extractor's convolutions carry no bias.
fn conv1d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let cfg = Conv1dConfig {
        stride,
        ..Default::default()
    };
    let weight = vb.get((out_channels, in_channels, kernel), "weight")?;
    Ok(Conv1d::new(weight, None, cfg))
}

/// A layer norm over the last axis, in the eps the reference gives it.
fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        vb.get(size, "weight")?,
        vb.get(size, "bias")?,
        eps,
    ))
}

/// The convolutional feature extractor: seven strided blocks that take 16 kHz
/// samples down to 50 frames a second.
///
/// In `layer_norm` mode — the mode this checkpoint is in — every block is
/// convolution → layer norm over channels → GELU, and no convolution has a
/// bias. The stride product is 320, which is why one frame here is one packet
/// there and one vocoder frame later.
#[derive(Debug)]
struct ConvExtractor {
    blocks: Vec<(Conv1d, LayerNorm)>,
}

impl ConvExtractor {
    fn load(vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::with_capacity(CONV_LAYERS.len());
        let mut in_channels = 1;
        for (index, &(channels, kernel, stride)) in
            CONV_LAYERS.iter().enumerate()
        {
            let block = vb.pp(index.to_string());
            blocks.push((
                conv1d_no_bias(
                    in_channels,
                    channels,
                    kernel,
                    stride,
                    block.pp("0"),
                )?,
                // The norm is the second element of the block's inner
                // Sequential, which is itself the block's third element.
                layer_norm(channels, ENCODER_NORM_EPS, block.pp("2").pp("1"))?,
            ));
            in_channels = channels;
        }
        Ok(Self { blocks })
    }

    /// Takes `[1, samples]` to `[1, channels, frames]`.
    fn forward(&self, samples: &Tensor) -> Result<Tensor> {
        let mut hidden = samples.unsqueeze(1)?;
        for (conv, norm) in &self.blocks {
            hidden = conv.forward(&hidden)?;
            hidden = norm
                .forward(&hidden.transpose(1, 2)?.contiguous()?)?
                .transpose(1, 2)?
                .contiguous()?;
            hidden = hidden.gelu_erf()?;
        }
        Ok(hidden)
    }
}

/// One transformer layer of the encoder, in the pre-norm arrangement the
/// checkpoint is trained in.
#[derive(Debug)]
struct EncoderLayer {
    attention: Attention,
    attn_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_norm: LayerNorm,
}

impl EncoderLayer {
    fn load(first: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(first, vb.pp("self_attn"))?,
            attn_norm: layer_norm(
                ENCODER_DIM,
                ENCODER_NORM_EPS,
                vb.pp("self_attn_layer_norm"),
            )?,
            fc1: candle_nn::linear(ENCODER_DIM, FFN_DIM, vb.pp("fc1"))?,
            fc2: candle_nn::linear(FFN_DIM, ENCODER_DIM, vb.pp("fc2"))?,
            final_norm: layer_norm(
                ENCODER_DIM,
                ENCODER_NORM_EPS,
                vb.pp("final_layer_norm"),
            )?,
        })
    }

    /// `hidden` is `[1, frames, dim]`; `bias` is the shared relative position
    /// bias, `[heads, frames, frames]`.
    fn forward(&self, hidden: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let normed = self.attn_norm.forward(hidden)?;
        let attended = self.attention.forward(&normed, bias)?;
        let hidden = (hidden + attended)?;

        let normed = self.final_norm.forward(&hidden)?;
        let ffn = self.fc2.forward(&self.fc1.forward(&normed)?.gelu_erf()?)?;
        hidden + ffn
    }
}

/// Self-attention with the gated relative position bias WavLM adds.
///
/// The bias itself is computed once, by the first layer, from a bucketed
/// distance table; every layer then **re-weights** it with a gate of its own,
/// driven by the query. That gate is the `gru_rel_pos` part: eight numbers per
/// query position, summed down to two, squashed, and combined into one scale
/// per position. So the layers share what "eleven frames apart" means and
/// disagree about how much it should matter here.
#[derive(Debug)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    grep_linear: Linear,
    grep_a: Tensor,
    /// Only the first layer carries the bias table.
    rel_bias: Option<Tensor>,
}

impl Attention {
    fn load(first: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: candle_nn::linear(
                ENCODER_DIM,
                ENCODER_DIM,
                vb.pp("q_proj"),
            )?,
            k_proj: candle_nn::linear(
                ENCODER_DIM,
                ENCODER_DIM,
                vb.pp("k_proj"),
            )?,
            v_proj: candle_nn::linear(
                ENCODER_DIM,
                ENCODER_DIM,
                vb.pp("v_proj"),
            )?,
            out_proj: candle_nn::linear(
                ENCODER_DIM,
                ENCODER_DIM,
                vb.pp("out_proj"),
            )?,
            grep_linear: candle_nn::linear(
                head_dim(),
                GREP_DIM,
                vb.pp("grep_linear"),
            )?,
            grep_a: vb.get((1, ATTENTION_HEADS, 1, 1), "grep_a")?,
            rel_bias: if first {
                Some(
                    vb.pp("relative_attention_bias")
                        .get((NUM_BUCKETS, ATTENTION_HEADS), "weight")?,
                )
            } else {
                None
            },
        })
    }

    /// The bias table this layer contributes, `[heads, frames, frames]`.
    fn position_bias(&self, frames: usize) -> Result<Option<Tensor>> {
        let Some(table) = &self.rel_bias else {
            return Ok(None);
        };
        let buckets = relative_buckets(frames);
        let index = Tensor::from_vec(buckets, frames * frames, table.device())?;
        let values = table.index_select(&index, 0)?;
        Ok(Some(
            values
                .reshape((frames, frames, ATTENTION_HEADS))?
                .permute((2, 0, 1))?
                .contiguous()?,
        ))
    }

    fn forward(&self, hidden: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let (batch, frames, _) = hidden.dims3()?;
        let heads = ATTENTION_HEADS;
        let dim = head_dim();
        let split = |x: Tensor| -> Result<Tensor> {
            x.reshape((batch, frames, heads, dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let query = split(self.q_proj.forward(hidden)?)?;
        let key = split(self.k_proj.forward(hidden)?)?;
        let value = split(self.v_proj.forward(hidden)?)?;

        // The gate reads the layer's **input** split into heads, not the
        // projected query — `grep_linear` is 64 wide, and 64 is the head
        // width, which makes the two easy to confuse.
        let gate = self.gate(&split(hidden.clone())?, batch, frames)?;
        let biased = bias.broadcast_mul(&gate)?;

        let scale = 1.0 / (dim as f64).sqrt();
        let scores = (query * scale)?.matmul(&key.transpose(2, 3)?)?;
        let scores = scores.broadcast_add(&biased)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = weights.matmul(&value)?;
        self.out_proj.forward(&context.transpose(1, 2)?.reshape((
            batch,
            frames,
            heads * dim,
        ))?)
    }

    /// The per-query scale on the shared bias, `[heads, frames, 1]`.
    ///
    /// `hidden` is the layer's input reshaped to `[batch, heads, frames,
    /// head_dim]`.
    fn gate(
        &self,
        hidden: &Tensor,
        batch: usize,
        frames: usize,
    ) -> Result<Tensor> {
        let heads = ATTENTION_HEADS;
        let raw = self.grep_linear.forward(hidden)?;
        let halves = raw
            .reshape((batch, heads, frames, 2, GREP_DIM / 2))?
            .sum(D::Minus1)?;
        let squashed = candle_nn::ops::sigmoid(&halves)?;
        let gate_a = squashed.narrow(D::Minus1, 0, 1)?;
        let gate_b = squashed.narrow(D::Minus1, 1, 1)?;
        let scaled = gate_b.broadcast_mul(&self.grep_a)?;
        let gate = ((gate_a * (scaled - 1.0)?)? + 2.0)?;
        gate.reshape((batch * heads, frames, 1))
    }
}

/// The bucket every (query, key) distance falls into.
///
/// Bidirectional: the top half of the buckets is for keys that come after the
/// query. Within a direction the first 80 buckets are exact — one per frame of
/// distance — and the rest are logarithmic out to 800 frames, which is 16
/// seconds, past which everything shares the last bucket.
pub(crate) fn relative_buckets(frames: usize) -> Vec<u32> {
    let half = NUM_BUCKETS / 2;
    let exact = half / 2;
    let ratio = (MAX_DISTANCE as f64 / exact as f64).ln();
    let mut out = Vec::with_capacity(frames * frames);
    for query in 0..frames {
        for key in 0..frames {
            let distance = key as i64 - query as i64;
            let forward = if distance > 0 { half } else { 0 };
            let magnitude = distance.unsigned_abs() as usize;
            let bucket = if magnitude < exact {
                magnitude
            } else {
                let scaled = exact +
                    ((magnitude as f64 / exact as f64).ln() / ratio *
                        (half - exact) as f64) as usize;
                scaled.min(half - 1)
            };
            out.push((forward + bucket) as u32);
        }
    }
    out
}

/// The encoder: DeWavLM-Omni, a WavLM-Large fine-tuned for enhancement.
///
/// It is run for two of its outputs at once. The first transformer layer still
/// carries the acoustics — what the room and the microphone did — and the
/// twenty-fourth carries what is being said. The adapter is given both: the
/// deep one says what the speech is, the shallow one says what it sounded like,
/// and the enhanced representation has to be true to both.
pub struct Encoder {
    extractor: ConvExtractor,
    post_norm: LayerNorm,
    proj: Linear,
    mask_emb: Tensor,
    pos_conv: Conv1d,
    layers: Vec<EncoderLayer>,
    device: Device,
    dtype: DType,
}

/// Every stage of one encoder pass, in order — what the parity tests compare
/// against the reference's own dump.
#[cfg(test)]
pub(crate) struct Stages {
    /// The extractor's output, `[1, channels, frames]`.
    pub conv_out: Tensor,
    /// Its normalization, `[1, frames, channels]`.
    pub conv_norm: Tensor,
    /// After the projection to the transformer's width.
    pub projected: Tensor,
    /// After the lost frames are replaced by the mask embedding.
    pub masked: Tensor,
    /// After the convolutional positional embedding is added.
    pub positioned: Tensor,
    /// The output of every transformer layer, in order.
    pub layers: Vec<Tensor>,
}

impl Encoder {
    /// Loads the encoder from the converted checkpoint.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let pos_cfg = Conv1dConfig {
            padding: CONV_POS / 2,
            groups: CONV_POS_GROUPS,
            ..Default::default()
        };
        let (last, _, _) = CONV_LAYERS[CONV_LAYERS.len() - 1];
        Ok(Self {
            extractor: ConvExtractor::load(
                vb.pp("feature_extractor").pp("conv_layers"),
            )?,
            post_norm: layer_norm(last, ENCODER_NORM_EPS, vb.pp("layer_norm"))?,
            proj: candle_nn::linear(
                last,
                ENCODER_DIM,
                vb.pp("post_extract_proj"),
            )?,
            mask_emb: vb.get(ENCODER_DIM, "mask_emb")?,
            pos_conv: conv1d(
                ENCODER_DIM,
                ENCODER_DIM,
                CONV_POS,
                pos_cfg,
                vb.pp("encoder").pp("pos_conv").pp("0"),
            )?,
            layers: (0..ENCODER_LAYERS)
                .map(|index| {
                    EncoderLayer::load(
                        index == 0,
                        vb.pp("encoder").pp("layers").pp(index.to_string()),
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// The device the encoder is on.
    pub fn device(&self) -> &Device { &self.device }

    /// The dtype the encoder computes in.
    pub fn dtype(&self) -> DType { self.dtype }

    /// The two tapped representations of one aligned window, each normalized
    /// over time and channels together, both `[1, frames, dim]`.
    ///
    /// `lost` marks frames the packet-loss detector flagged; those frames are
    /// replaced by the model's learned mask embedding before the transformer
    /// sees them, which is the whole of concealment.
    pub fn features(
        &self,
        samples: &Tensor,
        lost: &[bool],
    ) -> Result<(Tensor, Tensor)> {
        let extracted = self.extractor.forward(samples)?;
        let extracted = extracted.transpose(1, 2)?.contiguous()?;
        let normed = self.post_norm.forward(&extracted)?;
        let projected = self.proj.forward(&normed)?;
        let masked = self.apply_mask(&projected, lost)?;

        let positioned = self.positional(&masked)?;
        let mut hidden = positioned;
        let frames = hidden.dim(1)?;
        let bias = self.layers[0]
            .attention
            .position_bias(frames)?
            .expect("the first layer carries the bias table");
        let bias = bias.to_dtype(self.dtype)?;

        let mut acoustic = None;
        let mut phonetic = None;
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &bias)?;
            // `layer_reps` counts the extractor's output as 0, so layer `i`
            // produces representation `i + 1`.
            match index + 1 {
                TAP_ACOUSTIC => acoustic = Some(hidden.clone()),
                TAP_PHONETIC => phonetic = Some(hidden.clone()),
                _ => {},
            }
        }
        let acoustic = acoustic.expect("the acoustic tap is within the stack");
        let phonetic = phonetic.expect("the deep tap is the last layer");
        Ok((normalize_tap(&acoustic)?, normalize_tap(&phonetic)?))
    }

    /// Every stage of one pass, for the parity tests.
    #[cfg(test)]
    pub(crate) fn stages(
        &self,
        samples: &Tensor,
        lost: &[bool],
    ) -> Result<Stages> {
        let conv_out = self.extractor.forward(samples)?;
        let conv_norm = self
            .post_norm
            .forward(&conv_out.transpose(1, 2)?.contiguous()?)?;
        let projected = self.proj.forward(&conv_norm)?;
        let masked = self.apply_mask(&projected, lost)?;
        let positioned = self.positional(&masked)?;
        let frames = positioned.dim(1)?;
        let bias = self.layers[0]
            .attention
            .position_bias(frames)?
            .expect("the first layer carries the bias table")
            .to_dtype(self.dtype)?;
        let mut hidden = positioned.clone();
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &bias)?;
            layers.push(hidden.clone());
        }
        Ok(Stages {
            conv_out,
            conv_norm,
            projected,
            masked,
            positioned,
            layers,
        })
    }

    /// Replaces the flagged frames with the learned mask embedding.
    fn apply_mask(&self, hidden: &Tensor, lost: &[bool]) -> Result<Tensor> {
        if !lost.iter().any(|&flag| flag) {
            return Ok(hidden.clone());
        }
        let frames = hidden.dim(1)?;
        let keep: Vec<f32> = (0..frames)
            .map(|frame| f32::from(!lost.get(frame).copied().unwrap_or(false)))
            .collect();
        let keep = Tensor::from_vec(keep, (1, frames, 1), &self.device)?
            .to_dtype(self.dtype)?;
        let fill = (1.0 - &keep)?;
        let embedding = self
            .mask_emb
            .reshape((1, 1, ENCODER_DIM))?
            .to_dtype(self.dtype)?;
        hidden.broadcast_mul(&keep)? + fill.broadcast_mul(&embedding)?
    }

    /// Adds the convolutional positional embedding.
    ///
    /// The kernel is 128 taps wide and padded by 64, which leaves one sample
    /// too many; the reference drops the last one rather than padding
    /// asymmetrically.
    fn positional(&self, hidden: &Tensor) -> Result<Tensor> {
        let frames = hidden.dim(1)?;
        let conv = self
            .pos_conv
            .forward(&hidden.transpose(1, 2)?.contiguous()?)?;
        let conv = conv.narrow(2, 0, frames)?.gelu_erf()?;
        hidden + conv.transpose(1, 2)?
    }
}

/// The normalization each tap gets: over time **and** channels together, with
/// no learned affine.
///
/// Not the usual per-frame layer norm — the reference normalizes over
/// `feat.shape[1:]`, which for a `[batch, frames, dim]` tensor is both trailing
/// axes at once. That makes the tap scale-free as a whole rather than
/// frame by frame, which matters because the adapter adds the two taps
/// together and they come from layers whose activations differ by an order of
/// magnitude.
fn normalize_tap(hidden: &Tensor) -> Result<Tensor> {
    let (batch, frames, dim) = hidden.dims3()?;
    let flat = hidden
        .reshape((batch, frames * dim))?
        .to_dtype(DType::F32)?;
    let mean = flat.mean_keepdim(1)?;
    let centred = flat.broadcast_sub(&mean)?;
    let variance = centred.sqr()?.mean_keepdim(1)?;
    let normed = centred.broadcast_div(&(variance + TAP_NORM_EPS)?.sqrt()?)?;
    normed
        .reshape((batch, frames, dim))?
        .to_dtype(hidden.dtype())
}

/// A depthwise convolution over time, written out as shifted scaled copies.
///
/// candle implements a grouped convolution by splitting the input into
/// `groups` chunks and convolving each one, which for a depthwise block means
/// one convolution **per channel** — a thousand of them, twenty-four times per
/// window. On a CPU that is merely wasteful; on a GPU the dispatch overhead
/// swamps the arithmetic completely. A depthwise convolution is a weighted sum
/// of `kernel` time-shifted copies of the input, so this writes it that way:
/// thirteen kernel launches instead of a thousand convolutions, and exactly the
/// same numbers.
#[derive(Debug)]
struct Depthwise {
    /// One `[1, channels, 1]` weight per tap.
    taps: Vec<Tensor>,
    bias: Tensor,
    padding: usize,
}

impl Depthwise {
    fn load(channels: usize, kernel: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((channels, 1, kernel), "weight")?;
        Ok(Self {
            taps: (0..kernel)
                .map(|tap| weight.narrow(2, tap, 1)?.reshape((1, channels, 1)))
                .collect::<Result<Vec<_>>>()?,
            bias: vb.get(channels, "bias")?.reshape((1, channels, 1))?,
            padding: kernel / 2,
        })
    }

    /// `hidden` is `[1, channels, time]`, and so is the result.
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let (_, _, time) = hidden.dims3()?;
        let padded = hidden.pad_with_zeros(2, self.padding, self.padding)?;
        let mut sum: Option<Tensor> = None;
        for (tap, weight) in self.taps.iter().enumerate() {
            let shifted = padded.narrow(2, tap, time)?;
            let scaled = shifted.broadcast_mul(weight)?;
            sum = Some(match sum {
                Some(acc) => (acc + scaled)?,
                None => scaled,
            });
        }
        sum.expect("a kernel has at least one tap")
            .broadcast_add(&self.bias)
    }
}

/// A ConvNeXt block of the Vocos backbone: a depthwise convolution over time,
/// a norm, a pointwise mixer, and a per-channel scale on the residual branch.
#[derive(Debug)]
struct ConvNeXtBlock {
    dwconv: Depthwise,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            dwconv: Depthwise::load(BACKBONE_DIM, 7, vb.pp("dwconv"))?,
            norm: layer_norm(BACKBONE_DIM, BACKBONE_NORM_EPS, vb.pp("norm"))?,
            pwconv1: candle_nn::linear(
                BACKBONE_DIM,
                BACKBONE_FF,
                vb.pp("pwconv1"),
            )?,
            pwconv2: candle_nn::linear(
                BACKBONE_FF,
                BACKBONE_DIM,
                vb.pp("pwconv2"),
            )?,
            gamma: vb.get(BACKBONE_DIM, "gamma")?,
        })
    }

    /// `hidden` is `[1, channels, frames]`.
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let residual = hidden;
        let mixed = self.dwconv.forward(hidden)?;
        let mixed = self.norm.forward(&mixed.transpose(1, 2)?.contiguous()?)?;
        let mixed = self.pwconv1.forward(&mixed)?.gelu_erf()?;
        let mixed = self.pwconv2.forward(&mixed)?;
        let mixed = mixed.broadcast_mul(&self.gamma)?;
        residual + mixed.transpose(1, 2)?
    }
}

/// A residual block of the backbone's positional network.
#[derive(Debug)]
struct ResBlock {
    norm1: GroupNorm,
    conv1: Conv1d,
    norm2: GroupNorm,
    conv2: Conv1d,
}

impl ResBlock {
    fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            norm1: candle_nn::group_norm(
                POS_NET_GROUPS,
                BACKBONE_DIM,
                BACKBONE_NORM_EPS,
                vb.pp("norm1"),
            )?,
            conv1: conv1d(BACKBONE_DIM, BACKBONE_DIM, 3, cfg, vb.pp("conv1"))?,
            norm2: candle_nn::group_norm(
                POS_NET_GROUPS,
                BACKBONE_DIM,
                BACKBONE_NORM_EPS,
                vb.pp("norm2"),
            )?,
            conv2: conv1d(BACKBONE_DIM, BACKBONE_DIM, 3, cfg, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let branch = swish(&self.norm1.forward(hidden)?)?;
        let branch = self.conv1.forward(&branch)?;
        let branch = swish(&self.norm2.forward(&branch)?)?;
        let branch = self.conv2.forward(&branch)?;
        hidden + branch
    }
}

/// The reference's `nonlinearity`: `x · σ(x)`.
fn swish(hidden: &Tensor) -> Result<Tensor> {
    hidden * candle_nn::ops::sigmoid(hidden)?
}

/// The single-headed attention in the middle of the positional network.
#[derive(Debug)]
struct AttnBlock {
    norm: GroupNorm,
    query: Conv1d,
    key: Conv1d,
    value: Conv1d,
    out: Conv1d,
}

impl AttnBlock {
    fn load(vb: VarBuilder) -> Result<Self> {
        let point = |name: &str| {
            conv1d(
                BACKBONE_DIM,
                BACKBONE_DIM,
                1,
                Conv1dConfig::default(),
                vb.pp(name),
            )
        };
        Ok(Self {
            norm: candle_nn::group_norm(
                POS_NET_GROUPS,
                BACKBONE_DIM,
                BACKBONE_NORM_EPS,
                vb.pp("norm"),
            )?,
            query: point("q")?,
            key: point("k")?,
            value: point("v")?,
            out: point("proj_out")?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let normed = self.norm.forward(hidden)?;
        let query = self.query.forward(&normed)?;
        let key = self.key.forward(&normed)?;
        let value = self.value.forward(&normed)?;
        let scale = 1.0 / (BACKBONE_DIM as f64).sqrt();
        // `[1, frames, channels] · [1, channels, frames]`, normalized over the
        // key axis.
        let scores =
            (query.transpose(1, 2)?.contiguous()?.matmul(&key)? * scale)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = value.matmul(&weights.transpose(1, 2)?.contiguous()?)?;
        hidden + self.out.forward(&context)?
    }
}

/// One of the two Vocos backbones. Same geometry both times; only what feeds
/// them and what reads them differs.
pub struct Backbone {
    embed: Conv1d,
    res_in: Vec<ResBlock>,
    attn: Vec<AttnBlock>,
    res_out: Vec<ResBlock>,
    pos_norm: GroupNorm,
    norm: LayerNorm,
    blocks: Vec<ConvNeXtBlock>,
    final_norm: LayerNorm,
}

impl Backbone {
    /// Loads a backbone from the checkpoint's `decoder.` subtree.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let pos_net = vb.pp("pos_net");
        let half = POS_NET_RES / 2;
        let res = |from: usize, count: usize| {
            (0..count)
                .map(|index| {
                    ResBlock::load(pos_net.pp((from + index).to_string()))
                })
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            embed: conv1d(BACKBONE_DIM, BACKBONE_DIM, 7, cfg, vb.pp("embed"))?,
            res_in: res(0, half)?,
            attn: (0..POS_NET_ATTN)
                .map(|index| {
                    AttnBlock::load(pos_net.pp((half + index).to_string()))
                })
                .collect::<Result<Vec<_>>>()?,
            res_out: res(half + POS_NET_ATTN, half)?,
            pos_norm: candle_nn::group_norm(
                POS_NET_GROUPS,
                BACKBONE_DIM,
                BACKBONE_NORM_EPS,
                pos_net.pp((POS_NET_RES + POS_NET_ATTN).to_string()),
            )?,
            norm: layer_norm(BACKBONE_DIM, BACKBONE_NORM_EPS, vb.pp("norm"))?,
            blocks: (0..BACKBONE_LAYERS)
                .map(|index| {
                    ConvNeXtBlock::load(vb.pp("convnext").pp(index.to_string()))
                })
                .collect::<Result<Vec<_>>>()?,
            final_norm: layer_norm(
                BACKBONE_DIM,
                BACKBONE_NORM_EPS,
                vb.pp("final_layer_norm"),
            )?,
        })
    }

    /// `hidden` is `[1, channels, frames]`, and so is the result.
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let mut hidden = self.embed.forward(hidden)?;
        for block in &self.res_in {
            hidden = block.forward(&hidden)?;
        }
        for block in &self.attn {
            hidden = block.forward(&hidden)?;
        }
        for block in &self.res_out {
            hidden = block.forward(&hidden)?;
        }
        hidden = self.pos_norm.forward(&hidden)?;
        hidden = self
            .norm
            .forward(&hidden.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        self.final_norm
            .forward(&hidden.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()
    }
}

/// The adapter: the network that does the enhancing.
///
/// Its input is the two taps summed — the deep one first put through a linear
/// map of its own — and its output is what the acoustic tap *should* have been
/// if the recording had been clean. Everything downstream is reconstruction.
pub struct Adapter {
    proj: Linear,
    backbone: Backbone,
    head: Linear,
}

impl Adapter {
    /// Loads the adapter.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj: candle_nn::linear(BACKBONE_DIM, BACKBONE_DIM, vb.pp("proj"))?,
            backbone: Backbone::load(vb.pp("decoder"))?,
            head: candle_nn::linear(BACKBONE_DIM, BACKBONE_DIM, vb.pp("head"))?,
        })
    }

    /// The projection the deep tap goes through before the two are summed.
    #[cfg(test)]
    pub(crate) fn project(&self, phonetic: &Tensor) -> Result<Tensor> {
        self.proj.forward(phonetic)
    }

    /// The backbone, for the parity tests.
    #[cfg(test)]
    pub(crate) fn backbone(&self) -> &Backbone { &self.backbone }

    /// Both taps are `[1, frames, dim]`; so is the result.
    pub fn forward(
        &self,
        acoustic: &Tensor,
        phonetic: &Tensor,
    ) -> Result<Tensor> {
        let summed = (self.proj.forward(phonetic)? + acoustic)?;
        let hidden = self
            .backbone
            .forward(&summed.transpose(1, 2)?.contiguous()?)?;
        self.head.forward(&hidden.transpose(1, 2)?.contiguous()?)
    }
}

/// The vocoder: enhanced representation in, waveform out.
///
/// The network stops at a complex spectrum — a log-magnitude and a phase per
/// bin per frame — and the inverse transform that finishes the job runs on the
/// host, in [`istft`](super::super::istft), shared by both runtimes.
pub struct Vocoder {
    backbone: Backbone,
    out: Linear,
}

impl Vocoder {
    /// Loads the vocoder.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            backbone: Backbone::load(vb.pp("decoder"))?,
            out: candle_nn::linear(
                BACKBONE_DIM,
                2 * bins(),
                vb.pp("head").pp("out"),
            )?,
        })
    }

    /// The backbone, for the parity tests.
    #[cfg(test)]
    pub(crate) fn backbone(&self) -> &Backbone { &self.backbone }

    /// `hidden` is `[1, frames, dim]`. The result is the head's raw
    /// `[2 · bins, frames]` output as host values, ready for
    /// [`istft::spectrum_to_wave`](super::super::istft::spectrum_to_wave).
    pub fn spectrum(&self, hidden: &Tensor) -> Result<Vec<f32>> {
        let backbone = self
            .backbone
            .forward(&hidden.transpose(1, 2)?.contiguous()?)?;
        let raw = self
            .out
            .forward(&backbone.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()?;
        raw.i(0)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
    }
}
