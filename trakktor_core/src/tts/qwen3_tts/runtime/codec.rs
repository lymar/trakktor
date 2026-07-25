//! The codec decoder: frames of codes in, a 24 kHz waveform out.
//!
//! Purely feed-forward — no sampling, no state carried between calls — so with
//! the codes fixed its output is reproducible exactly. That makes it the
//! reference boundary the port is held to most strictly.
//!
//! The stages, in order: dequantize the 16 codebooks of every frame and sum
//! them; widen with a causal convolution; refine with a stack of
//! windowed-attention transformer layers; upsample twice through ConvNeXt
//! blocks; then run the convolutional decoder, whose four blocks together with
//! the earlier stages upsample by the full frame rate.
//!
//! Long inputs are decoded in chunks with a left context, exactly as the
//! reference does, so the result does not depend on the input's length.

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, ops::softmax};

use super::layers::{
    CausalConv1d, CausalConvTranspose1d, LayerScale, RmsNorm, SnakeBeta,
    rope_tables, rotate_half,
};
use crate::tts::qwen3_tts::{
    chunking::{chunk_plan, window_visible},
    config::CodecConfig,
};

/// Guard on the divisor when turning accumulated codebook sums into entries.
const CLUSTER_USAGE_EPS: f64 = 1e-5;

/// One residual vector-quantization group: a stack of codebooks whose
/// dequantized entries are summed, then projected back to the codec width.
#[derive(Debug)]
struct QuantizerGroup {
    /// Per-codebook entry tables, already divided by their usage counts.
    codebooks: Vec<Tensor>,
    /// The 1×1 convolution projecting the summed entries out, held as a matrix
    /// because a 1×1 convolution is a matrix multiply.
    output_proj: Tensor,
}

impl QuantizerGroup {
    /// Loads `count` codebooks and the group's output projection.
    fn load(count: usize, cfg: &CodecConfig, vb: VarBuilder) -> Result<Self> {
        let mut codebooks = Vec::with_capacity(count);
        for index in 0..count {
            let vb = vb.pp("vq").pp("layers").pp(index).pp("_codebook");
            let sums = vb
                .get((cfg.codebook_size, cfg.quantizer_dim), "embedding_sum")?;
            let usage = vb.get(cfg.codebook_size, "cluster_usage")?;
            // Entries are stored as running sums; dividing by the usage count
            // recovers the centroid.
            let usage = usage
                .to_dtype(DType::F32)?
                .clamp(CLUSTER_USAGE_EPS, f64::INFINITY)?
                .reshape((cfg.codebook_size, 1))?;
            codebooks.push(sums.to_dtype(DType::F32)?.broadcast_div(&usage)?);
        }
        let output_proj = vb
            .get(
                (cfg.codebook_dim, cfg.quantizer_dim, 1),
                "output_proj.weight",
            )?
            .to_dtype(DType::F32)?
            .squeeze(D::Minus1)?;
        Ok(Self {
            codebooks,
            output_proj,
        })
    }

    /// Dequantizes `codes`, shaped `[codebooks, frames]`, into
    /// `[codebook_dim, frames]`.
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let mut summed: Option<Tensor> = None;
        for (index, codebook) in self.codebooks.iter().enumerate() {
            let picked = codebook.index_select(&codes.i(index)?, 0)?;
            summed = Some(match summed {
                None => picked,
                Some(acc) => (acc + picked)?,
            });
        }
        let summed =
            summed.expect("a quantizer group has at least one codebook");
        // [frames, quantizer_dim] · [quantizer_dim, codebook_dim] → transposed
        // to the channels-first layout the convolutions expect.
        summed.matmul(&self.output_proj.t()?)?.t()
    }
}

/// The split quantizer: one semantic codebook plus the acoustic residuals,
/// each group projected separately and then summed.
#[derive(Debug)]
struct Quantizer {
    semantic: QuantizerGroup,
    acoustic: QuantizerGroup,
    num_semantic: usize,
}

impl Quantizer {
    fn load(cfg: &CodecConfig, vb: VarBuilder) -> Result<Self> {
        let num_semantic = cfg.num_semantic_quantizers;
        Ok(Self {
            semantic: QuantizerGroup::load(
                num_semantic,
                cfg,
                vb.pp("rvq_first"),
            )?,
            acoustic: QuantizerGroup::load(
                cfg.num_quantizers - num_semantic,
                cfg,
                vb.pp("rvq_rest"),
            )?,
            num_semantic,
        })
    }

    /// Dequantizes all codebooks of `codes`, shaped `[codebooks, frames]`.
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let total = codes.dim(0)?;
        let semantic =
            self.semantic
                .decode(&codes.narrow(0, 0, self.num_semantic)?)?;
        let acoustic = self.acoustic.decode(&codes.narrow(
            0,
            self.num_semantic,
            total - self.num_semantic,
        )?)?;
        semantic + acoustic
    }
}

/// A ConvNeXt block: depthwise convolution, normalization, and a widening
/// pointwise pair, added back onto the input through a learned scale.
#[derive(Debug)]
struct ConvNextBlock {
    dwconv: CausalConv1d,
    norm: candle_nn::LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNextBlock {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            // Depthwise: one group per channel.
            dwconv: CausalConv1d::load(dim, dim, 7, 1, dim, vb.pp("dwconv"))?,
            norm: candle_nn::LayerNorm::new(
                vb.get(dim, "norm.weight")?,
                vb.get(dim, "norm.bias")?,
                1e-6,
            ),
            pwconv1: Linear::new(
                vb.get((4 * dim, dim), "pwconv1.weight")?,
                Some(vb.get(4 * dim, "pwconv1.bias")?),
            ),
            pwconv2: Linear::new(
                vb.get((dim, 4 * dim), "pwconv2.weight")?,
                Some(vb.get(dim, "pwconv2.bias")?),
            ),
            gamma: vb.get(dim, "gamma")?,
        })
    }

    /// Runs the block over `xs`, shaped `[batch, channels, time]`.
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let hidden = self.dwconv.forward(xs)?.transpose(1, 2)?;
        let hidden = self.norm.forward(&hidden)?;
        let hidden = self.pwconv1.forward(&hidden)?.gelu_erf()?;
        let hidden = self.pwconv2.forward(&hidden)?;
        let hidden = hidden.broadcast_mul(&self.gamma)?.transpose(1, 2)?;
        residual + hidden
    }
}

/// One transformer layer of the pre-decoder stack: windowed self-attention and
/// a gated feed-forward, each added back through its own learned scale.
#[derive(Debug)]
struct TransformerLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    attn_scale: LayerScale,
    mlp_scale: LayerScale,
    num_heads: usize,
    head_dim: usize,
}

impl TransformerLayer {
    fn load(cfg: &CodecConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let inner = cfg.num_attention_heads * cfg.head_dim;
        let no_bias = |shape: (usize, usize), name: &str| -> Result<Linear> {
            Ok(Linear::new(vb.get(shape, name)?, None))
        };
        Ok(Self {
            input_layernorm: RmsNorm::load(
                hidden,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::load(
                hidden,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            q_proj: no_bias((inner, hidden), "self_attn.q_proj.weight")?,
            k_proj: no_bias((inner, hidden), "self_attn.k_proj.weight")?,
            v_proj: no_bias((inner, hidden), "self_attn.v_proj.weight")?,
            o_proj: no_bias((hidden, inner), "self_attn.o_proj.weight")?,
            gate_proj: no_bias(
                (cfg.intermediate_size, hidden),
                "mlp.gate_proj.weight",
            )?,
            up_proj: no_bias(
                (cfg.intermediate_size, hidden),
                "mlp.up_proj.weight",
            )?,
            down_proj: no_bias(
                (hidden, cfg.intermediate_size),
                "mlp.down_proj.weight",
            )?,
            attn_scale: LayerScale::load(
                hidden,
                vb.pp("self_attn_layer_scale"),
            )?,
            mlp_scale: LayerScale::load(hidden, vb.pp("mlp_layer_scale"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// Runs the layer over `xs`, shaped `[batch, time, hidden]`.
    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let shape = (batch, seq, self.num_heads, self.head_dim);

        let normed = self.input_layernorm.forward(xs)?;
        let query = self
            .q_proj
            .forward(&normed)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let key = self
            .k_proj
            .forward(&normed)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let value = self
            .v_proj
            .forward(&normed)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;

        let query = apply_rope(&query, cos, sin)?;
        let key = apply_rope(&key, cos, sin)?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let weights = (query.matmul(&key.transpose(2, 3)?)? * scale)?;
        let weights = weights.broadcast_add(mask)?;
        // The reference accumulates attention in full precision regardless of
        // the weights' dtype.
        let weights = softmax(&weights.to_dtype(DType::F32)?, D::Minus1)?
            .to_dtype(value.dtype())?;
        let attended = weights.matmul(&value)?.transpose(1, 2)?.reshape((
            batch,
            seq,
            self.num_heads * self.head_dim,
        ))?;
        let attended = self.o_proj.forward(&attended)?;
        let xs = (xs + self.attn_scale.forward(&attended)?)?;

        let normed = self.post_attention_layernorm.forward(&xs)?;
        let gated = (self.gate_proj.forward(&normed)?.silu()? *
            self.up_proj.forward(&normed)?)?;
        let projected = self.down_proj.forward(&gated)?;
        xs + self.mlp_scale.forward(&projected)?
    }
}

/// Applies rotary embeddings to `xs`, shaped `[batch, heads, time, head_dim]`.
fn apply_rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let rotated = rotate_half(xs)?;
    xs.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?
}

/// One block of the convolutional decoder: activate, upsample, then three
/// dilated residual units.
#[derive(Debug)]
struct DecoderBlock {
    act: SnakeBeta,
    upsample: CausalConvTranspose1d,
    units: Vec<ResidualUnit>,
}

/// The dilations the reference gives the three residual units of every block.
const RESIDUAL_DILATIONS: [usize; 3] = [1, 3, 9];

impl DecoderBlock {
    fn load(
        in_dim: usize,
        out_dim: usize,
        rate: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut units = Vec::with_capacity(RESIDUAL_DILATIONS.len());
        for (index, dilation) in RESIDUAL_DILATIONS.iter().enumerate() {
            // The units follow the activation and the upsample in the block.
            units.push(ResidualUnit::load(
                out_dim,
                *dilation,
                vb.pp(index + 2),
            )?);
        }
        Ok(Self {
            act: SnakeBeta::load(in_dim, vb.pp(0))?,
            upsample: CausalConvTranspose1d::load(
                in_dim,
                out_dim,
                2 * rate,
                rate,
                vb.pp(1),
            )?,
            units,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut hidden = self.upsample.forward(&self.act.forward(xs)?)?;
        for unit in &self.units {
            hidden = unit.forward(&hidden)?;
        }
        Ok(hidden)
    }
}

/// A residual unit: two activated convolutions added back onto the input.
#[derive(Debug)]
struct ResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv1d,
    act2: SnakeBeta,
    conv2: CausalConv1d,
}

impl ResidualUnit {
    fn load(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::load(dim, vb.pp("act1"))?,
            conv1: CausalConv1d::load(
                dim,
                dim,
                7,
                dilation,
                1,
                vb.pp("conv1"),
            )?,
            act2: SnakeBeta::load(dim, vb.pp("act2"))?,
            conv2: CausalConv1d::load(dim, dim, 1, 1, 1, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = self.conv1.forward(&self.act1.forward(xs)?)?;
        let hidden = self.conv2.forward(&self.act2.forward(&hidden)?)?;
        xs + hidden
    }
}

/// The codec decoder.
#[derive(Debug)]
pub struct CodecDecoder {
    cfg: CodecConfig,
    device: Device,
    quantizer: Quantizer,
    pre_conv: CausalConv1d,
    input_proj: Linear,
    layers: Vec<TransformerLayer>,
    norm: RmsNorm,
    output_proj: Linear,
    upsample: Vec<(CausalConvTranspose1d, ConvNextBlock)>,
    head_conv: CausalConv1d,
    blocks: Vec<DecoderBlock>,
    tail_act: SnakeBeta,
    tail_conv: CausalConv1d,
}

impl CodecDecoder {
    /// Loads the decoder half of a codec checkpoint.
    ///
    /// The codec always runs in full precision: it is the stage whose output
    /// must reproduce exactly.
    pub fn load(
        cfg: &CodecConfig,
        vb: VarBuilder,
        device: Device,
    ) -> Result<Self> {
        let vb = vb.pp("decoder");

        let quantizer = Quantizer::load(cfg, vb.pp("quantizer"))?;
        let pre_conv = CausalConv1d::load(
            cfg.codebook_dim,
            cfg.latent_dim,
            3,
            1,
            1,
            vb.pp("pre_conv"),
        )?;

        let transformer = vb.pp("pre_transformer");
        let input_proj = Linear::new(
            transformer
                .get((cfg.hidden_size, cfg.latent_dim), "input_proj.weight")?,
            Some(transformer.get(cfg.hidden_size, "input_proj.bias")?),
        );
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            layers.push(TransformerLayer::load(
                cfg,
                transformer.pp("layers").pp(index),
            )?);
        }
        let norm = RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            transformer.pp("norm"),
        )?;
        let output_proj = Linear::new(
            transformer
                .get((cfg.latent_dim, cfg.hidden_size), "output_proj.weight")?,
            Some(transformer.get(cfg.latent_dim, "output_proj.bias")?),
        );

        let mut upsample = Vec::with_capacity(cfg.upsampling_ratios.len());
        for (index, &ratio) in cfg.upsampling_ratios.iter().enumerate() {
            let vb = vb.pp("upsample").pp(index);
            upsample.push((
                CausalConvTranspose1d::load(
                    cfg.latent_dim,
                    cfg.latent_dim,
                    ratio,
                    ratio,
                    vb.pp(0),
                )?,
                ConvNextBlock::load(cfg.latent_dim, vb.pp(1))?,
            ));
        }

        let decoder = vb.pp("decoder");
        let head_conv = CausalConv1d::load(
            cfg.latent_dim,
            cfg.decoder_dim,
            7,
            1,
            1,
            decoder.pp(0),
        )?;
        let mut blocks = Vec::with_capacity(cfg.upsample_rates.len());
        for (index, &rate) in cfg.upsample_rates.iter().enumerate() {
            let in_dim = cfg.decoder_dim >> index;
            let out_dim = cfg.decoder_dim >> (index + 1);
            blocks.push(DecoderBlock::load(
                in_dim,
                out_dim,
                rate,
                decoder.pp(index + 1).pp("block"),
            )?);
        }
        let tail_dim = cfg.decoder_dim >> cfg.upsample_rates.len();
        let tail_index = cfg.upsample_rates.len() + 1;
        let tail_act = SnakeBeta::load(tail_dim, decoder.pp(tail_index))?;
        let tail_conv = CausalConv1d::load(
            tail_dim,
            1,
            7,
            1,
            1,
            decoder.pp(tail_index + 1),
        )?;

        Ok(Self {
            cfg: cfg.clone(),
            device,
            quantizer,
            pre_conv,
            input_proj,
            layers,
            norm,
            output_proj,
            upsample,
            head_conv,
            blocks,
            tail_act,
            tail_conv,
        })
    }

    /// The sample rate of the waveform this decoder produces.
    #[must_use]
    pub fn sample_rate(&self) -> u32 { self.cfg.output_sample_rate }

    /// Decodes `frames` — one row of `num_quantizers` codes per frame — into a
    /// mono waveform.
    ///
    /// Long inputs are split into chunks carrying a left context, exactly as
    /// the reference splits them, so the result does not depend on how many
    /// frames are decoded at once.
    ///
    /// # Errors
    ///
    /// Returns a backend error, or a shape error when a frame does not carry
    /// the expected number of codes.
    pub fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let quantizers = self.cfg.num_quantizers;
        for (index, frame) in frames.iter().enumerate() {
            if frame.len() != quantizers {
                candle_core::bail!(
                    "frame {index} carries {} codes, expected {quantizers}",
                    frame.len()
                );
            }
        }

        // Lay the codes out codebook-major: the quantizer indexes whole rows.
        let total = frames.len();
        let mut flat = vec![0u32; quantizers * total];
        for (t, frame) in frames.iter().enumerate() {
            for (q, &code) in frame.iter().enumerate() {
                flat[q * total + t] = code;
            }
        }
        let codes = Tensor::from_vec(flat, (quantizers, total), &self.device)?;

        let upsample = self.cfg.decode_upsample_rate;
        let mut wave: Vec<f32> = Vec::with_capacity(total * upsample);
        for chunk in chunk_plan(total) {
            let codes = codes.narrow(1, chunk.context_start(), chunk.span())?;
            let mut decoded = self.forward(&codes)?;
            // Drop the samples the context produced; they only prime the
            // convolutions.
            let decoded = decoded.split_off(chunk.context * upsample);
            wave.extend_from_slice(&decoded);
        }
        Ok(wave)
    }

    /// Runs the network over one chunk of codes, shaped `[codebooks, frames]`.
    fn forward(&self, codes: &Tensor) -> Result<Vec<f32>> {
        let frames = codes.dim(1)?;

        let hidden = self.quantizer.decode(codes)?.unsqueeze(0)?;
        let hidden = self.pre_conv.forward(&hidden)?.transpose(1, 2)?;

        // The attention stack sees a causal window, so a frame never depends
        // on frames after it.
        let (cos, sin) = self.rope(frames)?;
        let mask =
            sliding_window_mask(frames, self.cfg.sliding_window, &self.device)?;
        let mut hidden = self.input_proj.forward(&hidden)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let hidden = self.output_proj.forward(&hidden)?.transpose(1, 2)?;

        let mut hidden = hidden.contiguous()?;
        for (upsample, convnext) in &self.upsample {
            hidden = convnext.forward(&upsample.forward(&hidden)?)?;
        }

        let mut wave = self.head_conv.forward(&hidden)?;
        for block in &self.blocks {
            wave = block.forward(&wave)?;
        }
        let wave = self.tail_conv.forward(&self.tail_act.forward(&wave)?)?;
        wave.clamp(-1f32, 1f32)?.flatten_all()?.to_vec1::<f32>()
    }

    /// Builds the rotary tables for `frames` positions, laid out to match the
    /// halves `rotate_half` produces.
    fn rope(&self, frames: usize) -> Result<(Tensor, Tensor)> {
        let (cos, sin) = rope_tables(
            self.cfg.head_dim,
            frames,
            self.cfg.rope_theta,
            &self.device,
        )?;
        // The reference duplicates the table across both halves of the head.
        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?;
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;
        Ok((
            cos.unsqueeze(0)?.unsqueeze(0)?,
            sin.unsqueeze(0)?.unsqueeze(0)?,
        ))
    }
}

/// Builds the additive attention mask: a frame may attend to itself and to the
/// `window - 1` frames before it, and to nothing after it.
fn sliding_window_mask(
    frames: usize,
    window: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut mask = vec![0f32; frames * frames];
    for query in 0..frames {
        for key in 0..frames {
            if !window_visible(query, key, window) {
                mask[query * frames + key] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, frames, frames), device)
}

#[cfg(test)]
mod tests;
