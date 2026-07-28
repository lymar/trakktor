//! The DiT: the network that turns noise into a mel spectrogram.
//!
//! One forward pass takes the current state, the conditioning, the encoded
//! text, and a timestep, and predicts where the state should move. The timestep
//! does not enter as a token: it is embedded once per step and then *modulates*
//! every block — each one scales and shifts its normalized input and gates its
//! two branches by vectors derived from it. That is what makes the network a
//! diffusion transformer rather than a plain one.
//!
//! Two things are hoisted out of the per-step work, because they do not change
//! while the solver runs: the text encoder (four convolutional blocks over the
//! whole sequence) and the part of the input projection that only reads the
//! conditioning and the text. Both are computed once per utterance, for each
//! guidance branch, and per step only the state's own projection is left.
//!
//! Ported from F5-TTS (MIT).

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{Conv1d, Embedding, Linear, Module, VarBuilder, ops};

use super::layers::{ConvNeXtBlock, PlainNorm, grouped, mish};
use crate::tts::espeech::config::{
    DIM_HEAD, DitConfig, ROPE_THETA, TIME_SCALE,
};

/// Positions the sinusoidal text embedding is precomputed for. The reference
/// tabulates the same span — about 87 seconds of audio, well past any single
/// utterance.
const MAX_TEXT_POS: usize = 8192;

/// Epsilon of every normalization in the network.
const NORM_EPS: f64 = 1e-6;

/// The network, loaded.
pub struct Dit {
    cfg: DitConfig,
    dtype: DType,
    time_mlp: (Linear, Linear),
    text_embed: Embedding,
    text_blocks: Vec<ConvNeXtBlock>,
    /// Sinusoidal positions added to the text embedding, `[MAX_TEXT_POS,
    /// text_dim]`.
    text_pos: Tensor,
    /// The input projection, split by what it reads: the state's columns, and
    /// the conditioning's and text's columns folded into one matrix.
    proj_state: Tensor,
    proj_context: Tensor,
    proj_bias: Tensor,
    conv_pos: (Conv1d, Conv1d),
    blocks: Vec<DitBlock>,
    /// The affine-free norm before the final modulation.
    norm: PlainNorm,
    norm_out: Linear,
    proj_out: Linear,
    /// Rotary tables, `[MAX_TEXT_POS, DIM_HEAD / 2]` each.
    rope: (Tensor, Tensor),
}

impl Dit {
    /// Loads the network with the geometry derived from the checkpoint.
    pub fn load(
        cfg: DitConfig,
        dtype: DType,
        vb: VarBuilder,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let vb = vb.pp("transformer");
        let time = vb.pp("time_embed").pp("time_mlp");
        let time_mlp = (
            candle_nn::linear(cfg.time_dim, cfg.dim, time.pp("0"))?,
            candle_nn::linear(cfg.dim, cfg.dim, time.pp("2"))?,
        );

        let text = vb.pp("text_embed");
        let text_embed = candle_nn::embedding(
            cfg.vocab_size + 1,
            cfg.text_dim,
            text.pp("text_embed"),
        )?;
        let text_blocks = (0..cfg.text_conv_layers)
            .map(|index| {
                ConvNeXtBlock::load_v2(
                    cfg.text_dim,
                    cfg.text_ff_inner,
                    text.pp("text_blocks").pp(index.to_string()),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let text_pos =
            sinusoidal_positions(cfg.text_dim, MAX_TEXT_POS, device)?
                .to_dtype(dtype)?;

        let input = vb.pp("input_embed");
        // [dim, mel * 2 + text_dim]: the state, the conditioning, and the text
        // side by side. Splitting the matrix is what lets the context half be
        // projected once per utterance instead of once per step.
        let proj = input.pp("proj");
        let width = cfg.mel_channels * 2 + cfg.text_dim;
        let weight = proj.get((cfg.dim, width), "weight")?;
        let proj_state =
            weight.narrow(1, 0, cfg.mel_channels)?.t()?.contiguous()?;
        let proj_context = weight
            .narrow(1, cfg.mel_channels, cfg.mel_channels + cfg.text_dim)?
            .t()?
            .contiguous()?;
        let proj_bias = proj.get(cfg.dim, "bias")?;

        let conv = input.pp("conv_pos_embed").pp("conv1d");
        let conv_pos = (
            grouped(
                cfg.dim,
                cfg.conv_pos_kernel,
                cfg.conv_pos_groups,
                conv.pp("0"),
            )?,
            grouped(
                cfg.dim,
                cfg.conv_pos_kernel,
                cfg.conv_pos_groups,
                conv.pp("2"),
            )?,
        );

        let blocks = (0..cfg.depth)
            .map(|index| {
                DitBlock::load(
                    &cfg,
                    vb.pp("transformer_blocks").pp(index.to_string()),
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let norm_out = candle_nn::linear(
            cfg.dim,
            cfg.dim * 2,
            vb.pp("norm_out").pp("linear"),
        )?;
        let proj_out =
            candle_nn::linear(cfg.dim, cfg.mel_channels, vb.pp("proj_out"))?;

        let rope = rope_tables(DIM_HEAD, MAX_TEXT_POS, ROPE_THETA, device)?;
        let rope = (rope.0.to_dtype(dtype)?, rope.1.to_dtype(dtype)?);

        Ok(Self {
            cfg,
            dtype,
            time_mlp,
            text_embed,
            text_blocks,
            text_pos,
            proj_state,
            proj_context,
            proj_bias,
            conv_pos,
            blocks,
            norm: PlainNorm::new(cfg.dim, NORM_EPS, device)?,
            norm_out,
            proj_out,
            rope,
        })
    }

    /// The geometry this was loaded with.
    pub fn config(&self) -> &DitConfig { &self.cfg }

    /// The tensor type the network computes in.
    pub fn dtype(&self) -> DType { self.dtype }

    /// Encodes the text for one guidance branch.
    ///
    /// `ids` are the character ids, already shifted so that zero is the filler
    /// token, padded or cut to `frames`. `drop` replaces every id with the
    /// filler — the unconditional branch — while keeping the padding mask of
    /// the real text, exactly as the reference does.
    pub fn encode_text(
        &self,
        ids: &Tensor,
        keep: &Tensor,
        drop: bool,
    ) -> Result<Tensor> {
        let frames = ids.dim(1)?;
        let ids = if drop { ids.zeros_like()? } else { ids.clone() };
        let mut hidden = self.text_embed.forward(&ids)?;
        hidden = hidden.broadcast_add(
            &self.text_pos.narrow(0, 0, frames)?.unsqueeze(0)?,
        )?;
        // Positions past the end of the text are held at zero, before the
        // blocks and after every one of them.
        hidden = hidden.broadcast_mul(keep)?;
        for block in &self.text_blocks {
            hidden = block.forward(&hidden)?.broadcast_mul(keep)?;
        }
        Ok(hidden)
    }

    /// Projects everything that does not change between solver steps: the
    /// conditioning and the encoded text, plus the projection's bias.
    ///
    /// `cond` is `[branches, frames, mel]` (zero in the unconditional branch)
    /// and `text` `[branches, frames, text_dim]`.
    pub fn project_context(
        &self,
        cond: &Tensor,
        text: &Tensor,
    ) -> Result<Tensor> {
        let context = Tensor::cat(&[cond, text], D::Minus1)?;
        context
            .broadcast_matmul(&self.proj_context)?
            .broadcast_add(&self.proj_bias)
    }

    /// Predicts the flow at time `t` for state `x`.
    ///
    /// `context` is what [`project_context`](Self::project_context) returned,
    /// `[branches, frames, dim]`; `x` is `[1, frames, mel]` and is shared by
    /// the branches. The result is `[branches, frames, mel]`.
    pub fn forward(
        &self,
        x: &Tensor,
        context: &Tensor,
        t: f32,
    ) -> Result<Tensor> {
        let branches = context.dim(0)?;
        let time = self.embed_time(t, branches)?;

        let mut hidden =
            context.broadcast_add(&x.broadcast_matmul(&self.proj_state)?)?;
        hidden = (self.position_embed(&hidden)? + hidden)?;

        for block in &self.blocks {
            hidden = block.forward(&hidden, &time, &self.rope)?;
        }

        // The final modulation states the scale before the shift — the blocks
        // state the shift first.
        let modulation = self.norm_out.forward(&time.silu()?)?;
        let scale = modulation.narrow(1, 0, self.cfg.dim)?.unsqueeze(1)?;
        let shift = modulation
            .narrow(1, self.cfg.dim, self.cfg.dim)?
            .unsqueeze(1)?;
        let normed = self
            .norm
            .forward(&hidden)?
            .broadcast_mul(&(scale + 1.0)?)?
            .broadcast_add(&shift)?;
        self.proj_out.forward(&normed)
    }

    /// The convolutional position embedding: two grouped convolutions over
    /// time, which is what gives the network its sense of order beyond the
    /// rotary tables.
    fn position_embed(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = xs.transpose(1, 2)?.contiguous()?;
        let hidden = mish(&self.conv_pos.0.forward(&hidden)?)?;
        let hidden = mish(&self.conv_pos.1.forward(&hidden)?)?;
        hidden.transpose(1, 2)?.contiguous()
    }

    /// Embeds the timestep and repeats it for every guidance branch.
    fn embed_time(&self, t: f32, branches: usize) -> Result<Tensor> {
        let half = self.cfg.time_dim / 2;
        let device = self.proj_bias.device();
        let step = (10_000f64).ln() / (half - 1) as f64;
        let scaled: Vec<f32> = (0..half)
            .map(|index| {
                (TIME_SCALE * f64::from(t) * (-step * index as f64).exp())
                    as f32
            })
            .collect();
        let angles = Tensor::from_vec(scaled, (1, half), device)?;
        let embedded = Tensor::cat(&[angles.sin()?, angles.cos()?], D::Minus1)?
            .to_dtype(self.dtype)?;
        let hidden = self.time_mlp.0.forward(&embedded)?;
        let hidden = self.time_mlp.1.forward(&hidden.silu()?)?;
        if branches == 1 {
            Ok(hidden)
        } else {
            Tensor::cat(&vec![&hidden; branches], 0)
        }
    }
}

/// One block: modulated self-attention, then a modulated feed-forward.
struct DitBlock {
    modulation: Linear,
    /// The affine-free norm both halves of the block open with.
    norm: PlainNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    ff_in: Linear,
    ff_out: Linear,
    heads: usize,
    dim: usize,
}

impl DitBlock {
    fn load(
        cfg: &DitConfig,
        vb: VarBuilder,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let attn = vb.pp("attn");
        let ff = vb.pp("ff").pp("ff");
        Ok(Self {
            norm: PlainNorm::new(cfg.dim, NORM_EPS, device)?,
            modulation: candle_nn::linear(
                cfg.dim,
                cfg.dim * 6,
                vb.pp("attn_norm").pp("linear"),
            )?,
            // Kept as three matmuls rather than one stacked matrix: fusing them
            // is the usual win, and here it measured *slower* (1:16–1:22
            // against 1:09 on Metal), because the fused output then has to be
            // sliced three ways and each slice copied. The launch overhead this
            // would save is not what this network is spending its time on.
            to_q: candle_nn::linear(cfg.dim, cfg.dim, attn.pp("to_q"))?,
            to_k: candle_nn::linear(cfg.dim, cfg.dim, attn.pp("to_k"))?,
            to_v: candle_nn::linear(cfg.dim, cfg.dim, attn.pp("to_v"))?,
            to_out: candle_nn::linear(
                cfg.dim,
                cfg.dim,
                attn.pp("to_out").pp("0"),
            )?,
            ff_in: candle_nn::linear(
                cfg.dim,
                cfg.ff_inner,
                ff.pp("0").pp("0"),
            )?,
            ff_out: candle_nn::linear(cfg.ff_inner, cfg.dim, ff.pp("2"))?,
            heads: cfg.heads,
            dim: cfg.dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        time: &Tensor,
        rope: &(Tensor, Tensor),
    ) -> Result<Tensor> {
        let modulation = self.modulation.forward(&time.silu()?)?;
        let part = |index: usize| -> Result<Tensor> {
            modulation
                .narrow(1, index * self.dim, self.dim)?
                .unsqueeze(1)
        };
        let (shift_attn, scale_attn, gate_attn) =
            (part(0)?, part(1)?, part(2)?);
        let (shift_ff, scale_ff, gate_ff) = (part(3)?, part(4)?, part(5)?);

        let normed = self
            .norm
            .forward(xs)?
            .broadcast_mul(&(scale_attn + 1.0)?)?
            .broadcast_add(&shift_attn)?;
        let attended = self.attend(&normed, rope)?;
        let xs = (xs + attended.broadcast_mul(&gate_attn)?)?;

        let normed = self
            .norm
            .forward(&xs)?
            .broadcast_mul(&(scale_ff + 1.0)?)?
            .broadcast_add(&shift_ff)?;
        // The feed-forward is the one place the reference asks for the `tanh`
        // approximation of GELU.
        let hidden = self.ff_in.forward(&normed)?.gelu()?;
        let hidden = self.ff_out.forward(&hidden)?;
        xs + hidden.broadcast_mul(&gate_ff)?
    }

    /// Bidirectional self-attention with rotary positions on both queries and
    /// keys. There is no mask: a single utterance is one unpadded sequence.
    fn attend(&self, xs: &Tensor, rope: &(Tensor, Tensor)) -> Result<Tensor> {
        let (batch, frames, _) = xs.dims3()?;
        let split = |projected: Tensor| -> Result<Tensor> {
            projected
                .reshape((batch, frames, self.heads, DIM_HEAD))?
                .transpose(1, 2)?
                .contiguous()
        };
        let query = split(self.to_q.forward(xs)?)?;
        let key = split(self.to_k.forward(xs)?)?;
        let value = split(self.to_v.forward(xs)?)?;

        let cos = rope.0.narrow(0, 0, frames)?;
        let sin = rope.1.narrow(0, 0, frames)?;
        // The rotation pairs neighbouring channels, not the two halves of the
        // head — the reference's rotary layout.
        let query = candle_nn::rotary_emb::rope_i(&query, &cos, &sin)?;
        let key = candle_nn::rotary_emb::rope_i(&key, &cos, &sin)?;

        let scale = 1.0 / (DIM_HEAD as f64).sqrt();
        // On Metal the fused kernel is not a nicety: written out, the scores of
        // one layer are `heads × frames²` values — hundreds of megabytes for an
        // ordinary utterance, written and read back once per layer per step.
        // The fused path never materializes them. Everywhere else there is no
        // such kernel, and the plain form is what the parity tests run.
        let context = if xs.device().is_metal() {
            // A softcap of one is how that kernel spells "no softcapping".
            ops::sdpa(&query, &key, &value, None, false, scale as f32, 1.0)?
        } else {
            let scores =
                (query.matmul(&key.transpose(D::Minus2, D::Minus1)?)? * scale)?;
            ops::softmax_last_dim(&scores)?.matmul(&value)?
        };
        let context = context.transpose(1, 2)?.reshape((
            batch,
            frames,
            self.heads * DIM_HEAD,
        ))?;
        self.to_out.forward(&context)
    }
}

/// The sinusoidal positions the text embedding is offset by: cosines and sines
/// of geometrically spaced frequencies, the halves laid out one after the
/// other.
fn sinusoidal_positions(
    dim: usize,
    positions: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let half = dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|index| {
            (1.0 / ROPE_THETA.powf(2.0 * index as f64 / dim as f64)) as f32
        })
        .collect();
    let inv = Tensor::from_vec(inv, (1, half), device)?;
    let steps: Vec<f32> = (0..positions).map(|step| step as f32).collect();
    let steps = Tensor::from_vec(steps, (positions, 1), device)?;
    let angles = steps.broadcast_mul(&inv)?;
    Tensor::cat(&[angles.cos()?, angles.sin()?], D::Minus1)
}

/// Cosine and sine tables for the rotary embedding, one row per position and
/// one column per channel pair.
fn rope_tables(
    dim: usize,
    positions: usize,
    theta: f64,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let inv: Vec<f32> = (0..dim / 2)
        .map(|index| (1.0 / theta.powf(2.0 * index as f64 / dim as f64)) as f32)
        .collect();
    let inv = Tensor::from_vec(inv, (1, dim / 2), device)?;
    let steps: Vec<f32> = (0..positions).map(|step| step as f32).collect();
    let steps = Tensor::from_vec(steps, (positions, 1), device)?;
    let angles = steps.broadcast_mul(&inv)?;
    Ok((angles.cos()?, angles.sin()?))
}
