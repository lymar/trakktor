//! The vision tower: a SigLIP-shaped encoder that accepts any patch grid.
//!
//! Three things make it different from a fixed-resolution ViT, and all three
//! come from the same requirement — a text block is whatever shape it is.
//!
//! **Positions are learned for one square and stretched onto the actual grid.**
//! The checkpoint carries a 27×27 grid of position vectors (the 384-pixel
//! square it was trained at); for a block of, say, 35×71 patches they are
//! interpolated bilinearly onto that shape. Two published implementations do
//! this interpolation *differently* — the authors' own code with
//! `align_corners=False`, the `transformers` port on a `linspace(0, side−1, n)`
//! grid, i.e. `align_corners=True`. This port follows `transformers`: that is
//! the reference that runs on the target machine and the one every quality
//! measurement was taken against.
//!
//! **On top of that sits a two-dimensional rotary embedding**: half the head
//! dimension rotates with the patch's row, half with its column.
//!
//! **Attention is dense but chunked.** A block of a few thousand patches would
//! need hundreds of megabytes for one `[heads, seq, seq]` score matrix in
//! `f32`, so the queries are walked in slices. The arithmetic is unchanged —
//! softmax is per row — only the peak allocation shrinks.

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, ops::softmax_last_dim};

use crate::ocr::vl::{config::VisionConfig, tables};

/// How many query positions are scored against the keys at a time.
const ATTENTION_CHUNK: usize = 1024;

/// Layer normalization with a learned gain and offset.
fn layer_norm(
    size: usize,
    eps: f64,
    vb: VarBuilder,
) -> Result<candle_nn::LayerNorm> {
    candle_nn::layer_norm(
        size,
        candle_nn::LayerNormConfig {
            eps,
            remove_mean: true,
            affine: true,
        },
        vb,
    )
}

/// A linear layer with a bias, which every layer of this tower has.
fn linear(rows: usize, cols: usize, vb: VarBuilder) -> Result<Linear> {
    Ok(Linear::new(
        vb.get((rows, cols), "weight")?,
        Some(vb.get(rows, "bias")?),
    ))
}

/// The patch embedding and the learned position grid.
struct Embeddings {
    /// The patch projection, held as a matrix: the convolution has kernel and
    /// stride both equal to the patch size, so it is a linear map of one
    /// patch's pixels and is applied as one.
    projection: Tensor,
    bias: Tensor,
    /// The learned position grid, on the host: it is interpolated per input
    /// shape, which is a scattered gather better done outside the tensor
    /// library.
    positions: Vec<f32>,
    /// Side of that grid.
    side: usize,
    dim: usize,
}

impl Embeddings {
    fn load(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch = cfg.patch_size;
        let weight = vb.get(
            (cfg.hidden_size, cfg.num_channels, patch, patch),
            "patch_embedding.weight",
        )?;
        let side = cfg.image_size / patch;
        let positions = vb
            .get((side * side, cfg.hidden_size), "position_embedding.weight")?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(Self {
            projection: weight
                .reshape((cfg.hidden_size, cfg.num_channels * patch * patch))?
                .t()?
                .contiguous()?,
            bias: vb.get(cfg.hidden_size, "patch_embedding.bias")?,
            positions,
            side,
            dim: cfg.hidden_size,
        })
    }

    /// Projects the patches and adds their positions.
    ///
    /// `pixels` is `[patches, channels × patch × patch]`.
    fn forward(
        &self,
        pixels: &Tensor,
        grid: (usize, usize, usize),
    ) -> Result<Tensor> {
        let embedded =
            pixels.matmul(&self.projection)?.broadcast_add(&self.bias)?;
        let (_, height, width) = grid;
        let positions = Tensor::from_vec(
            tables::interpolate_positions(
                &self.positions,
                self.side,
                self.dim,
                height,
                width,
            ),
            (height * width, self.dim),
            embedded.device(),
        )?
        .to_dtype(embedded.dtype())?;
        embedded.broadcast_add(&positions)
    }
}

/// Multi-head self-attention over the patches.
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn load(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let size = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        Ok(Self {
            q_proj: linear(size, size, vb.pp("q_proj"))?,
            k_proj: linear(size, size, vb.pp("k_proj"))?,
            v_proj: linear(size, size, vb.pp("v_proj"))?,
            out_proj: linear(size, size, vb.pp("out_proj"))?,
            heads: cfg.num_attention_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// `xs` is `[seq, hidden]`; `cos` and `sin` are `[seq, head_dim / 2]`,
    /// in `f32`.
    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (seq, hidden) = xs.dims2()?;
        let split = |projected: Tensor| -> Result<Tensor> {
            projected
                .reshape((seq, self.heads, self.head_dim))?
                .transpose(0, 1)?
                .contiguous()
        };
        // The rotation is applied in full precision, as the reference does:
        // the angles are the same for every layer, and rounding them once per
        // layer in half precision drifts.
        let rope = |xs: &Tensor| -> Result<Tensor> {
            candle_nn::rotary_emb::rope(
                &xs.unsqueeze(0)?.to_dtype(DType::F32)?,
                cos,
                sin,
            )?
            .to_dtype(xs.dtype())?
            .squeeze(0)
        };

        let query = rope(&split(self.q_proj.forward(xs)?)?)?;
        let key = rope(&split(self.k_proj.forward(xs)?)?)?;
        let value = split(self.v_proj.forward(xs)?)?;
        // A transposed view is a layout the matrix product reads directly;
        // nothing to copy.
        let keys = key.transpose(1, 2)?;

        let mut attended = Vec::with_capacity(seq.div_ceil(ATTENTION_CHUNK));
        let mut at = 0;
        while at < seq {
            let len = ATTENTION_CHUNK.min(seq - at);
            let scores =
                (query.narrow(1, at, len)?.matmul(&keys)? * self.scale)?;
            let weights = softmax_last_dim(&scores.to_dtype(DType::F32)?)?
                .to_dtype(value.dtype())?;
            attended.push(weights.matmul(&value)?);
            at += len;
        }

        let attended = if attended.len() == 1 {
            attended.remove(0)
        } else {
            Tensor::cat(&attended, 1)?
        };
        self.out_proj.forward(
            &attended
                .transpose(0, 1)?
                .contiguous()?
                .reshape((seq, hidden))?,
        )
    }
}

/// One encoder layer: pre-normalized attention, then a pre-normalized
/// feed-forward, each added back onto the residual stream.
struct EncoderLayer {
    layer_norm1: candle_nn::LayerNorm,
    attention: Attention,
    layer_norm2: candle_nn::LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl EncoderLayer {
    fn load(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let eps = cfg.layer_norm_eps;
        Ok(Self {
            layer_norm1: layer_norm(
                cfg.hidden_size,
                eps,
                vb.pp("layer_norm1"),
            )?,
            attention: Attention::load(cfg, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(
                cfg.hidden_size,
                eps,
                vb.pp("layer_norm2"),
            )?,
            fc1: linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("mlp.fc1"),
            )?,
            fc2: linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp.fc2"),
            )?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let normed = self.layer_norm1.forward(xs)?;
        let xs = (xs + self.attention.forward(&normed, cos, sin)?)?;
        let normed = self.layer_norm2.forward(&xs)?;
        // `gelu_pytorch_tanh` — the tanh approximation, not the exact one.
        let hidden = self.fc1.forward(&normed)?.gelu()?;
        xs + self.fc2.forward(&hidden)?
    }
}

/// The tower.
pub struct Tower {
    embeddings: Embeddings,
    layers: Vec<EncoderLayer>,
    post_layernorm: candle_nn::LayerNorm,
    head_dim: usize,
}

impl Tower {
    pub fn load(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("vision_model");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::load(
                cfg,
                vb.pp(format!("encoder.layers.{index}")),
            )?);
        }
        Ok(Self {
            embeddings: Embeddings::load(cfg, vb.pp("embeddings"))?,
            layers,
            post_layernorm: layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("post_layernorm"),
            )?,
            head_dim: cfg.head_dim(),
        })
    }

    /// Lifts the two-dimensional rotary tables for a patch grid onto the
    /// device, `[patches, head_dim / 2]` each.
    fn rotary(
        &self,
        grid: (usize, usize, usize),
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let (_, height, width) = grid;
        let angles = Tensor::from_vec(
            tables::tower_angles(grid, self.head_dim),
            (height * width, self.head_dim / 2),
            device,
        )?;
        Ok((angles.cos()?, angles.sin()?))
    }

    /// Encodes one picture's patches.
    ///
    /// `pixels` is `[patches, channels × patch × patch]`, `grid` the shape
    /// those patches were cut from. The result is `[patches, hidden]`.
    pub fn forward(
        &self,
        pixels: &Tensor,
        grid: (usize, usize, usize),
    ) -> Result<Tensor> {
        let (cos, sin) = self.rotary(grid, pixels.device())?;
        let mut hidden = self.embeddings.forward(pixels, grid)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin)?;
        }
        self.post_layernorm.forward(&hidden)
    }
}

/// The projector: normalize, fold `merge × merge` patches into one token, and
/// map that token into the decoder's width.
pub struct Projector {
    pre_norm: candle_nn::LayerNorm,
    linear_1: Linear,
    linear_2: Linear,
    merge: usize,
}

impl Projector {
    pub fn load(
        vision: &VisionConfig,
        hidden_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let merged = vision.hidden_size * vision.spatial_merge_size.pow(2);
        Ok(Self {
            // The projector's normalization keeps the decoder's epsilon, not
            // the tower's — it belongs to the join, not to either side.
            pre_norm: layer_norm(vision.hidden_size, 1e-5, vb.pp("pre_norm"))?,
            linear_1: linear(merged, merged, vb.pp("linear_1"))?,
            linear_2: linear(hidden_size, merged, vb.pp("linear_2"))?,
            merge: vision.spatial_merge_size,
        })
    }

    /// `features` is `[patches, vision hidden]`; the result is
    /// `[patches / merge², decoder hidden]`.
    pub fn forward(
        &self,
        features: &Tensor,
        grid: (usize, usize, usize),
    ) -> Result<Tensor> {
        let (t, height, width) = grid;
        let merge = self.merge;
        let dim = features.dim(D::Minus1)?;
        let folded = self
            .pre_norm
            .forward(features)?
            .reshape((t, height / merge, merge, width / merge, merge, dim))?
            .permute((0, 1, 3, 2, 4, 5))?
            .contiguous()?
            .reshape((
                t * (height / merge) * (width / merge),
                merge * merge * dim,
            ))?;
        let hidden = self.linear_1.forward(&folded)?.gelu_erf()?;
        self.linear_2.forward(&hidden)
    }
}
