//! The vision tower and the projector on burn.
//!
//! A port of the candle network in
//! [`runtime::vision`](super::super::runtime::vision) with the same numerical
//! semantics, written idiomatically for burn rather than as a mirror. What
//! differs, and why:
//!
//! - **Attention runs through burn's fused scaled-dot-product `attention`**,
//!   dense and unmasked, called on queries walked in slices exactly like the
//!   candle tower: the arithmetic is unchanged, only the peak allocation of the
//!   fallback path is bounded. The call is made **in f32** whatever the backend
//!   element is — the fused kernel's half-precision path has bitten an earlier
//!   port, and the rotation feeding it is applied in full precision anyway, as
//!   the reference does.
//! - **Layer normalization is spelled out** — mean subtracted first, then the
//!   variance of the centered values, statistics in f32 — because that is the
//!   numerically stable form the candle port insists on for this tower's large
//!   activations.
//! - The feed-forward's `gelu_pytorch_tanh` maps to burn's
//!   [`gelu_approximate`](activation::gelu_approximate); the projector's exact
//!   GELU maps to [`gelu`](activation::gelu).

use burn::tensor::{
    DType, Tensor, TensorData, activation,
    backend::Backend,
    module::{attention, linear},
    ops::AttentionModuleOptions,
};

use super::{Weights, apply_rope, host_tables, matrix, weight};
use crate::ocr::{
    error::OcrError,
    vl::{config::VisionConfig, tables},
};

/// How many query positions are scored against the keys at a time.
const ATTENTION_CHUNK: usize = 1024;

/// Layer normalization with a learned gain and offset.
///
/// The mean is removed before the variance is taken — the stable form, not
/// `E[x²] − m²` — and the statistics are held in f32; the gain and offset are
/// applied in the compute dtype, as candle's module does.
struct LayerNorm<B: Backend> {
    gamma: Tensor<B, 1>,
    beta: Tensor<B, 1>,
    size: usize,
    eps: f64,
}

impl<B: Backend> LayerNorm<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        size: usize,
        eps: f64,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            gamma: weight(
                weights,
                device,
                &format!("{prefix}.weight"),
                [size],
            )?,
            beta: weight(weights, device, &format!("{prefix}.bias"), [size])?,
            size,
            eps,
        })
    }

    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dtype = x.dtype();
        let x = x.cast(DType::F32);
        let centered = x.clone() - x.mean_dim(D - 1);
        let variance = (centered.clone() * centered.clone()).mean_dim(D - 1);
        let normed = centered / variance.add_scalar(self.eps).sqrt();
        let mut shape = [1usize; D];
        shape[D - 1] = self.size;
        normed.cast(dtype) * self.gamma.clone().reshape(shape) +
            self.beta.clone().reshape(shape)
    }
}

/// A linear layer's weight and bias, in burn's `[in, out]` layout — every
/// layer of this tower has a bias.
struct Linear<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Tensor<B, 1>,
}

impl<B: Backend> Linear<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            weight: matrix(
                weights,
                device,
                &format!("{prefix}.weight"),
                out_dim,
                in_dim,
            )?,
            bias: weight(
                weights,
                device,
                &format!("{prefix}.bias"),
                [out_dim],
            )?,
        })
    }

    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        linear(x, self.weight.clone(), Some(self.bias.clone()))
    }
}

/// Multi-head self-attention over the patches.
struct Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    out_proj: Linear<B>,
    heads: usize,
    head_dim: usize,
}

impl<B: Backend> Attention<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        cfg: &VisionConfig,
    ) -> Result<Self, OcrError> {
        let size = cfg.hidden_size;
        let load = |name: &str| {
            Linear::load(
                weights,
                device,
                &format!("{prefix}.{name}"),
                size,
                size,
            )
        };
        Ok(Self {
            q_proj: load("q_proj")?,
            k_proj: load("k_proj")?,
            v_proj: load("v_proj")?,
            out_proj: load("out_proj")?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// `x` is `[seq, hidden]`; `cos` and `sin` are `[1, 1, seq, head_dim]`,
    /// in f32.
    fn forward(
        &self,
        x: Tensor<B, 2>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        let [seq, hidden] = x.dims();
        let dtype = x.dtype();
        let split = |projected: Tensor<B, 2>| -> Tensor<B, 4> {
            projected
                .reshape([1, seq, self.heads, self.head_dim])
                .swap_dims(1, 2)
        };
        // The rotation is applied in full precision, as the reference does:
        // the angles are the same for every layer, and rounding them once per
        // layer in half precision drifts. The fused attention call stays in
        // f32 with it.
        let rope = |projected: Tensor<B, 4>| -> Tensor<B, 4> {
            apply_rope(projected.cast(DType::F32), cos, sin)
        };

        let query = rope(split(self.q_proj.forward(x.clone())));
        let key = rope(split(self.k_proj.forward(x.clone())));
        let value = split(self.v_proj.forward(x)).cast(DType::F32);

        let mut attended = Vec::with_capacity(seq.div_ceil(ATTENTION_CHUNK));
        let mut at = 0;
        while at < seq {
            let len = ATTENTION_CHUNK.min(seq - at);
            attended.push(attention(
                query.clone().narrow(2, at, len),
                key.clone(),
                value.clone(),
                None,
                None,
                AttentionModuleOptions::default(),
            ));
            at += len;
        }
        let attended = if attended.len() == 1 {
            attended.remove(0)
        } else {
            Tensor::cat(attended, 2)
        };

        self.out_proj.forward(
            attended.cast(dtype).swap_dims(1, 2).reshape([seq, hidden]),
        )
    }
}

/// One encoder layer: pre-normalized attention, then a pre-normalized
/// feed-forward, each added back onto the residual stream.
struct EncoderLayer<B: Backend> {
    layer_norm1: LayerNorm<B>,
    attention: Attention<B>,
    layer_norm2: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> EncoderLayer<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        cfg: &VisionConfig,
    ) -> Result<Self, OcrError> {
        let eps = cfg.layer_norm_eps;
        Ok(Self {
            layer_norm1: LayerNorm::load(
                weights,
                device,
                &format!("{prefix}.layer_norm1"),
                cfg.hidden_size,
                eps,
            )?,
            attention: Attention::load(
                weights,
                device,
                &format!("{prefix}.self_attn"),
                cfg,
            )?,
            layer_norm2: LayerNorm::load(
                weights,
                device,
                &format!("{prefix}.layer_norm2"),
                cfg.hidden_size,
                eps,
            )?,
            fc1: Linear::load(
                weights,
                device,
                &format!("{prefix}.mlp.fc1"),
                cfg.intermediate_size,
                cfg.hidden_size,
            )?,
            fc2: Linear::load(
                weights,
                device,
                &format!("{prefix}.mlp.fc2"),
                cfg.hidden_size,
                cfg.intermediate_size,
            )?,
        })
    }

    fn forward(
        &self,
        x: Tensor<B, 2>,
        cos: &Tensor<B, 4>,
        sin: &Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        let normed = self.layer_norm1.forward(x.clone());
        let x = x + self.attention.forward(normed, cos, sin);
        let normed = self.layer_norm2.forward(x.clone());
        // `gelu_pytorch_tanh` — the tanh approximation, not the exact one.
        let hidden = activation::gelu_approximate(self.fc1.forward(normed));
        x + self.fc2.forward(hidden)
    }
}

/// The tower.
pub struct Tower<B: Backend> {
    /// The patch projection, held as a `[channels × patch², hidden]` matrix:
    /// the convolution has kernel and stride both equal to the patch size, so
    /// it is a linear map of one patch's pixels and is applied as one.
    projection: Tensor<B, 2>,
    bias: Tensor<B, 1>,
    /// The learned position grid, on the host: it is interpolated per input
    /// shape, which is a scattered gather better done outside the tensor
    /// library.
    positions: Vec<f32>,
    /// Side of that grid.
    side: usize,
    dim: usize,
    layers: Vec<EncoderLayer<B>>,
    post_layernorm: LayerNorm<B>,
    head_dim: usize,
}

impl<B: Backend> Tower<B> {
    pub(super) fn load(
        weights: &Weights,
        device: &B::Device,
        cfg: &VisionConfig,
    ) -> Result<Self, OcrError> {
        const PREFIX: &str = "visual.vision_model";
        let patch = cfg.patch_size;
        let pixels = cfg.num_channels * patch * patch;
        let side = cfg.image_size / patch;

        let projection = {
            let (values, dims) = weights.parts(&format!(
                "{PREFIX}.embeddings.patch_embedding.weight"
            ))?;
            let shape = [cfg.hidden_size, cfg.num_channels, patch, patch];
            if dims != shape {
                return Err(super::model_err(
                    "patch_embedding.weight",
                    format!("shape {dims:?}, expected {shape:?}"),
                ));
            }
            // `[hidden, pixels]` row-major, transposed into burn's layout.
            Tensor::from_data(
                TensorData::new(
                    super::transpose_2d(&values, cfg.hidden_size, pixels),
                    [pixels, cfg.hidden_size],
                ),
                device,
            )
        };
        let (positions, dims) = weights
            .parts(&format!("{PREFIX}.embeddings.position_embedding.weight"))?;
        if dims != [side * side, cfg.hidden_size] {
            return Err(super::model_err(
                "position_embedding.weight",
                format!("shape {dims:?}, expected a {side}×{side} grid"),
            ));
        }

        let layers = (0..cfg.num_hidden_layers)
            .map(|index| {
                EncoderLayer::load(
                    weights,
                    device,
                    &format!("{PREFIX}.encoder.layers.{index}"),
                    cfg,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            projection,
            bias: weight(
                weights,
                device,
                &format!("{PREFIX}.embeddings.patch_embedding.bias"),
                [cfg.hidden_size],
            )?,
            positions,
            side,
            dim: cfg.hidden_size,
            layers,
            post_layernorm: LayerNorm::load(
                weights,
                device,
                &format!("{PREFIX}.post_layernorm"),
                cfg.hidden_size,
                cfg.layer_norm_eps,
            )?,
            head_dim: cfg.head_dim(),
        })
    }

    /// Encodes one picture's patches.
    ///
    /// `pixels` is `[patches, channels × patch × patch]`, `grid` the shape
    /// those patches were cut from. The result is `[patches, hidden]`.
    pub fn forward(
        &self,
        pixels: Tensor<B, 2>,
        grid: (usize, usize, usize),
    ) -> Tensor<B, 2> {
        let device = pixels.device();
        let patches = pixels.dims()[0];
        let (_, height, width) = grid;

        let embedded =
            linear(pixels, self.projection.clone(), Some(self.bias.clone()));
        let positions = Tensor::<B, 2>::from_data(
            TensorData::new(
                tables::interpolate_positions(
                    &self.positions,
                    self.side,
                    self.dim,
                    height,
                    width,
                ),
                [patches, self.dim],
            ),
            &device,
        );
        let mut hidden = embedded + positions;

        // The rotary tables, in f32 on any backend, as the candle tower
        // holds them.
        let (cos, sin) = host_tables(
            &tables::tower_angles(grid, self.head_dim),
            patches,
            self.head_dim,
        );
        let lift = |values: Vec<f32>| -> Tensor<B, 4> {
            Tensor::from_data(
                TensorData::new(values, [1, 1, patches, self.head_dim]),
                (&device, DType::F32),
            )
        };
        let (cos, sin) = (lift(cos), lift(sin));

        for layer in &self.layers {
            hidden = layer.forward(hidden, &cos, &sin);
        }
        self.post_layernorm.forward(hidden)
    }
}

/// The projector: normalize, fold `merge × merge` patches into one token, and
/// map that token into the decoder's width.
pub struct Projector<B: Backend> {
    pre_norm: LayerNorm<B>,
    linear_1: Linear<B>,
    linear_2: Linear<B>,
    merge: usize,
}

impl<B: Backend> Projector<B> {
    pub(super) fn load(
        weights: &Weights,
        device: &B::Device,
        vision: &VisionConfig,
        hidden_size: usize,
    ) -> Result<Self, OcrError> {
        let merged = vision.hidden_size * vision.spatial_merge_size.pow(2);
        Ok(Self {
            // The projector's normalization keeps the decoder's epsilon, not
            // the tower's — it belongs to the join, not to either side.
            pre_norm: LayerNorm::load(
                weights,
                device,
                "mlp_AR.pre_norm",
                vision.hidden_size,
                1e-5,
            )?,
            linear_1: Linear::load(
                weights,
                device,
                "mlp_AR.linear_1",
                merged,
                merged,
            )?,
            linear_2: Linear::load(
                weights,
                device,
                "mlp_AR.linear_2",
                hidden_size,
                merged,
            )?,
            merge: vision.spatial_merge_size,
        })
    }

    /// `features` is `[patches, vision hidden]`; the result is
    /// `[patches / merge², decoder hidden]`.
    pub fn forward(
        &self,
        features: Tensor<B, 2>,
        grid: (usize, usize, usize),
    ) -> Tensor<B, 2> {
        let (t, height, width) = grid;
        let merge = self.merge;
        let dim = features.dims()[1];
        let folded = self
            .pre_norm
            .forward(features)
            .reshape([t, height / merge, merge, width / merge, merge, dim])
            .permute([0, 1, 3, 2, 4, 5])
            .reshape([
                t * (height / merge) * (width / merge),
                merge * merge * dim,
            ]);
        // The exact GELU (erf), where the tower's feed-forward uses the tanh
        // approximation.
        let hidden = activation::gelu(self.linear_1.forward(folded));
        self.linear_2.forward(hidden)
    }
}
