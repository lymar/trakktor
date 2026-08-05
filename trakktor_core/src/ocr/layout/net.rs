//! The layout network on candle: `PP-DocLayout_plus-L`, an RT-DETR-L.
//!
//! Four parts, and only the last one is unusual:
//!
//! - **`PPHGNetV2-L`** — an ordinary convolutional backbone. Its weights are
//!   numbered in declaration order, so the port walks the same structure with a
//!   counter rather than spelling eighty names out; the accompanying batch
//!   normalization is always `batch_norm2d_{n + 80}`, a rule that holds across
//!   the whole artifact.
//! - **`HybridEncoder`** — three 1×1 projections to 256 channels, one
//!   transformer layer over the coarsest map, and two feature pyramids running
//!   in opposite directions.
//! - **`RTDETRTransformer`** — a query-selection head over the 13 125 flattened
//!   positions, then six decoder layers, each refining three hundred boxes.
//! - **multi-scale deformable attention**, which needs a bilinear sampler that
//!   neither runtime has. It is written out here (four gathers and a blend per
//!   level) rather than approximated: the sampling rule is the whole of what
//!   the decoder does with the feature maps.
//!
//! Three things the export folded into constants, and each removes a piece of
//! the usual RT-DETR bookkeeping: the input size is fixed at 800×800, so the
//! **anchors** and the **mask of valid positions** are published as weights,
//! and the transformer layer's **positional embedding** is published too. None
//! of the three is computed here.
//!
//! What the network returns is not what the graph returns: the graph's tail
//! decodes boxes, takes the top three hundred and scales them onto the page.
//! That part is three hundred rows of arithmetic and lives on the host, in
//! [`super::detect`], where it can be read.

#[cfg(test)]
mod tests;

use candle_core::{D, DType, Device, Tensor};

use super::sampling::{self, Level};
use crate::ocr::{
    error::OcrError,
    paddle::net::{BatchNorm, Conv, LayerNorm, Loader},
};

/// Channels the encoder and the decoder work in.
const MODEL: usize = 256;
/// Attention heads, both in the encoder layer and in the decoder.
const HEADS: usize = 8;
/// Feature levels the decoder samples from.
const LEVELS: usize = 3;
/// Sampling points per head per level.
const POINTS: usize = 4;
/// Decoder layers.
const DEPTH: usize = 6;
/// Queries kept out of the encoder's positions.
const QUERIES: usize = 300;
/// Width of the feed-forward layer, in both the encoder and the decoder.
const FFN: usize = 1024;
/// The epsilon every `LayerNorm` in this graph carries.
const NORM_EPS: f64 = 1e-5;
/// Batch normalization always sits this far behind its convolution in the
/// published numbering.
const BN_OFFSET: usize = 80;

/// What the network says about a page.
pub struct Prediction {
    /// `[300, classes]` — logits, not probabilities. The sigmoid belongs to
    /// the decode, which also needs the raw values to rank across classes.
    pub logits: Vec<f32>,
    /// `[300, 4]` — boxes as centre, width and height, normalized to the input
    /// square.
    pub boxes: Vec<f32>,
    pub classes: usize,
}

/// A convolution followed by inference-time batch normalization.
#[derive(Debug)]
struct ConvBn {
    conv: Conv,
    norm: BatchNorm,
}

impl ConvBn {
    fn load(
        loader: &Loader,
        index: usize,
        dims: [usize; 4],
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            conv: Conv::load(
                loader,
                &format!("conv2d_{index}"),
                dims,
                stride,
                padding,
                groups,
            )?,
            norm: BatchNorm::load(
                loader,
                &format!("batch_norm2d_{}", index + BN_OFFSET),
                dims[0],
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        self.norm.forward(&self.conv.forward(x)?)
    }
}

fn relu(x: &Tensor) -> Result<Tensor, OcrError> { Ok(x.relu()?) }

fn silu(x: &Tensor) -> Result<Tensor, OcrError> { Ok(x.silu()?) }

// ------------------------------------------------------------------------
// Backbone
// ------------------------------------------------------------------------

/// The stem: two strided convolutions with a two-way split in between.
///
/// The split is the only part that needs care. Two kernels of size two run over
/// the first convolution's output padded by one on the right and the bottom,
/// while the same padded tensor goes through a max pool of the same shape; the
/// two results are concatenated. Padding one side only is what Paddle spells
/// `SAME` for an even kernel, and candle pads symmetrically, so the padding is
/// done by hand here.
#[derive(Debug)]
struct Stem {
    first: ConvBn,
    left: ConvBn,
    left2: ConvBn,
    third: ConvBn,
    fourth: ConvBn,
}

impl Stem {
    fn load(loader: &Loader, next: &mut usize) -> Result<Self, OcrError> {
        let mut conv = |dims: [usize; 4], stride, padding, groups| {
            let at = *next;
            *next += 1;
            ConvBn::load(loader, at, dims, stride, padding, groups)
        };
        Ok(Self {
            first: conv([32, 3, 3, 3], 2, 1, 1)?,
            left: conv([16, 32, 2, 2], 1, 0, 1)?,
            left2: conv([32, 16, 2, 2], 1, 0, 1)?,
            third: conv([32, 64, 3, 3], 2, 1, 1)?,
            fourth: conv([48, 32, 1, 1], 1, 0, 1)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let y = relu(&self.first.forward(x)?)?;
        let padded = pad_end(&y, 1)?;
        let left = relu(&self.left.forward(&padded)?)?;
        let left = relu(&self.left2.forward(&pad_end(&left, 1)?)?)?;
        let right = padded.max_pool2d_with_stride(2, 1)?;
        let y = Tensor::cat(&[&right, &left], 1)?;
        let y = relu(&self.third.forward(&y)?)?;
        relu(&self.fourth.forward(&y)?)
    }
}

/// Pads the right and bottom edges with zeros. The activations feeding it are
/// non-negative, so zero is also the identity for the max pool that reads it.
fn pad_end(x: &Tensor, by: usize) -> Result<Tensor, OcrError> {
    Ok(x.pad_with_zeros(2, 0, by)?.pad_with_zeros(3, 0, by)?)
}

/// One block of a stage: six convolutions whose outputs are all kept, then
/// concatenated with the input and squeezed back down.
#[derive(Debug)]
struct HgBlock {
    layers: Vec<HgLayer>,
    squeeze: ConvBn,
    excite: ConvBn,
    residual: bool,
}

/// A layer inside a block: one convolution, or — in the deeper stages — a
/// pointwise convolution followed by a depthwise one.
#[derive(Debug)]
enum HgLayer {
    Plain(ConvBn),
    Light { point: ConvBn, depth: ConvBn },
}

impl HgLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        match self {
            Self::Plain(conv) => relu(&conv.forward(x)?),
            // The pointwise half carries no activation; only the depthwise one
            // does.
            Self::Light { point, depth } => {
                relu(&depth.forward(&point.forward(x)?)?)
            },
        }
    }
}

impl HgBlock {
    #[allow(clippy::too_many_arguments)]
    fn load(
        loader: &Loader,
        next: &mut usize,
        in_channels: usize,
        mid: usize,
        out: usize,
        kernel: usize,
        light: bool,
        residual: bool,
    ) -> Result<Self, OcrError> {
        const LAYERS: usize = 6;
        let pad = (kernel - 1) / 2;
        let mut layers = Vec::with_capacity(LAYERS);
        for at in 0..LAYERS {
            let from = if at == 0 { in_channels } else { mid };
            let mut conv = |dims: [usize; 4], stride, padding, groups| {
                let index = *next;
                *next += 1;
                ConvBn::load(loader, index, dims, stride, padding, groups)
            };
            layers.push(if light {
                HgLayer::Light {
                    point: conv([mid, from, 1, 1], 1, 0, 1)?,
                    depth: conv([mid, 1, kernel, kernel], 1, pad, mid)?,
                }
            } else {
                HgLayer::Plain(conv([mid, from, kernel, kernel], 1, pad, 1)?)
            });
        }
        let total = in_channels + LAYERS * mid;
        let squeeze = {
            let index = *next;
            *next += 1;
            ConvBn::load(loader, index, [out / 2, total, 1, 1], 1, 0, 1)?
        };
        let excite = {
            let index = *next;
            *next += 1;
            ConvBn::load(loader, index, [out, out / 2, 1, 1], 1, 0, 1)?
        };
        Ok(Self {
            layers,
            squeeze,
            excite,
            residual,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let mut kept = Vec::with_capacity(self.layers.len() + 1);
        kept.push(x.clone());
        let mut state = x.clone();
        for layer in &self.layers {
            state = layer.forward(&state)?;
            kept.push(state.clone());
        }
        let joined = Tensor::cat(&kept, 1)?;
        let y = relu(&self.squeeze.forward(&joined)?)?;
        let y = relu(&self.excite.forward(&y)?)?;
        if self.residual { Ok((y + x)?) } else { Ok(y) }
    }
}

/// One stage: an optional depthwise halving of the resolution, then its blocks.
#[derive(Debug)]
struct HgStage {
    downsample: Option<ConvBn>,
    blocks: Vec<HgBlock>,
}

/// The four stages of `PPHGNetV2-L`, as its published configuration spells
/// them.
struct StageSpec {
    in_channels: usize,
    mid: usize,
    out: usize,
    blocks: usize,
    kernel: usize,
    light: bool,
    downsample: bool,
}

const STAGES: [StageSpec; 4] = [
    StageSpec {
        in_channels: 48,
        mid: 48,
        out: 128,
        blocks: 1,
        kernel: 3,
        light: false,
        downsample: false,
    },
    StageSpec {
        in_channels: 128,
        mid: 96,
        out: 512,
        blocks: 1,
        kernel: 3,
        light: false,
        downsample: true,
    },
    StageSpec {
        in_channels: 512,
        mid: 192,
        out: 1024,
        blocks: 3,
        kernel: 5,
        light: true,
        downsample: true,
    },
    StageSpec {
        in_channels: 1024,
        mid: 384,
        out: 2048,
        blocks: 1,
        kernel: 5,
        light: true,
        downsample: true,
    },
];

#[derive(Debug)]
struct Backbone {
    stem: Stem,
    stages: Vec<HgStage>,
}

impl Backbone {
    fn load(loader: &Loader, next: &mut usize) -> Result<Self, OcrError> {
        let stem = Stem::load(loader, next)?;
        let mut stages = Vec::with_capacity(STAGES.len());
        for spec in &STAGES {
            let downsample = if spec.downsample {
                let index = *next;
                *next += 1;
                // Depthwise, and with no activation of its own.
                Some(ConvBn::load(
                    loader,
                    index,
                    [spec.in_channels, 1, 3, 3],
                    2,
                    1,
                    spec.in_channels,
                )?)
            } else {
                None
            };
            let mut blocks = Vec::with_capacity(spec.blocks);
            for at in 0..spec.blocks {
                blocks.push(HgBlock::load(
                    loader,
                    next,
                    if at == 0 { spec.in_channels } else { spec.out },
                    spec.mid,
                    spec.out,
                    spec.kernel,
                    spec.light,
                    at != 0,
                )?);
            }
            stages.push(HgStage { downsample, blocks });
        }
        Ok(Self { stem, stages })
    }

    /// The last three stages' outputs, coarsest last.
    fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>, OcrError> {
        let mut state = self.stem.forward(x)?;
        let mut out = Vec::with_capacity(LEVELS);
        for (at, stage) in self.stages.iter().enumerate() {
            if let Some(down) = &stage.downsample {
                state = down.forward(&state)?;
            }
            for block in &stage.blocks {
                state = block.forward(&state)?;
            }
            if at > 0 {
                out.push(state.clone());
            }
        }
        Ok(out)
    }
}

// ------------------------------------------------------------------------
// Hybrid encoder
// ------------------------------------------------------------------------

/// A fully connected layer, loaded by its published index.
fn linear(
    loader: &Loader,
    index: usize,
    from: usize,
    to: usize,
) -> Result<crate::ocr::paddle::net::Linear, OcrError> {
    crate::ocr::paddle::net::Linear::load(
        loader,
        &format!("linear_{index}"),
        from,
        to,
    )
}

fn norm(
    loader: &Loader,
    index: usize,
    width: usize,
) -> Result<LayerNorm, OcrError> {
    LayerNorm::load(loader, &format!("layer_norm_{index}"), width, NORM_EPS)
}

/// Multi-head self-attention with a fused `qkv` projection.
///
/// The published weight is one `[256, 768]` matrix and one `[768]` bias, and
/// the graph slices them into three equal blocks — queries, keys, values, in
/// that order. The positional embedding goes into the queries and the keys and
/// **not** into the values, which is a property of the layer rather than of
/// this port and is easy to lose.
#[derive(Debug, Clone)]
struct Attention {
    qkv: Tensor,
    qkv_bias: Tensor,
    out: crate::ocr::paddle::net::Linear,
}

impl Attention {
    fn load(
        loader: &Loader,
        attention: usize,
        out: usize,
    ) -> Result<Self, OcrError> {
        let name = format!("multi_head_attention_{attention}");
        Ok(Self {
            qkv: loader.get(&format!("{name}.w_0"), &[MODEL, 3 * MODEL])?,
            qkv_bias: loader.get(&format!("{name}.b_0"), &[3 * MODEL])?,
            out: linear(loader, out, MODEL, MODEL)?,
        })
    }

    /// `x` is `[tokens, MODEL]`.
    fn forward(
        &self,
        x: &Tensor,
        positions: Option<&Tensor>,
    ) -> Result<Tensor, OcrError> {
        let tokens = x.dim(0)?;
        let head = MODEL / HEADS;
        let queried = match positions {
            None => x.clone(),
            Some(positions) => (x + positions)?,
        };
        let part = |source: &Tensor, at: usize| -> Result<Tensor, OcrError> {
            Ok(source
                .matmul(&self.qkv.narrow(1, at * MODEL, MODEL)?.contiguous()?)?
                .broadcast_add(&self.qkv_bias.narrow(0, at * MODEL, MODEL)?)?
                .reshape((tokens, HEADS, head))?
                .transpose(0, 1)?
                .contiguous()?)
        };
        let q = part(&queried, 0)?;
        let k = part(&queried, 1)?;
        let v = part(x, 2)?;

        let scores = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? *
            (head as f64).powf(-0.5))?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let joined = weights
            .matmul(&v)?
            .transpose(0, 1)?
            .reshape((tokens, MODEL))?;
        self.out.forward(&joined)
    }
}

/// The single transformer layer over the coarsest feature map.
///
/// Post-norm, feed-forward with a GELU in the middle, and a positional
/// embedding that is a published constant because the input geometry is fixed.
#[derive(Debug)]
struct EncoderLayer {
    attention: Attention,
    attention_norm: LayerNorm,
    up: crate::ocr::paddle::net::Linear,
    down: crate::ocr::paddle::net::Linear,
    final_norm: LayerNorm,
    positions: Tensor,
}

impl EncoderLayer {
    fn load(loader: &Loader, tokens: usize) -> Result<Self, OcrError> {
        Ok(Self {
            attention: Attention::load(loader, 0, 0)?,
            attention_norm: norm(loader, 0, MODEL)?,
            up: linear(loader, 1, MODEL, FFN)?,
            down: linear(loader, 2, FFN, MODEL)?,
            final_norm: norm(loader, 1, MODEL)?,
            positions: loader
                .get("eager_tmp_0", &[1, tokens, MODEL])?
                .reshape((tokens, MODEL))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let attended = self.attention.forward(x, Some(&self.positions))?;
        let x = self.attention_norm.forward(&(x + attended)?)?;
        let up = self.up.forward(&x)?.gelu_erf()?;
        let down = self.down.forward(&up)?;
        self.final_norm.forward(&(&x + down)?)
    }
}

/// A rep-block, published in its collapsed form: one 3×3 convolution with a
/// bias, and no batch normalization at all.
#[derive(Debug)]
struct RepBlock {
    conv: Conv,
}

impl RepBlock {
    fn load(loader: &Loader, index: usize) -> Result<Self, OcrError> {
        Ok(Self {
            conv: Conv::load(
                loader,
                &format!("conv2d_{index}"),
                [MODEL, MODEL, 3, 3],
                1,
                1,
                1,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        silu(&self.conv.forward(x)?)
    }
}

/// The block on every rung of both pyramids: two 1×1 branches, three rep-blocks
/// on one of them, and a sum.
#[derive(Debug)]
struct CspRep {
    left: ConvBn,
    blocks: [RepBlock; 3],
    right: ConvBn,
}

impl CspRep {
    fn load(
        loader: &Loader,
        left: usize,
        blocks: usize,
        right: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            left: ConvBn::load(
                loader,
                left,
                [MODEL, 2 * MODEL, 1, 1],
                1,
                0,
                1,
            )?,
            blocks: [
                RepBlock::load(loader, blocks)?,
                RepBlock::load(loader, blocks + 1)?,
                RepBlock::load(loader, blocks + 2)?,
            ],
            right: ConvBn::load(
                loader,
                right,
                [MODEL, 2 * MODEL, 1, 1],
                1,
                0,
                1,
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let mut left = silu(&self.left.forward(x)?)?;
        for block in &self.blocks {
            left = block.forward(&left)?;
        }
        let right = silu(&self.right.forward(x)?)?;
        Ok((left + right)?)
    }
}

#[derive(Debug)]
struct Encoder {
    project: [ConvBn; LEVELS],
    layer: EncoderLayer,
    lateral: [ConvBn; 2],
    fpn: [CspRep; 2],
    downsample: [ConvBn; 2],
    pan: [CspRep; 2],
}

impl Encoder {
    fn load(loader: &Loader, coarsest_tokens: usize) -> Result<Self, OcrError> {
        let project = |index: usize, from: usize| {
            ConvBn::load(loader, index, [MODEL, from, 1, 1], 1, 0, 1)
        };
        let lateral = |index: usize| {
            ConvBn::load(loader, index, [MODEL, MODEL, 1, 1], 1, 0, 1)
        };
        let down = |index: usize| {
            ConvBn::load(loader, index, [MODEL, MODEL, 3, 3], 2, 1, 1)
        };
        Ok(Self {
            project: [
                project(80, 512)?,
                project(81, 1024)?,
                project(82, 2048)?,
            ],
            layer: EncoderLayer::load(loader, coarsest_tokens)?,
            lateral: [lateral(83)?, lateral(92)?],
            fpn: [
                CspRep::load(loader, 84, 122, 85)?,
                CspRep::load(loader, 93, 125, 94)?,
            ],
            downsample: [down(101)?, down(110)?],
            pan: [
                CspRep::load(loader, 102, 128, 103)?,
                CspRep::load(loader, 111, 131, 112)?,
            ],
        })
    }

    fn forward(&self, features: &[Tensor]) -> Result<Vec<Tensor>, OcrError> {
        let mut levels: Vec<Tensor> = features
            .iter()
            .zip(&self.project)
            .map(|(feature, project)| project.forward(feature))
            .collect::<Result<_, _>>()?;

        // The transformer runs on the coarsest map only: it is the one small
        // enough for attention over every position to be worth it.
        let coarsest = levels.len() - 1;
        levels[coarsest] = {
            let map = &levels[coarsest];
            let (_, channels, height, width) = map.dims4()?;
            let tokens = map
                .reshape((channels, height * width))?
                .transpose(0, 1)?
                .contiguous()?;
            self.layer
                .forward(&tokens)?
                .transpose(0, 1)?
                .reshape((1, channels, height, width))?
        };

        // Top down: each rung is upsampled and joined with the backbone map
        // below it.
        let mut top_down = vec![levels[coarsest].clone()];
        for at in 0..2 {
            let below = &levels[coarsest - at - 1];
            let top =
                self.lateral[at].forward(top_down.last().expect("a rung"))?;
            let top = silu(&top)?;
            *top_down.last_mut().expect("a rung") = top.clone();
            let (_, _, height, width) = top.dims4()?;
            let up = top.upsample_nearest2d(height * 2, width * 2)?;
            top_down
                .push(self.fpn[at].forward(&Tensor::cat(&[&up, below], 1)?)?);
        }
        top_down.reverse();

        // Bottom up: back the other way, halving the resolution each time.
        let mut out = vec![top_down[0].clone()];
        for at in 0..2 {
            let below = out.last().expect("a rung");
            let down = silu(&self.downsample[at].forward(below)?)?;
            let joined = Tensor::cat(&[&down, &top_down[at + 1]], 1)?;
            out.push(self.pan[at].forward(&joined)?);
        }
        Ok(out)
    }
}

// ------------------------------------------------------------------------
// Decoder
// ------------------------------------------------------------------------

/// A two- or three-layer perceptron with ReLU between the layers, which is how
/// this graph spells every small prediction head.
#[derive(Debug)]
struct Mlp {
    layers: Vec<crate::ocr::paddle::net::Linear>,
}

impl Mlp {
    fn load(
        loader: &Loader,
        first: usize,
        widths: &[usize],
    ) -> Result<Self, OcrError> {
        let mut layers = Vec::with_capacity(widths.len() - 1);
        for at in 0..widths.len() - 1 {
            layers.push(linear(
                loader,
                first + at,
                widths[at],
                widths[at + 1],
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let mut state = x.clone();
        for (at, layer) in self.layers.iter().enumerate() {
            state = layer.forward(&state)?;
            if at + 1 < self.layers.len() {
                state = relu(&state)?;
            }
        }
        Ok(state)
    }
}

/// One decoder layer.
///
/// Every weight of it is a *copy*: the exporter duplicated one module six times
/// and told the copies apart by a suffix, so each layer asks for its own copy
/// of the same published name.
#[derive(Debug)]
struct DecoderLayer {
    attention: Attention,
    attention_norm: LayerNorm,
    offsets: crate::ocr::paddle::net::Linear,
    weights: crate::ocr::paddle::net::Linear,
    value: crate::ocr::paddle::net::Linear,
    sampled_out: crate::ocr::paddle::net::Linear,
    cross_norm: LayerNorm,
    up: crate::ocr::paddle::net::Linear,
    down: crate::ocr::paddle::net::Linear,
    final_norm: LayerNorm,
    /// The box refinement this layer contributes.
    box_head: Mlp,
}

impl DecoderLayer {
    fn load(loader: &Loader, at: usize) -> Result<Self, OcrError> {
        let copy = |name: &str, dims: &[usize]| loader.copy(name, at, dims);
        let linear_copy =
            |index: usize,
             from: usize,
             to: usize|
             -> Result<crate::ocr::paddle::net::Linear, OcrError> {
                Ok(crate::ocr::paddle::net::Linear::from_parts(
                    copy(&format!("linear_{index}.w_0"), &[from, to])?,
                    Some(copy(&format!("linear_{index}.b_0"), &[to])?),
                ))
            };
        let norm_copy = |index: usize| -> Result<LayerNorm, OcrError> {
            Ok(LayerNorm::from_parts(
                copy(&format!("layer_norm_{index}.w_0"), &[MODEL])?,
                copy(&format!("layer_norm_{index}.b_0"), &[MODEL])?,
                NORM_EPS,
            ))
        };
        Ok(Self {
            attention: Attention {
                qkv: copy("multi_head_attention_1.w_0", &[MODEL, 3 * MODEL])?,
                qkv_bias: copy("multi_head_attention_1.b_0", &[3 * MODEL])?,
                out: linear_copy(3, MODEL, MODEL)?,
            },
            attention_norm: norm_copy(2)?,
            offsets: linear_copy(4, MODEL, HEADS * LEVELS * POINTS * 2)?,
            weights: linear_copy(5, MODEL, HEADS * LEVELS * POINTS)?,
            value: linear_copy(6, MODEL, MODEL)?,
            sampled_out: linear_copy(7, MODEL, MODEL)?,
            cross_norm: norm_copy(3)?,
            up: linear_copy(8, MODEL, FFN)?,
            down: linear_copy(9, FFN, MODEL)?,
            final_norm: norm_copy(4)?,
            // The six box heads are separate modules rather than copies, so
            // they carry plain indices: 23, 26, 29, …
            box_head: Mlp::load(
                loader,
                23 + 3 * at,
                &[MODEL, MODEL, MODEL, 4],
            )?,
        })
    }
}

/// The decoder: query selection, then six refining layers.
#[derive(Debug)]
struct Decoder {
    project: [ConvBn; LEVELS],
    anchors: Tensor,
    valid: Tensor,
    memory_project: crate::ocr::paddle::net::Linear,
    memory_norm: LayerNorm,
    memory_score: crate::ocr::paddle::net::Linear,
    memory_box: Mlp,
    query_positions: Mlp,
    layers: Vec<DecoderLayer>,
    score: crate::ocr::paddle::net::Linear,
    classes: usize,
}

impl Decoder {
    fn load(
        loader: &Loader,
        positions: usize,
        classes: usize,
    ) -> Result<Self, OcrError> {
        let project = |index: usize| {
            ConvBn::load(loader, index, [MODEL, MODEL, 1, 1], 1, 0, 1)
        };
        let mut layers = Vec::with_capacity(DEPTH);
        for at in 0..DEPTH {
            layers.push(DecoderLayer::load(loader, at)?);
        }
        Ok(Self {
            project: [project(119)?, project(120)?, project(121)?],
            anchors: loader
                .get("eager_tmp_1", &[1, positions, 4])?
                .reshape((positions, 4))?,
            valid: loader
                .get("eager_tmp_2", &[1, positions, 1])?
                .reshape((positions, 1))?,
            memory_project: linear(loader, 12, MODEL, MODEL)?,
            memory_norm: norm(loader, 5, MODEL)?,
            memory_score: linear(loader, 13, MODEL, classes)?,
            memory_box: Mlp::load(loader, 14, &[MODEL, MODEL, MODEL, 4])?,
            query_positions: Mlp::load(loader, 10, &[4, 2 * MODEL, MODEL])?,
            layers,
            score: linear(loader, 22, MODEL, classes)?,
            classes,
        })
    }

    fn forward(&self, maps: &[Tensor]) -> Result<Prediction, OcrError> {
        // Flatten the three maps into one sequence of positions.
        let mut levels = Vec::with_capacity(LEVELS);
        let mut flattened = Vec::with_capacity(LEVELS);
        let mut start = 0usize;
        for (map, project) in maps.iter().zip(&self.project) {
            let projected = project.forward(map)?;
            let (_, channels, height, width) = projected.dims4()?;
            flattened.push(
                projected
                    .reshape((channels, height * width))?
                    .transpose(0, 1)?
                    .contiguous()?,
            );
            levels.push(Level {
                height,
                width,
                start,
            });
            start += height * width;
        }
        let memory = Tensor::cat(&flattened, 0)?;
        // The mask zeroes the positions whose anchor falls outside the page —
        // the graph multiplies rather than selects, and so does this.
        let memory = memory.broadcast_mul(&self.valid)?;

        let projected = self
            .memory_norm
            .forward(&self.memory_project.forward(&memory)?)?;
        let scores = self.memory_score.forward(&projected)?;
        let boxes = (self.memory_box.forward(&projected)? + &self.anchors)?;

        // Query selection: the three hundred positions whose best class scores
        // highest. Done on the host — three hundred out of thirteen thousand is
        // a sort, not arithmetic, and the indices are needed as indices.
        let best = scores.max(D::Minus1)?.to_vec1::<f32>()?;
        let mut order: Vec<u32> = (0..best.len() as u32).collect();
        order.sort_by(|a, b| best[*b as usize].total_cmp(&best[*a as usize]));
        order.truncate(QUERIES);
        let picked = Tensor::from_vec(order, QUERIES, memory.device())?;

        let mut reference = boxes.index_select(&picked, 0)?;
        let mut state = projected.index_select(&picked, 0)?;
        reference = candle_nn::ops::sigmoid(&reference)?;

        for layer in &self.layers {
            let positions = self.query_positions.forward(&reference)?;
            let attended = layer.attention.forward(&state, Some(&positions))?;
            state = layer.attention_norm.forward(&(&state + attended)?)?;

            let sampled = deformable(
                layer, &state, &positions, &memory, &reference, &levels,
            )?;
            state = layer.cross_norm.forward(&(&state + sampled)?)?;

            let up = relu(&layer.up.forward(&state)?)?;
            let down = layer.down.forward(&up)?;
            state = layer.final_norm.forward(&(&state + down)?)?;

            let delta = layer.box_head.forward(&state)?;
            reference = candle_nn::ops::sigmoid(
                &(delta + inverse_sigmoid(&reference)?)?,
            )?;
        }

        Ok(Prediction {
            logits: self
                .score
                .forward(&state)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            boxes: reference.flatten_all()?.to_vec1::<f32>()?,
            classes: self.classes,
        })
    }
}

/// `log(x / (1 - x))`, clamped the way the reference clamps it.
fn inverse_sigmoid(x: &Tensor) -> Result<Tensor, OcrError> {
    const EPS: f64 = 1e-5;
    let x = x.clamp(0f32, 1f32)?;
    let low = x.clamp(EPS as f32, f32::MAX)?;
    let high = (1.0 - &x)?.clamp(EPS as f32, f32::MAX)?;
    Ok((low / high)?.log()?)
}

/// Multi-scale deformable attention for one decoder layer.
///
/// Each query looks at four points on each of the three feature maps, in a
/// window the size of its own current box, and blends what it finds by learned
/// weights. The sampling is bilinear with zeros outside the map, which is
/// `grid_sample` — an operation neither runtime has, so it is four gathers and
/// a weighted sum here.
fn deformable(
    layer: &DecoderLayer,
    state: &Tensor,
    positions: &Tensor,
    memory: &Tensor,
    reference: &Tensor,
    levels: &[Level],
) -> Result<Tensor, OcrError> {
    let head = MODEL / HEADS;
    let queried = (state + positions)?;

    // Values are projected once and split by head; the heads are the leading
    // axis so that a gather can run over all of them at once.
    let value = layer.value.forward(memory)?;

    let offsets = layer
        .offsets
        .forward(&queried)?
        .reshape((QUERIES, HEADS, LEVELS, POINTS, 2))?;
    let weights = layer.weights.forward(&queried)?.reshape((
        QUERIES,
        HEADS,
        LEVELS * POINTS,
    ))?;
    let weights = candle_nn::ops::softmax_last_dim(&weights)?
        .reshape((QUERIES, HEADS, LEVELS, POINTS))?;

    // A box-shaped window: the offsets are in units of a quarter of the box's
    // own width and height.
    let centre = reference.narrow(1, 0, 2)?.reshape((QUERIES, 1, 1, 1, 2))?;
    let size = reference.narrow(1, 2, 2)?.reshape((QUERIES, 1, 1, 1, 2))?;
    let scaled = offsets
        .broadcast_mul(&(size * (0.5 / POINTS as f64))?)?
        .broadcast_add(&centre)?;

    let samples = QUERIES * POINTS;
    let mut total: Option<Tensor> = None;
    for (at, level) in levels.iter().enumerate() {
        let count = level.height * level.width;
        // Heads are folded into the position axis so that one flat lookup
        // serves all eight of them: head `h` owns rows `h * count ..`.
        let map = value
            .narrow(0, level.start, count)?
            .reshape((count, HEADS, head))?
            .transpose(0, 1)?
            .reshape((HEADS * count, head))?
            .contiguous()?;
        let grid = scaled
            .narrow(2, at, 1)?
            .reshape((QUERIES, HEADS, POINTS, 2))?
            .transpose(0, 1)?
            .reshape((HEADS * samples, 2))?
            .contiguous()?;
        let sampled = sample_bilinear(&map, &grid, level, count)?;
        let weight = weights
            .narrow(2, at, 1)?
            .reshape((QUERIES, HEADS, POINTS))?
            .transpose(0, 1)?
            .reshape((HEADS, samples, 1))?
            .contiguous()?;
        let weighted = sampled
            .broadcast_mul(&weight)?
            .reshape((HEADS, QUERIES, POINTS, head))?
            .sum(2)?;
        total = Some(match total {
            None => weighted,
            Some(sum) => (sum + weighted)?,
        });
    }
    let joined = total
        .expect("three levels")
        .transpose(0, 1)?
        .reshape((QUERIES, MODEL))?;
    layer.sampled_out.forward(&joined)
}

/// Bilinear sampling of a feature map at normalized positions.
///
/// `map` is `[HEADS * positions, channels]`, positions laid out row by row and
/// heads one after another; `grid` is `[HEADS * samples, 2]` in `0..1` of the
/// map's own width and height, in the same head order. The result is
/// `[HEADS, samples, channels]`.
///
/// The positions come back to the host for the index arithmetic — a few
/// thousand floats per level per layer, and both runtimes do it through the
/// same [`sampling::corners`] so they cannot round a corner differently.
fn sample_bilinear(
    map: &Tensor,
    grid: &Tensor,
    level: &Level,
    count: usize,
) -> Result<Tensor, OcrError> {
    let device = map.device();
    let channels = map.dim(1)?;
    let rows = grid.dim(0)?;
    let samples = rows / HEADS;
    let (indices, blend) = sampling::corners(
        &grid.flatten_all()?.to_vec1::<f32>()?,
        level,
        count,
        HEADS,
    );

    let mut total: Option<Tensor> = None;
    for corner in 0..4 {
        let index = Tensor::from_slice(
            &indices[corner * rows..(corner + 1) * rows],
            rows,
            device,
        )?;
        let gathered = map.index_select(&index, 0)?;
        let weight = Tensor::from_slice(
            &blend[corner * rows..(corner + 1) * rows],
            (rows, 1),
            device,
        )?
        .to_dtype(map.dtype())?;
        let scaled = gathered.broadcast_mul(&weight)?;
        total = Some(match total {
            None => scaled,
            Some(sum) => (sum + scaled)?,
        });
    }
    Ok(total
        .expect("four corners")
        .reshape((HEADS, samples, channels))?)
}

// ------------------------------------------------------------------------
// The network
// ------------------------------------------------------------------------

/// The whole detector.
#[derive(Debug)]
pub struct Net {
    backbone: Backbone,
    encoder: Encoder,
    decoder: Decoder,
    device: Device,
}

impl Net {
    /// The device the weights were loaded onto, which is where the input has
    /// to be built.
    pub fn device(&self) -> &Device { &self.device }

    /// Loads the network for a square input of `side` pixels and `classes`
    /// labels.
    pub fn load(
        loader: &Loader,
        side: usize,
        classes: usize,
    ) -> Result<Self, OcrError> {
        // The three feature levels of a square input: an eighth, a sixteenth
        // and a thirty-second of the side.
        let coarsest = side / 32;
        let positions =
            (side / 8).pow(2) + (side / 16).pow(2) + coarsest.pow(2);
        let mut next = 0usize;
        let backbone = Backbone::load(loader, &mut next)?;
        Ok(Self {
            backbone,
            encoder: Encoder::load(loader, coarsest * coarsest)?,
            decoder: Decoder::load(loader, positions, classes)?,
            device: loader.device().clone(),
        })
    }

    /// Runs one page. `input` is `1 × 3 × side × side`.
    pub fn forward(&self, input: &Tensor) -> Result<Prediction, OcrError> {
        let features = self.backbone.forward(input)?;
        let maps = self.encoder.forward(&features)?;
        self.decoder.forward(&maps)
    }
}

/// Builds the input tensor from prepared planar data.
pub fn input(
    data: &[f32],
    side: usize,
    device: &Device,
) -> Result<Tensor, OcrError> {
    Ok(Tensor::from_slice(data, (1, 3, side, side), device)?
        .to_dtype(DType::F32)?)
}
