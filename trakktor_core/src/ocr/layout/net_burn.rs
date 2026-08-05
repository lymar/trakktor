//! The layout network on burn.
//!
//! An alternative implementation of [`net`](super::net) with the same numerical
//! semantics, written idiomatically for burn rather than as a mirror. The
//! checkpoint is the same published directory: weights are read through the
//! shared Paddle artifact reader and handed over as `f32` values, which is what
//! this family of models is published in.
//!
//! **This is the first Paddle-format network in the tree with a burn runtime,**
//! and it is the easiest one to give it to: the export fixes the input at
//! 800×800, so every shape in the graph is a constant. The worry that kept burn
//! away from OCR — autotuning against shapes that change with every page —
//! simply does not arise here.
//!
//! Backends: ndarray on the CPU and wgpu with MSL-compiled kernels on Metal,
//! both in f32. Half precision is not offered: the network is one page's worth
//! of convolutions, its weights are published in f32, and the accuracy of a
//! box's corner is the whole output.
//!
//! Two things are deliberately shared with the candle runtime rather than
//! rewritten: the host-side [corner arithmetic](super::sampling) of the
//! deformable sampler, so the two cannot round a sample differently, and the
//! decode of the network's output, which lives above both.

#[cfg(test)]
mod tests;

use std::path::Path;

use burn::{
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        wgpu::{CubeBackend, WgpuDevice, WgpuRuntime},
    },
    tensor::{
        Int, Tensor, TensorData, activation,
        backend::Backend,
        module::{conv2d, interpolate, linear, max_pool2d},
        ops::{ConvOptions, InterpolateMode, InterpolateOptions},
    },
};

use super::{
    net::Prediction,
    sampling::{self, Level},
};
use crate::ocr::{
    error::OcrError,
    paddle::{
        artifact::{Artifact, RawTensor},
        pipeline::Device,
    },
};

// The network's shape, spelled the same way as in the candle runtime.
const MODEL: usize = 256;
const HEADS: usize = 8;
const LEVELS: usize = 3;
const POINTS: usize = 4;
const DEPTH: usize = 6;
const QUERIES: usize = 300;
const FFN: usize = 1024;
const NORM_EPS: f64 = 1e-5;
const BN_OFFSET: usize = 80;

/// The network behind whichever backend was chosen.
pub enum Net {
    Cpu(Model<NdArray<f32>>),
    Metal(Model<CubeBackend<WgpuRuntime, f32, i32, u32>>),
}

impl Net {
    /// Reads the artifacts in `dir` onto `device`.
    pub fn load(
        dir: &Path,
        device: Device,
        side: usize,
        classes: usize,
    ) -> Result<Self, OcrError> {
        let artifact = Artifact::load(dir)?;
        match device {
            Device::Cpu => Ok(Self::Cpu(Model::load(
                &artifact,
                &NdArrayDevice::Cpu,
                side,
                classes,
            )?)),
            Device::Metal => Ok(Self::Metal(Model::load(
                &artifact,
                &WgpuDevice::default(),
                side,
                classes,
            )?)),
        }
    }

    /// Runs one page from planar `1 × 3 × side × side` data.
    pub fn forward(
        &self,
        data: &[f32],
        side: usize,
    ) -> Result<Prediction, OcrError> {
        match self {
            Self::Cpu(model) => model.forward(data, side),
            Self::Metal(model) => model.forward(data, side),
        }
    }
}

/// Reads a weight of a known shape.
fn weight<B: Backend, const D: usize>(
    artifact: &Artifact,
    device: &B::Device,
    name: &str,
    shape: [usize; D],
) -> Result<Tensor<B, D>, OcrError> {
    let raw: &RawTensor = raw(artifact, name, &shape)?;
    Ok(Tensor::from_data(
        TensorData::new(raw.data.clone(), shape),
        device,
    ))
}

/// Looks a weight up by the name the network calls it, allowing for the suffix
/// the exporter appends to a module it copied. A name the exporter suffixed
/// *several* times belongs to a repeated module and is asked for by
/// [`copy`] instead, so an ambiguous base is left to fail as missing.
fn raw<'a>(
    artifact: &'a Artifact,
    name: &str,
    shape: &[usize],
) -> Result<&'a RawTensor, OcrError> {
    if artifact.has(name) {
        return artifact.shaped(name, shape);
    }
    let copies: Vec<&str> = artifact
        .names()
        .filter(|other| strip_copies(other) == name)
        .collect();
    match copies.as_slice() {
        [only] => artifact.shaped(only, shape),
        _ => artifact.shaped(name, shape),
    }
}

/// Reads one copy of a weight a repeated module owns.
///
/// The exporter duplicated the decoder layer six times and told the copies
/// apart by a `_deepcopy_<n>` suffix chain whose numbers rise with declaration
/// order. Same rule as the candle loader's, and it is the whole of the mapping.
fn copy<B: Backend, const D: usize>(
    artifact: &Artifact,
    device: &B::Device,
    base: &str,
    at: usize,
    shape: [usize; D],
) -> Result<Tensor<B, D>, OcrError> {
    let mut names: Vec<&str> = artifact
        .names()
        .filter(|name| strip_copies(name) == base)
        .collect();
    names.sort_by_key(|name| copy_indices(name));
    let name = names.get(at).ok_or_else(|| {
        OcrError::Artifact(format!(
            "`{base}` has {} copies, not {}",
            names.len(),
            at + 1
        ))
    })?;
    weight(artifact, device, name, shape)
}

fn strip_copies(name: &str) -> &str {
    let mut base = name;
    while let Some(cut) = base.rfind("_deepcopy_") {
        let tail = &base[cut + "_deepcopy_".len()..];
        if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) {
            base = &base[..cut];
        } else {
            break;
        }
    }
    base
}

fn copy_indices(name: &str) -> Vec<u64> {
    let mut indices = Vec::new();
    let mut rest = name;
    while let Some(cut) = rest.rfind("_deepcopy_") {
        match rest[cut + "_deepcopy_".len()..].parse::<u64>() {
            Ok(index) => {
                indices.push(index);
                rest = &rest[..cut];
            },
            Err(_) => break,
        }
    }
    indices.reverse();
    indices
}

/// A convolution, and the batch normalization that always follows it, folded
/// into the per-channel scale and shift it collapses to.
struct ConvBn<B: Backend> {
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 1>>,
    scale: Tensor<B, 4>,
    shift: Tensor<B, 4>,
    stride: usize,
    padding: usize,
    groups: usize,
}

impl<B: Backend> ConvBn<B> {
    fn load(
        artifact: &Artifact,
        device: &B::Device,
        index: usize,
        dims: [usize; 4],
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> Result<Self, OcrError> {
        let channels = dims[0];
        let name = format!("batch_norm2d_{}", index + BN_OFFSET);
        let read = |suffix: &str| -> Result<Vec<f32>, OcrError> {
            Ok(raw(artifact, &format!("{name}.{suffix}"), &[channels])?
                .data
                .clone())
        };
        let (gamma, beta, mean, var) =
            (read("w_0")?, read("b_0")?, read("w_1")?, read("w_2")?);
        let mut scale = Vec::with_capacity(channels);
        let mut shift = Vec::with_capacity(channels);
        for c in 0..channels {
            let s = gamma[c] / (var[c] + NORM_EPS as f32).sqrt();
            scale.push(s);
            shift.push(beta[c] - mean[c] * s);
        }
        Ok(Self {
            weight: weight(
                artifact,
                device,
                &format!("conv2d_{index}.w_0"),
                dims,
            )?,
            bias: None,
            scale: Tensor::from_data(
                TensorData::new(scale, [1, channels, 1, 1]),
                device,
            ),
            shift: Tensor::from_data(
                TensorData::new(shift, [1, channels, 1, 1]),
                device,
            ),
            stride,
            padding,
            groups,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let y = conv2d(
            x,
            self.weight.clone(),
            self.bias.clone(),
            ConvOptions::new(
                [self.stride, self.stride],
                [self.padding, self.padding],
                [1, 1],
                self.groups,
            ),
        );
        y.mul(self.scale.clone()).add(self.shift.clone())
    }
}

/// A convolution with a bias and no normalization: the collapsed form of a
/// rep-block, which is how these are published.
struct ConvBias<B: Backend> {
    weight: Tensor<B, 4>,
    bias: Tensor<B, 1>,
}

impl<B: Backend> ConvBias<B> {
    fn load(
        artifact: &Artifact,
        device: &B::Device,
        index: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            weight: weight(
                artifact,
                device,
                &format!("conv2d_{index}.w_0"),
                [MODEL, MODEL, 3, 3],
            )?,
            bias: weight(
                artifact,
                device,
                &format!("conv2d_{index}.b_0"),
                [MODEL],
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        activation::silu(conv2d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
        ))
    }
}

/// A fully connected layer. Paddle stores `[in, out]`, which is the layout
/// burn's `linear` wants, so the weight is used as it comes off disk.
struct Linear<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> Linear<B> {
    fn load(
        artifact: &Artifact,
        device: &B::Device,
        index: usize,
        from: usize,
        to: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            weight: weight(
                artifact,
                device,
                &format!("linear_{index}.w_0"),
                [from, to],
            )?,
            bias: Some(weight(
                artifact,
                device,
                &format!("linear_{index}.b_0"),
                [to],
            )?),
        })
    }

    fn load_copy(
        artifact: &Artifact,
        device: &B::Device,
        index: usize,
        at: usize,
        from: usize,
        to: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            weight: copy(
                artifact,
                device,
                &format!("linear_{index}.w_0"),
                at,
                [from, to],
            )?,
            bias: Some(copy(
                artifact,
                device,
                &format!("linear_{index}.b_0"),
                at,
                [to],
            )?),
        })
    }

    fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        linear(x, self.weight.clone(), self.bias.clone())
    }
}

/// Layer normalization over the last axis, with the biased variance and the
/// epsilon inside the square root — the form every published graph carries.
struct Norm<B: Backend> {
    scale: Tensor<B, 2>,
    shift: Tensor<B, 2>,
}

impl<B: Backend> Norm<B> {
    fn load(
        artifact: &Artifact,
        device: &B::Device,
        index: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            scale: weight::<B, 1>(
                artifact,
                device,
                &format!("layer_norm_{index}.w_0"),
                [MODEL],
            )?
            .reshape([1, MODEL]),
            shift: weight::<B, 1>(
                artifact,
                device,
                &format!("layer_norm_{index}.b_0"),
                [MODEL],
            )?
            .reshape([1, MODEL]),
        })
    }

    fn load_copy(
        artifact: &Artifact,
        device: &B::Device,
        index: usize,
        at: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            scale: copy::<B, 1>(
                artifact,
                device,
                &format!("layer_norm_{index}.w_0"),
                at,
                [MODEL],
            )?
            .reshape([1, MODEL]),
            shift: copy::<B, 1>(
                artifact,
                device,
                &format!("layer_norm_{index}.b_0"),
                at,
                [MODEL],
            )?
            .reshape([1, MODEL]),
        })
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mean = x.clone().mean_dim(1);
        let centered = x.sub(mean);
        let variance = centered.clone().powi_scalar(2).mean_dim(1);
        let normed = centered.div(variance.add_scalar(NORM_EPS).sqrt());
        normed.mul(self.scale.clone()).add(self.shift.clone())
    }
}

/// Multi-head self-attention with the fused `qkv` projection Paddle publishes.
struct Attention<B: Backend> {
    qkv: Tensor<B, 2>,
    qkv_bias: Tensor<B, 1>,
    out: Linear<B>,
}

impl<B: Backend> Attention<B> {
    fn forward(
        &self,
        x: Tensor<B, 2>,
        positions: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 2> {
        let tokens = x.dims()[0];
        let head = MODEL / HEADS;
        let queried = match positions {
            None => x.clone(),
            Some(positions) => x.clone().add(positions),
        };
        let part = |source: Tensor<B, 2>, at: usize| -> Tensor<B, 3> {
            let weight = self
                .qkv
                .clone()
                .slice([0..MODEL, at * MODEL..(at + 1) * MODEL]);
            let bias =
                self.qkv_bias.clone().slice([at * MODEL..(at + 1) * MODEL]);
            linear(source, weight, Some(bias))
                .reshape([tokens, HEADS, head])
                .swap_dims(0, 1)
        };
        let q = part(queried.clone(), 0);
        let k = part(queried, 1);
        let v = part(x, 2);

        let scores = q
            .matmul(k.swap_dims(1, 2))
            .mul_scalar((head as f64).powf(-0.5));
        let weights = activation::softmax(scores, 2);
        let joined = weights.matmul(v).swap_dims(0, 1).reshape([tokens, MODEL]);
        self.out.forward(joined)
    }
}

/// The two- or three-layer perceptron this graph spells every small head with.
struct Mlp<B: Backend> {
    layers: Vec<Linear<B>>,
}

impl<B: Backend> Mlp<B> {
    fn load(
        artifact: &Artifact,
        device: &B::Device,
        first: usize,
        widths: &[usize],
    ) -> Result<Self, OcrError> {
        let mut layers = Vec::with_capacity(widths.len() - 1);
        for at in 0..widths.len() - 1 {
            layers.push(Linear::load(
                artifact,
                device,
                first + at,
                widths[at],
                widths[at + 1],
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut state = x;
        for (at, layer) in self.layers.iter().enumerate() {
            state = layer.forward(state);
            if at + 1 < self.layers.len() {
                state = activation::relu(state);
            }
        }
        state
    }
}

/// One layer of a stage's block.
enum HgLayer<B: Backend> {
    Plain(ConvBn<B>),
    Light { point: ConvBn<B>, depth: ConvBn<B> },
}

impl<B: Backend> HgLayer<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::Plain(conv) => activation::relu(conv.forward(x)),
            Self::Light { point, depth } => {
                activation::relu(depth.forward(point.forward(x)))
            },
        }
    }
}

struct HgBlock<B: Backend> {
    layers: Vec<HgLayer<B>>,
    squeeze: ConvBn<B>,
    excite: ConvBn<B>,
    residual: bool,
}

impl<B: Backend> HgBlock<B> {
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut kept = Vec::with_capacity(self.layers.len() + 1);
        kept.push(x.clone());
        let mut state = x.clone();
        for layer in &self.layers {
            state = layer.forward(state);
            kept.push(state.clone());
        }
        let joined = Tensor::cat(kept, 1);
        let y = activation::relu(self.squeeze.forward(joined));
        let y = activation::relu(self.excite.forward(y));
        if self.residual { y.add(x) } else { y }
    }
}

struct HgStage<B: Backend> {
    downsample: Option<ConvBn<B>>,
    blocks: Vec<HgBlock<B>>,
}

/// The whole network, on one backend.
pub struct Model<B: Backend> {
    device: B::Device,
    // Backbone.
    stem: [ConvBn<B>; 5],
    stages: Vec<HgStage<B>>,
    // Encoder.
    project: Vec<ConvBn<B>>,
    encoder_attention: Attention<B>,
    encoder_norms: [Norm<B>; 2],
    encoder_up: Linear<B>,
    encoder_down: Linear<B>,
    encoder_positions: Tensor<B, 2>,
    lateral: Vec<ConvBn<B>>,
    fpn: Vec<CspRep<B>>,
    downsample: Vec<ConvBn<B>>,
    pan: Vec<CspRep<B>>,
    // Decoder.
    decoder_project: Vec<ConvBn<B>>,
    anchors: Tensor<B, 2>,
    valid: Tensor<B, 2>,
    memory_project: Linear<B>,
    memory_norm: Norm<B>,
    memory_score: Linear<B>,
    memory_box: Mlp<B>,
    query_positions: Mlp<B>,
    layers: Vec<DecoderLayer<B>>,
    score: Linear<B>,
    classes: usize,
}

struct CspRep<B: Backend> {
    left: ConvBn<B>,
    blocks: Vec<ConvBias<B>>,
    right: ConvBn<B>,
}

impl<B: Backend> CspRep<B> {
    fn load(
        artifact: &Artifact,
        device: &B::Device,
        left: usize,
        blocks: usize,
        right: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            left: ConvBn::load(
                artifact,
                device,
                left,
                [MODEL, 2 * MODEL, 1, 1],
                1,
                0,
                1,
            )?,
            blocks: (0..3)
                .map(|at| ConvBias::load(artifact, device, blocks + at))
                .collect::<Result<_, _>>()?,
            right: ConvBn::load(
                artifact,
                device,
                right,
                [MODEL, 2 * MODEL, 1, 1],
                1,
                0,
                1,
            )?,
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut left = activation::silu(self.left.forward(x.clone()));
        for block in &self.blocks {
            left = block.forward(left);
        }
        left.add(activation::silu(self.right.forward(x)))
    }
}

struct DecoderLayer<B: Backend> {
    attention: Attention<B>,
    attention_norm: Norm<B>,
    offsets: Linear<B>,
    weights: Linear<B>,
    value: Linear<B>,
    sampled_out: Linear<B>,
    cross_norm: Norm<B>,
    up: Linear<B>,
    down: Linear<B>,
    final_norm: Norm<B>,
    box_head: Mlp<B>,
}

impl<B: Backend> DecoderLayer<B> {
    fn load(
        artifact: &Artifact,
        device: &B::Device,
        at: usize,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            attention: Attention {
                qkv: copy(
                    artifact,
                    device,
                    "multi_head_attention_1.w_0",
                    at,
                    [MODEL, 3 * MODEL],
                )?,
                qkv_bias: copy(
                    artifact,
                    device,
                    "multi_head_attention_1.b_0",
                    at,
                    [3 * MODEL],
                )?,
                out: Linear::load_copy(artifact, device, 3, at, MODEL, MODEL)?,
            },
            attention_norm: Norm::load_copy(artifact, device, 2, at)?,
            offsets: Linear::load_copy(
                artifact,
                device,
                4,
                at,
                MODEL,
                HEADS * LEVELS * POINTS * 2,
            )?,
            weights: Linear::load_copy(
                artifact,
                device,
                5,
                at,
                MODEL,
                HEADS * LEVELS * POINTS,
            )?,
            value: Linear::load_copy(artifact, device, 6, at, MODEL, MODEL)?,
            sampled_out: Linear::load_copy(
                artifact, device, 7, at, MODEL, MODEL,
            )?,
            cross_norm: Norm::load_copy(artifact, device, 3, at)?,
            up: Linear::load_copy(artifact, device, 8, at, MODEL, FFN)?,
            down: Linear::load_copy(artifact, device, 9, at, FFN, MODEL)?,
            final_norm: Norm::load_copy(artifact, device, 4, at)?,
            box_head: Mlp::load(
                artifact,
                device,
                23 + 3 * at,
                &[MODEL, MODEL, MODEL, 4],
            )?,
        })
    }
}

/// The stages of `PPHGNetV2-L`, as in the candle runtime.
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

impl<B: Backend> Model<B> {
    fn load(
        artifact: &Artifact,
        device: &B::Device,
        side: usize,
        classes: usize,
    ) -> Result<Self, OcrError> {
        let coarsest = side / 32;
        let positions =
            (side / 8).pow(2) + (side / 16).pow(2) + coarsest.pow(2);

        let mut next = 0usize;
        let mut conv = |dims: [usize; 4],
                        stride,
                        padding,
                        groups|
         -> Result<ConvBn<B>, OcrError> {
            let at = next;
            next += 1;
            ConvBn::load(artifact, device, at, dims, stride, padding, groups)
        };
        let stem = [
            conv([32, 3, 3, 3], 2, 1, 1)?,
            conv([16, 32, 2, 2], 1, 0, 1)?,
            conv([32, 16, 2, 2], 1, 0, 1)?,
            conv([32, 64, 3, 3], 2, 1, 1)?,
            conv([48, 32, 1, 1], 1, 0, 1)?,
        ];

        let mut stages = Vec::with_capacity(STAGES.len());
        for spec in &STAGES {
            let downsample = if spec.downsample {
                Some(conv([spec.in_channels, 1, 3, 3], 2, 1, spec.in_channels)?)
            } else {
                None
            };
            let mut blocks = Vec::with_capacity(spec.blocks);
            for at in 0..spec.blocks {
                let from = if at == 0 { spec.in_channels } else { spec.out };
                let pad = (spec.kernel - 1) / 2;
                let mut layers = Vec::with_capacity(6);
                for layer in 0..6 {
                    let source = if layer == 0 { from } else { spec.mid };
                    layers.push(if spec.light {
                        HgLayer::Light {
                            point: conv([spec.mid, source, 1, 1], 1, 0, 1)?,
                            depth: conv(
                                [spec.mid, 1, spec.kernel, spec.kernel],
                                1,
                                pad,
                                spec.mid,
                            )?,
                        }
                    } else {
                        HgLayer::Plain(conv(
                            [spec.mid, source, spec.kernel, spec.kernel],
                            1,
                            pad,
                            1,
                        )?)
                    });
                }
                let total = from + 6 * spec.mid;
                blocks.push(HgBlock {
                    layers,
                    squeeze: conv([spec.out / 2, total, 1, 1], 1, 0, 1)?,
                    excite: conv([spec.out, spec.out / 2, 1, 1], 1, 0, 1)?,
                    residual: at != 0,
                });
            }
            stages.push(HgStage { downsample, blocks });
        }
        drop(conv);

        let plain = |index: usize,
                     dims: [usize; 4],
                     stride,
                     padding|
         -> Result<ConvBn<B>, OcrError> {
            ConvBn::load(artifact, device, index, dims, stride, padding, 1)
        };
        let tokens = coarsest * coarsest;

        let mut layers = Vec::with_capacity(DEPTH);
        for at in 0..DEPTH {
            layers.push(DecoderLayer::load(artifact, device, at)?);
        }

        Ok(Self {
            device: device.clone(),
            stem,
            stages,
            project: vec![
                plain(80, [MODEL, 512, 1, 1], 1, 0)?,
                plain(81, [MODEL, 1024, 1, 1], 1, 0)?,
                plain(82, [MODEL, 2048, 1, 1], 1, 0)?,
            ],
            encoder_attention: Attention {
                qkv: weight(
                    artifact,
                    device,
                    "multi_head_attention_0.w_0",
                    [MODEL, 3 * MODEL],
                )?,
                qkv_bias: weight(
                    artifact,
                    device,
                    "multi_head_attention_0.b_0",
                    [3 * MODEL],
                )?,
                out: Linear::load(artifact, device, 0, MODEL, MODEL)?,
            },
            encoder_norms: [
                Norm::load(artifact, device, 0)?,
                Norm::load(artifact, device, 1)?,
            ],
            encoder_up: Linear::load(artifact, device, 1, MODEL, FFN)?,
            encoder_down: Linear::load(artifact, device, 2, FFN, MODEL)?,
            encoder_positions: weight::<B, 3>(
                artifact,
                device,
                "eager_tmp_0",
                [1, tokens, MODEL],
            )?
            .reshape([tokens, MODEL]),
            lateral: vec![
                plain(83, [MODEL, MODEL, 1, 1], 1, 0)?,
                plain(92, [MODEL, MODEL, 1, 1], 1, 0)?,
            ],
            fpn: vec![
                CspRep::load(artifact, device, 84, 122, 85)?,
                CspRep::load(artifact, device, 93, 125, 94)?,
            ],
            downsample: vec![
                plain(101, [MODEL, MODEL, 3, 3], 2, 1)?,
                plain(110, [MODEL, MODEL, 3, 3], 2, 1)?,
            ],
            pan: vec![
                CspRep::load(artifact, device, 102, 128, 103)?,
                CspRep::load(artifact, device, 111, 131, 112)?,
            ],
            decoder_project: vec![
                plain(119, [MODEL, MODEL, 1, 1], 1, 0)?,
                plain(120, [MODEL, MODEL, 1, 1], 1, 0)?,
                plain(121, [MODEL, MODEL, 1, 1], 1, 0)?,
            ],
            anchors: weight::<B, 3>(
                artifact,
                device,
                "eager_tmp_1",
                [1, positions, 4],
            )?
            .reshape([positions, 4]),
            valid: weight::<B, 3>(
                artifact,
                device,
                "eager_tmp_2",
                [1, positions, 1],
            )?
            .reshape([positions, 1]),
            memory_project: Linear::load(artifact, device, 12, MODEL, MODEL)?,
            memory_norm: Norm::load(artifact, device, 5)?,
            memory_score: Linear::load(artifact, device, 13, MODEL, classes)?,
            memory_box: Mlp::load(
                artifact,
                device,
                14,
                &[MODEL, MODEL, MODEL, 4],
            )?,
            query_positions: Mlp::load(
                artifact,
                device,
                10,
                &[4, 2 * MODEL, MODEL],
            )?,
            layers,
            score: Linear::load(artifact, device, 22, MODEL, classes)?,
            classes,
        })
    }

    fn forward(
        &self,
        data: &[f32],
        side: usize,
    ) -> Result<Prediction, OcrError> {
        let input: Tensor<B, 4> = Tensor::from_data(
            TensorData::new(data.to_vec(), [1, 3, side, side]),
            &self.device,
        );
        let features = self.backbone(input);
        let maps = self.encoder(features);
        self.decoder(maps)
    }

    fn backbone(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let y = activation::relu(self.stem[0].forward(x));
        let padded = pad_end(y);
        let left = activation::relu(self.stem[1].forward(padded.clone()));
        let left = activation::relu(self.stem[2].forward(pad_end(left)));
        // Kernel two, stride one, on a tensor already padded on the far edges:
        // the `SAME` pooling of an even kernel, which pads one side only.
        let right = max_pool2d(padded, [2, 2], [1, 1], [0, 0], [1, 1], false);
        let y = Tensor::cat(vec![right, left], 1);
        let y = activation::relu(self.stem[3].forward(y));
        let mut state = activation::relu(self.stem[4].forward(y));

        let mut out = Vec::with_capacity(LEVELS);
        for (at, stage) in self.stages.iter().enumerate() {
            if let Some(down) = &stage.downsample {
                state = down.forward(state);
            }
            for block in &stage.blocks {
                state = block.forward(state);
            }
            if at > 0 {
                out.push(state.clone());
            }
        }
        out
    }

    fn encoder(&self, features: Vec<Tensor<B, 4>>) -> Vec<Tensor<B, 4>> {
        let mut levels: Vec<Tensor<B, 4>> = features
            .into_iter()
            .zip(&self.project)
            .map(|(feature, project)| project.forward(feature))
            .collect();

        let coarsest = levels.len() - 1;
        levels[coarsest] = {
            let map = levels[coarsest].clone();
            let [_, channels, height, width] = map.dims();
            let tokens =
                map.reshape([channels, height * width]).swap_dims(0, 1);
            let attended = self
                .encoder_attention
                .forward(tokens.clone(), Some(self.encoder_positions.clone()));
            let x = self.encoder_norms[0].forward(tokens.add(attended));
            let up = activation::gelu(self.encoder_up.forward(x.clone()));
            let down = self.encoder_down.forward(up);
            let x = self.encoder_norms[1].forward(x.add(down));
            x.swap_dims(0, 1).reshape([1, channels, height, width])
        };

        let mut top_down = vec![levels[coarsest].clone()];
        for at in 0..2 {
            let below = levels[coarsest - at - 1].clone();
            let top = activation::silu(
                self.lateral[at]
                    .forward(top_down.last().expect("a rung").clone()),
            );
            *top_down.last_mut().expect("a rung") = top.clone();
            let [_, _, height, width] = top.dims();
            let up = interpolate(
                top,
                [height * 2, width * 2],
                InterpolateOptions::new(InterpolateMode::Nearest),
            );
            top_down
                .push(self.fpn[at].forward(Tensor::cat(vec![up, below], 1)));
        }
        top_down.reverse();

        let mut out = vec![top_down[0].clone()];
        for at in 0..2 {
            let below = out.last().expect("a rung").clone();
            let down = activation::silu(self.downsample[at].forward(below));
            let joined = Tensor::cat(vec![down, top_down[at + 1].clone()], 1);
            out.push(self.pan[at].forward(joined));
        }
        out
    }

    fn decoder(&self, maps: Vec<Tensor<B, 4>>) -> Result<Prediction, OcrError> {
        let mut levels = Vec::with_capacity(LEVELS);
        let mut flattened = Vec::with_capacity(LEVELS);
        let mut start = 0usize;
        for (map, project) in maps.into_iter().zip(&self.decoder_project) {
            let projected = project.forward(map);
            let [_, channels, height, width] = projected.dims();
            flattened.push(
                projected
                    .reshape([channels, height * width])
                    .swap_dims(0, 1),
            );
            levels.push(Level {
                height,
                width,
                start,
            });
            start += height * width;
        }
        let memory = Tensor::cat(flattened, 0).mul(self.valid.clone());

        let projected = self
            .memory_norm
            .forward(self.memory_project.forward(memory.clone()));
        let scores = self.memory_score.forward(projected.clone());
        let boxes = self
            .memory_box
            .forward(projected.clone())
            .add(self.anchors.clone());

        // Query selection on the host: three hundred out of thirteen thousand
        // is a sort, and the result is needed as indices anyway.
        let best: Vec<f32> =
            scores.max_dim(1).into_data().into_vec().map_err(|e| {
                OcrError::Runtime(format!("reading scores: {e:?}"))
            })?;
        let mut order: Vec<i32> = (0..best.len() as i32).collect();
        order.sort_by(|a, b| best[*b as usize].total_cmp(&best[*a as usize]));
        order.truncate(QUERIES);
        let picked: Tensor<B, 1, Int> =
            Tensor::from_data(TensorData::new(order, [QUERIES]), &self.device);

        let mut reference =
            activation::sigmoid(boxes.select(0, picked.clone()));
        let mut state = projected.select(0, picked);

        for layer in &self.layers {
            let positions = self.query_positions.forward(reference.clone());
            let attended = layer
                .attention
                .forward(state.clone(), Some(positions.clone()));
            state = layer.attention_norm.forward(state.add(attended));

            let sampled = self.deformable(
                layer,
                state.clone(),
                positions,
                memory.clone(),
                reference.clone(),
                &levels,
            )?;
            state = layer.cross_norm.forward(state.add(sampled));

            let up = activation::relu(layer.up.forward(state.clone()));
            let down = layer.down.forward(up);
            state = layer.final_norm.forward(state.add(down));

            let delta = layer.box_head.forward(state.clone());
            reference =
                activation::sigmoid(delta.add(inverse_sigmoid(reference)));
        }

        let logits = self
            .score
            .forward(state)
            .into_data()
            .into_vec()
            .map_err(|e| OcrError::Runtime(format!("reading logits: {e:?}")))?;
        let boxes = reference
            .into_data()
            .into_vec()
            .map_err(|e| OcrError::Runtime(format!("reading boxes: {e:?}")))?;
        Ok(Prediction {
            logits,
            boxes,
            classes: self.classes,
        })
    }

    /// Multi-scale deformable attention: four points per head per level, in a
    /// window the size of the query's own box.
    fn deformable(
        &self,
        layer: &DecoderLayer<B>,
        state: Tensor<B, 2>,
        positions: Tensor<B, 2>,
        memory: Tensor<B, 2>,
        reference: Tensor<B, 2>,
        levels: &[Level],
    ) -> Result<Tensor<B, 2>, OcrError> {
        let head = MODEL / HEADS;
        let samples = QUERIES * POINTS;
        let queried = state.add(positions);
        let value = layer.value.forward(memory);

        let offsets = layer
            .offsets
            .forward(queried.clone())
            .reshape([QUERIES, HEADS, LEVELS, POINTS, 2]);
        let weights = activation::softmax(
            layer.weights.forward(queried).reshape([
                QUERIES,
                HEADS,
                LEVELS * POINTS,
            ]),
            2,
        )
        .reshape([QUERIES, HEADS, LEVELS, POINTS]);

        let centre = reference
            .clone()
            .slice([0..QUERIES, 0..2])
            .reshape([QUERIES, 1, 1, 1, 2]);
        let size = reference
            .slice([0..QUERIES, 2..4])
            .reshape([QUERIES, 1, 1, 1, 2]);
        let scaled = offsets
            .mul(size.mul_scalar(0.5 / POINTS as f64))
            .add(centre);

        let mut total: Option<Tensor<B, 3>> = None;
        for (at, level) in levels.iter().enumerate() {
            let count = level.count();
            let map = value
                .clone()
                .slice([level.start..level.start + count, 0..MODEL])
                .reshape([count, HEADS, head])
                .swap_dims(0, 1)
                .reshape([HEADS * count, head]);
            let grid = scaled
                .clone()
                .slice([0..QUERIES, 0..HEADS, at..at + 1, 0..POINTS, 0..2])
                .reshape([QUERIES, HEADS, POINTS, 2])
                .swap_dims(0, 1)
                .reshape([HEADS * samples, 2]);
            let sampled = self.sample(map, grid, level, count, head)?;
            let weight = weights
                .clone()
                .slice([0..QUERIES, 0..HEADS, at..at + 1, 0..POINTS])
                .reshape([QUERIES, HEADS, POINTS])
                .swap_dims(0, 1)
                .reshape([HEADS, samples, 1]);
            let weighted = sampled
                .mul(weight)
                .reshape([HEADS, QUERIES, POINTS, head])
                .sum_dim(2)
                .reshape([HEADS, QUERIES, head]);
            total = Some(match total {
                None => weighted,
                Some(sum) => sum.add(weighted),
            });
        }
        let joined = total
            .expect("three levels")
            .swap_dims(0, 1)
            .reshape([QUERIES, MODEL]);
        Ok(layer.sampled_out.forward(joined))
    }

    /// Bilinear sampling, four gathers and a blend. The corner arithmetic is
    /// the shared host one, so the two runtimes cannot round it differently.
    fn sample(
        &self,
        map: Tensor<B, 2>,
        grid: Tensor<B, 2>,
        level: &Level,
        count: usize,
        head: usize,
    ) -> Result<Tensor<B, 3>, OcrError> {
        let rows = grid.dims()[0];
        let samples = rows / HEADS;
        let grid: Vec<f32> = grid
            .into_data()
            .into_vec()
            .map_err(|e| OcrError::Runtime(format!("reading a grid: {e:?}")))?;
        let (indices, blend) = sampling::corners(&grid, level, count, HEADS);

        let mut total: Option<Tensor<B, 2>> = None;
        for corner in 0..4 {
            let index: Tensor<B, 1, Int> = Tensor::from_data(
                TensorData::new(
                    indices[corner * rows..(corner + 1) * rows]
                        .iter()
                        .map(|at| *at as i32)
                        .collect::<Vec<_>>(),
                    [rows],
                ),
                &self.device,
            );
            let weight: Tensor<B, 2> = Tensor::from_data(
                TensorData::new(
                    blend[corner * rows..(corner + 1) * rows].to_vec(),
                    [rows, 1],
                ),
                &self.device,
            );
            let gathered = map.clone().select(0, index).mul(weight);
            total = Some(match total {
                None => gathered,
                Some(sum) => sum.add(gathered),
            });
        }
        Ok(total.expect("four corners").reshape([HEADS, samples, head]))
    }
}

/// Pads the right and bottom edges with zeros.
fn pad_end<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, channels, height, width] = x.dims();
    let device = x.device();
    let right: Tensor<B, 4> =
        Tensor::zeros([batch, channels, height, 1], &device);
    let x = Tensor::cat(vec![x, right], 3);
    let bottom: Tensor<B, 4> =
        Tensor::zeros([batch, channels, 1, width + 1], &device);
    Tensor::cat(vec![x, bottom], 2)
}

/// `log(x / (1 - x))`, clamped the way the reference clamps it.
fn inverse_sigmoid<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    const EPS: f32 = 1e-5;
    let x = x.clamp(0.0, 1.0);
    let low = x.clone().clamp_min(EPS);
    let high = x.neg().add_scalar(1.0).clamp_min(EPS);
    low.div(high).log()
}
