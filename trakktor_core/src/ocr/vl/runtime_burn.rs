//! The burn-backed runtime.
//!
//! An alternative [`VlModel`] implementation on [burn](burn), selectable at
//! run time next to the candle one. Backends: ndarray on the CPU (f32 —
//! the ndarray backend has no half-precision element), and wgpu with
//! MSL-compiled kernels on Metal, with operator fusion and autotuning.
//!
//! **This runtime computes in f32 only**, on either device — on Metal too,
//! where the candle runtime runs at f16. Not the port's choice: on the f16
//! backend (burn 0.21 / cubecl 0.10) the operator-fusion engine's kernel
//! planner panics on this model's cast-mixed elementwise chains
//! (`burn-cubecl-fusion .../codegen/ir.rs` `resolve_arg`, an index out of
//! bounds, after which the fusion stream is poisoned and every later call
//! fails), with the tensors' f32 pins removed just the same. The f32 path
//! compiles and runs the same chains cleanly. Half precision here is memory,
//! not correctness — the parity contract is stated in f32 — so it waits for
//! an upstream fix rather than a workaround.
//!
//! The checkpoint is the same published directory the candle runtime loads:
//! tensors are read through candle's safetensors reader and converted through
//! f32 into tensors of the target backend. The host-side tables — the
//! interpolated position grid and the rotary angles — come from the same
//! shared [`tables`](super::tables) module the candle runtime uses, computed
//! once on the host so the two runtimes cannot drift there; the cosine and
//! sine are likewise taken on the host, so every backend starts from
//! bit-identical rows.
//!
//! The backend choice is erased behind a boxed [`VlModel`], so the driver
//! sees one type.
//!
//! burn operations panic on shape mismatches instead of returning errors, so
//! every weight is shape-checked against the checkpoint at load time; a panic
//! past loading is a bug, not a data condition.

pub mod ernie;
#[cfg(test)]
mod tests;
pub mod vision;

use std::path::Path;

use burn::{
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        wgpu::{Metal, WgpuDevice},
    },
    tensor::{DType, Int, Tensor, TensorData, backend::Backend},
};

use self::{
    ernie::{Cache, Decoder},
    vision::{Projector, Tower},
};
use super::{
    config::{ModelConfig, WEIGHTS_FILE},
    generate::VlModel,
    image::Prepared,
    tables,
};
use crate::ocr::error::OcrError;

/// Maps a backend or checkpoint failure onto the domain's model error.
pub(super) fn model_err(context: &str, e: impl std::fmt::Display) -> OcrError {
    OcrError::Artifact(format!("{context}: {e}"))
}

/// Lazy access to the checkpoint's tensors through candle's safetensors
/// reader: every weight is read on demand and handed over as f32 values plus
/// its shape.
pub(super) struct Weights(candle_core::safetensors::MmapedSafetensors);

impl Weights {
    /// Opens the weights file in a checkpoint directory.
    fn open(dir: &Path) -> Result<Self, OcrError> {
        let path = dir.join(WEIGHTS_FILE);
        if !path.is_file() {
            return Err(OcrError::Artifact(format!(
                "no {WEIGHTS_FILE} in {}",
                dir.display()
            )));
        }
        // SAFETY: the weights are memory-mapped read-only and must not change
        // while mapped; they are ours, under the model directory, and
        // verified against a digest when they were fetched.
        let inner =
            unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }
                .map_err(|e| model_err(&path.display().to_string(), e))?;
        Ok(Self(inner))
    }

    /// The named tensor as f32 values and its shape.
    pub(super) fn parts(
        &self,
        key: &str,
    ) -> Result<(Vec<f32>, Vec<usize>), OcrError> {
        let tensor = self
            .0
            .load(key, &candle_core::Device::Cpu)
            .map_err(|e| model_err(key, e))?;
        let dims = tensor.dims().to_vec();
        let values = tensor
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| model_err(key, e))?;
        Ok((values, dims))
    }
}

/// Reads a checkpoint tensor of the given shape as a burn tensor.
pub(super) fn weight<B: Backend, const D: usize>(
    weights: &Weights,
    device: &B::Device,
    key: &str,
    shape: [usize; D],
) -> Result<Tensor<B, D>, OcrError> {
    let (values, dims) = weights.parts(key)?;
    if dims != shape {
        return Err(model_err(
            key,
            format!("shape {dims:?}, expected {shape:?}"),
        ));
    }
    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

/// Reads an `[out, in]` checkpoint matrix as row-major values, shape-checked.
pub(super) fn rows_of(
    weights: &Weights,
    key: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<Vec<f32>, OcrError> {
    let (values, dims) = weights.parts(key)?;
    if dims != [out_dim, in_dim] {
        return Err(model_err(
            key,
            format!("shape {dims:?}, expected {:?}", [out_dim, in_dim]),
        ));
    }
    Ok(values)
}

/// Reads an `[out, in]` checkpoint matrix as burn's `[in, out]` layout.
pub(super) fn matrix<B: Backend>(
    weights: &Weights,
    device: &B::Device,
    key: &str,
    out_dim: usize,
    in_dim: usize,
) -> Result<Tensor<B, 2>, OcrError> {
    let values = rows_of(weights, key, out_dim, in_dim)?;
    Ok(Tensor::from_data(
        TensorData::new(
            transpose_2d(&values, out_dim, in_dim),
            [in_dim, out_dim],
        ),
        device,
    ))
}

/// Blocked out-of-place transpose of a row-major `[rows, cols]` matrix.
pub(super) fn transpose_2d(
    values: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    const TILE: usize = 64;
    let mut out = vec![0.0f32; values.len()];
    for row0 in (0..rows).step_by(TILE) {
        for col0 in (0..cols).step_by(TILE) {
            for row in row0..(row0 + TILE).min(rows) {
                for col in col0..(col0 + TILE).min(cols) {
                    out[col * rows + row] = values[row * cols + col];
                }
            }
        }
    }
    out
}

/// Root-mean-square normalization with a learned per-channel gain.
///
/// Normalizes over the last axis, with the statistics in f32 and the result
/// cast back before the gain is applied — the same order candle's fused
/// kernel uses.
pub(super) struct RmsNorm<B: Backend> {
    weight: Tensor<B, 1>,
    size: usize,
    eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub(super) fn load(
        weights: &Weights,
        device: &B::Device,
        key: &str,
        size: usize,
        eps: f64,
    ) -> Result<Self, OcrError> {
        Ok(Self {
            weight: weight(weights, device, key, [size])?,
            size,
            eps,
        })
    }

    pub(super) fn forward<const D: usize>(
        &self,
        x: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let dtype = x.dtype();
        let x = x.cast(DType::F32);
        let variance = (x.clone() * x.clone()).mean_dim(D - 1);
        let normed = x / variance.add_scalar(self.eps).sqrt();
        let mut shape = [1usize; D];
        shape[D - 1] = self.size;
        normed.cast(dtype) * self.weight.clone().reshape(shape)
    }
}

/// Rotates the halves of the last axis: `[a, b] → [-b, a]`.
fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let half = x.dims()[3] / 2;
    let first = x.clone().narrow(3, 0, half);
    let second = x.narrow(3, half, half);
    Tensor::cat(vec![-second, first], 3)
}

/// Applies rotary positions to `x`, shaped `[batch, heads, seq, head_dim]`.
///
/// `cos` and `sin` are `[1, 1, seq, head_dim]` with the half-table laid
/// across both halves of the head — the same pairing of channel `i` with
/// channel `i + head_dim / 2` that candle's fused rope uses.
pub(super) fn apply_rope<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 4>,
    sin: &Tensor<B, 4>,
) -> Tensor<B, 4> {
    let rotated = rotate_half(x.clone());
    x * cos.clone() + rotated * sin.clone()
}

/// Repeats key/value heads so every query head has a partner.
pub(super) fn repeat_kv<B: Backend>(
    x: Tensor<B, 4>,
    groups: usize,
) -> Tensor<B, 4> {
    if groups == 1 {
        return x;
    }
    let [batch, heads, seq, dim] = x.dims();
    x.reshape([batch, heads, 1, seq, dim])
        .repeat_dim(2, groups)
        .reshape([batch, heads * groups, seq, dim])
}

/// The cosine and sine of the given angle rows, taken on the host and laid
/// across both halves of the head dimension, `[rows, dim]` each.
///
/// Taking them on the host keeps every backend on bit-identical tables — a
/// device cosine is whatever the backend's kernel makes it.
pub(super) fn host_tables(
    angles: &[f32],
    rows: usize,
    dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut cos = vec![0.0f32; rows * dim];
    let mut sin = vec![0.0f32; rows * dim];
    for row in 0..rows {
        for index in 0..half {
            let angle = angles[row * half + index];
            let (sine, cosine) = angle.sin_cos();
            cos[row * dim + index] = cosine;
            cos[row * dim + half + index] = cosine;
            sin[row * dim + index] = sine;
            sin[row * dim + half + index] = sine;
        }
    }
    (cos, sin)
}

/// Turns a list of token ids into `[1, ids]` indices on the device.
pub(super) fn indices<B: Backend>(
    ids: &[u32],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let ids: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
    let len = ids.len();
    Tensor::from_data(TensorData::new(ids, [1, len]), device)
}

/// One generation's state: the cache and the rotary rows, sized once for the
/// whole answer. How far the decode has advanced is the cache's fill.
struct Session<B: Backend> {
    cache: Cache<B>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
}

/// The networks running on one burn backend.
pub struct BurnModel<B: Backend> {
    device: B::Device,
    cfg: ModelConfig,
    tower: Tower<B>,
    projector: Projector<B>,
    decoder: Decoder<B>,
    session: Option<Session<B>>,
}

impl<B: Backend> BurnModel<B> {
    /// Loads the checkpoint onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`OcrError`] when the weights are missing, malformed, or do
    /// not match the declared geometry.
    pub fn load(
        dir: &Path,
        cfg: ModelConfig,
        device: B::Device,
    ) -> Result<Self, OcrError> {
        let weights = Weights::open(dir)?;
        let tower = Tower::load(&weights, &device, &cfg.vision)?;
        let projector =
            Projector::load(&weights, &device, &cfg.vision, cfg.hidden_size)?;
        let decoder = Decoder::load(&weights, &device, &cfg)?;
        Ok(Self {
            device,
            cfg,
            tower,
            projector,
            decoder,
            session: None,
        })
    }

    /// The probability row of the position the logits score. The softmax
    /// stays on the device; the row comes back in one transfer, and the wait
    /// is what a decode step's time is made of.
    fn probabilities(
        &self,
        logits: Tensor<B, 3>,
    ) -> Result<Vec<f32>, OcrError> {
        burn::tensor::activation::softmax(logits.cast(DType::F32), 2)
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| {
                OcrError::Runtime(format!("reading the logits: {e:?}"))
            })
    }
}

impl<B: Backend> VlModel for BurnModel<B> {
    fn prime(
        &mut self,
        picture: &Prepared,
        tokens: &[u32],
        image_at: usize,
        positions: &[[i64; 3]],
    ) -> Result<Vec<f32>, OcrError> {
        self.session = None;
        let places = picture.tokens(self.cfg.vision.spatial_merge_size);

        // 1. The picture, once: patches → tower → projector.
        let patches = picture.patches();
        let pixels = Tensor::<B, 2>::from_data(
            TensorData::new(
                picture.pixels.clone(),
                [patches, picture.pixels.len() / patches],
            ),
            &self.device,
        );
        let features = self.tower.forward(pixels, picture.grid);
        let vision = self.projector.forward(features, picture.grid);

        // 2. The picture's tokens take the placeholders' places.
        let embedded = self.decoder.embed(tokens, &self.device);
        let mut parts = Vec::with_capacity(3);
        if image_at > 0 {
            parts.push(embedded.clone().narrow(1, 0, image_at));
        }
        parts.push(vision.unsqueeze_dim(0));
        let after = image_at + places;
        if after < tokens.len() {
            parts.push(embedded.narrow(1, after, tokens.len() - after));
        }
        let inputs = Tensor::cat(parts, 1);

        // 3. Prefill. The cache and the rotary rows are sized once, up front,
        // for every position the answer can reach.
        let prompt = tokens.len();
        let capacity = positions.len();
        let angles = tables::decoder_angles(
            positions,
            &self.cfg.mrope_section,
            self.cfg.rope_theta,
            self.cfg.head_dim,
        );
        let (cos, sin) = host_tables(&angles, capacity, self.cfg.head_dim);
        let lift = |values: Vec<f32>| -> Tensor<B, 4> {
            Tensor::from_data(
                TensorData::new(values, [1, 1, capacity, self.cfg.head_dim]),
                &self.device,
            )
        };
        let (cos, sin) = (lift(cos), lift(sin));

        let mut cache = Cache::new(&self.cfg, capacity, &self.device);
        let logits = self.decoder.forward(
            inputs,
            cos.clone().narrow(2, 0, prompt),
            sin.clone().narrow(2, 0, prompt),
            &mut cache,
        );

        let row = self.probabilities(logits)?;
        self.session = Some(Session { cache, cos, sin });
        Ok(row)
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>, OcrError> {
        let Some(session) = self.session.as_mut() else {
            return Err(OcrError::Runtime(
                "a decode step before the generation was primed".into(),
            ));
        };
        let embedded = self.decoder.embed(&[token], &self.device);
        // The next row of the precomputed tables is exactly the cache's fill.
        let row = session.cache.len();
        let logits = self.decoder.forward(
            embedded,
            session.cos.clone().narrow(2, row, 1),
            session.sin.clone().narrow(2, row, 1),
            &mut session.cache,
        );
        self.probabilities(logits)
    }
}

/// Loads a checkpoint on the burn CPU backend (ndarray), which computes in
/// f32 — the same precision the candle runtime uses on the CPU.
///
/// # Errors
///
/// See [`BurnModel::load`].
pub fn load_cpu(
    dir: &Path,
    cfg: ModelConfig,
) -> Result<Box<dyn VlModel>, OcrError> {
    Ok(Box::new(BurnModel::<NdArray<f32>>::load(
        dir,
        cfg,
        NdArrayDevice::Cpu,
    )?))
}

/// Loads a checkpoint on the burn Metal backend (wgpu with MSL-compiled
/// kernels), in f32 — the one precision this runtime serves (see the module
/// notes on the f16 backend).
///
/// # Errors
///
/// See [`BurnModel::load`].
pub fn load_metal(
    dir: &Path,
    cfg: ModelConfig,
) -> Result<Box<dyn VlModel>, OcrError> {
    Ok(Box::new(BurnModel::<Metal<f32>>::load(
        dir,
        cfg,
        WgpuDevice::default(),
    )?))
}
