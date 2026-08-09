//! The flow model's velocity field on burn — the same network as
//! [the candle one](super::super::runtime::cfm), including the one thing that
//! is precomputed: the conditioning's thirty 1×1 convolutions, which do not
//! change while the solver runs.
//!
//! Ported from resemble-enhance (MIT).

use burn::tensor::{Tensor, activation, backend::Backend};

use super::{Weights, irmae::Conv};
use crate::enhance::{
    EnhanceError,
    resemble::config::{
        LATENT, MELS, TIME_EMB, TIME_EMB_MAX_EXPONENT, WN_DILATION_CYCLE,
        WN_HIDDEN, WN_KERNEL, WN_LAYERS,
    },
};

/// The epsilon of torch's `InstanceNorm1d`.
const INSTANCE_NORM_EPS: f64 = 1e-5;

/// One layer: a dilated convolution, gated, split into what continues and what
/// is collected.
struct Layer<B: Backend> {
    global: Conv<B>,
    local: Conv<B>,
    dilated: Conv<B>,
    out: Conv<B>,
}

impl<B: Backend> Layer<B> {
    fn load(
        weights: &Weights,
        device: &B::Device,
        prefix: &str,
        dilation: usize,
    ) -> Result<Self, EnhanceError> {
        Ok(Self {
            global: Conv::load(
                weights,
                device,
                &format!("{prefix}.gconv"),
                TIME_EMB,
                WN_HIDDEN,
                1,
                1,
                true,
            )?,
            local: Conv::load(
                weights,
                device,
                &format!("{prefix}.lconv"),
                MELS,
                2 * WN_HIDDEN,
                1,
                1,
                true,
            )?,
            dilated: Conv::load(
                weights,
                device,
                &format!("{prefix}.dconv"),
                WN_HIDDEN,
                2 * WN_HIDDEN,
                WN_KERNEL,
                dilation,
                true,
            )?,
            out: Conv::load(
                weights,
                device,
                &format!("{prefix}.out"),
                WN_HIDDEN,
                2 * WN_HIDDEN,
                1,
                1,
                true,
            )?,
        })
    }

    fn forward(
        &self,
        x: Tensor<B, 3>,
        local: &Tensor<B, 3>,
        global: &Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let hidden = x.clone() + self.global.forward(global.clone());
        let hidden = self.dilated.forward(hidden) + local.clone();
        let gated = activation::tanh(hidden.clone().narrow(1, 0, WN_HIDDEN)) *
            activation::sigmoid(hidden.narrow(1, WN_HIDDEN, WN_HIDDEN));
        let out = self.out.forward(gated);
        let carried = out.clone().narrow(1, 0, WN_HIDDEN);
        let collected = out.narrow(1, WN_HIDDEN, WN_HIDDEN);
        ((carried + x) / 2f64.sqrt(), collected)
    }
}

/// The conditioning, convolved once for every layer that will want it.
pub struct Conditioning<B: Backend> {
    local: Vec<Tensor<B, 3>>,
}

/// The velocity field.
pub struct Velocity<B: Backend> {
    start: Conv<B>,
    layers: Vec<Layer<B>>,
    end: Conv<B>,
}

impl<B: Backend> Velocity<B> {
    /// Loads it.
    ///
    /// # Errors
    ///
    /// Returns [`EnhanceError::Checkpoint`] when a weight is missing or has the
    /// wrong shape.
    pub fn load(
        weights: &Weights,
        device: &B::Device,
    ) -> Result<Self, EnhanceError> {
        let root = "lcfm.cfm.net";
        let mut layers = Vec::with_capacity(WN_LAYERS);
        for index in 0..WN_LAYERS {
            layers.push(Layer::load(
                weights,
                device,
                &format!("{root}.layers.{index}"),
                1 << (index % WN_DILATION_CYCLE),
            )?);
        }
        Ok(Self {
            start: Conv::load(
                weights,
                device,
                &format!("{root}.start"),
                LATENT,
                WN_HIDDEN,
                1,
                1,
                true,
            )?,
            layers,
            end: Conv::load(
                weights,
                device,
                &format!("{root}.end"),
                WN_HIDDEN,
                LATENT,
                1,
                1,
                true,
            )?,
        })
    }

    /// Normalizes the mel and convolves it for every layer, once per chunk.
    pub fn condition(&self, mel: Tensor<B, 3>) -> Conditioning<B> {
        let normed = instance_norm(mel);
        Conditioning {
            local: self
                .layers
                .iter()
                .map(|layer| layer.local.forward(normed.clone()))
                .collect(),
        }
    }

    /// The velocity at `point` and time `at`.
    pub fn forward(
        &self,
        point: Tensor<B, 3>,
        at: f64,
        conditioning: &Conditioning<B>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let global = embed::<B>(at, device);
        let mut hidden = self.start.forward(point);
        let mut collected: Option<Tensor<B, 3>> = None;
        for (layer, local) in self.layers.iter().zip(&conditioning.local) {
            let (next, skip) = layer.forward(hidden, local, &global);
            hidden = next;
            collected = Some(match collected {
                Some(sum) => sum + skip,
                None => skip,
            });
        }
        let collected = collected.expect("the network has at least one layer");
        self.end
            .forward(collected / (self.layers.len() as f64).sqrt())
    }
}

/// The sinusoidal embedding of one time, as `[1, TIME_EMB, 1]`.
fn embed<B: Backend>(at: f64, device: &B::Device) -> Tensor<B, 3> {
    let at = at.clamp(0.0, 1.0);
    let half = TIME_EMB / 2;
    let mut values = vec![0f32; TIME_EMB];
    for index in 0..half {
        let exponent =
            f64::from(TIME_EMB_MAX_EXPONENT) * index as f64 / (half - 1) as f64;
        let angle = at * 10f64.powf(exponent);
        values[index] = angle.sin() as f32;
        values[half + index] = angle.cos() as f32;
    }
    Tensor::from_data(
        burn::tensor::TensorData::new(values, [1, TIME_EMB, 1]),
        device,
    )
}

/// Normalization of every channel over time, without an affine — torch's
/// `InstanceNorm1d` as it is built by default.
fn instance_norm<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let mean = x.clone().mean_dim(2);
    let centered = x - mean;
    let variance = centered.clone().powi_scalar(2).mean_dim(2);
    centered / (variance + INSTANCE_NORM_EPS).sqrt()
}
