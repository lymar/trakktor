//! The flow model's velocity field on candle — thirty dilated gated layers,
//! and the most expensive thing in the pipeline.
//!
//! # What it is asked
//!
//! Given a point `ψ` in the latent, a time `t` in `[0, 1]` and the mel of the
//! recording, it answers "which way, and how fast". The solver
//! ([`solver`](super::super::solver)) integrates that from `t = 0` to `t = 1`
//! and the endpoint is the latent the decoder turns into a spectrum. With the
//! midpoint rule and the default budget that is thirty-two passes through this
//! network per chunk, which is where the time goes.
//!
//! # The one thing that is precomputed
//!
//! The mel enters every layer through a 1×1 convolution, and the mel does not
//! change while the solver runs. Those thirty convolutions are therefore done
//! **once per chunk** rather than once per evaluation — the same numbers, and
//! about four hundred gigaflops of them saved on a thirty-second chunk. The
//! time embedding cannot be cached the same way: `t` is what moves.
//!
//! Ported from resemble-enhance (MIT).

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};

use super::super::config::{
    LATENT, MELS, TIME_EMB, TIME_EMB_MAX_EXPONENT, WN_DILATION_CYCLE,
    WN_HIDDEN, WN_KERNEL, WN_LAYERS,
};

/// The epsilon of torch's `InstanceNorm1d`.
const INSTANCE_NORM_EPS: f64 = 1e-5;

/// One layer: a dilated convolution, gated, split into what continues and what
/// is collected.
#[derive(Debug)]
struct Layer {
    global: Conv1d,
    local: Conv1d,
    dilated: Conv1d,
    out: Conv1d,
}

impl Layer {
    fn load(dilation: usize, vb: VarBuilder) -> Result<Self> {
        let point = |inputs, outputs, name: &str| {
            candle_nn::conv1d(
                inputs,
                outputs,
                1,
                Conv1dConfig::default(),
                vb.pp(name),
            )
        };
        Ok(Self {
            global: point(TIME_EMB, WN_HIDDEN, "gconv")?,
            local: point(MELS, 2 * WN_HIDDEN, "lconv")?,
            dilated: candle_nn::conv1d(
                WN_HIDDEN,
                2 * WN_HIDDEN,
                WN_KERNEL,
                Conv1dConfig {
                    padding: dilation * (WN_KERNEL / 2),
                    dilation,
                    ..Default::default()
                },
                vb.pp("dconv"),
            )?,
            out: point(WN_HIDDEN, 2 * WN_HIDDEN, "out")?,
        })
    }

    /// Returns what continues and what is collected.
    ///
    /// `local` is this layer's share of the conditioning, already convolved.
    fn forward(
        &self,
        xs: &Tensor,
        local: &Tensor,
        global: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let hidden = xs.broadcast_add(&self.global.forward(global)?)?;
        let hidden = (self.dilated.forward(&hidden)? + local)?;
        let gated = (hidden.narrow(1, 0, WN_HIDDEN)?.tanh()? *
            candle_nn::ops::sigmoid(
                &hidden.narrow(1, WN_HIDDEN, WN_HIDDEN)?,
            )?)?;
        let out = self.out.forward(&gated)?;
        let carried = out.narrow(1, 0, WN_HIDDEN)?;
        let collected = out.narrow(1, WN_HIDDEN, WN_HIDDEN)?;
        // The residual is halved in energy rather than in amplitude, which is
        // what keeps thirty of them from growing without bound.
        Ok((((carried + xs)? / 2f64.sqrt())?, collected))
    }
}

/// The conditioning, convolved once for every layer that will want it.
#[derive(Debug)]
pub struct Conditioning {
    local: Vec<Tensor>,
}

/// The velocity field.
#[derive(Debug)]
pub struct Velocity {
    start: Conv1d,
    layers: Vec<Layer>,
    end: Conv1d,
    device: Device,
    dtype: DType,
}

impl Velocity {
    /// Loads it from a checkpoint rooted at `vb`.
    ///
    /// # Errors
    ///
    /// Returns whatever candle reports for a missing or misshapen tensor.
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(WN_LAYERS);
        for index in 0..WN_LAYERS {
            layers.push(Layer::load(
                1 << (index % WN_DILATION_CYCLE),
                vb.pp(format!("layers.{index}")),
            )?);
        }
        Ok(Self {
            start: candle_nn::conv1d(
                LATENT,
                WN_HIDDEN,
                1,
                Conv1dConfig::default(),
                vb.pp("start"),
            )?,
            layers,
            end: candle_nn::conv1d(
                WN_HIDDEN,
                LATENT,
                1,
                Conv1dConfig::default(),
                vb.pp("end"),
            )?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Normalizes the mel and convolves it for every layer, once per chunk.
    ///
    /// # Errors
    ///
    /// Returns whatever candle reports for a failed operation.
    pub fn condition(&self, mel: &Tensor) -> Result<Conditioning> {
        let normed = instance_norm(mel)?;
        let mut local = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            local.push(layer.local.forward(&normed)?);
        }
        Ok(Conditioning { local })
    }

    /// The velocity at `point` and time `at`.
    ///
    /// # Errors
    ///
    /// Returns whatever candle reports for a failed operation.
    pub fn forward(
        &self,
        point: &Tensor,
        at: f64,
        conditioning: &Conditioning,
    ) -> Result<Tensor> {
        let global = self.embed(at)?;
        let mut hidden = self.start.forward(point)?;
        let mut collected: Option<Tensor> = None;
        for (layer, local) in self.layers.iter().zip(&conditioning.local) {
            let (next, skip) = layer.forward(&hidden, local, &global)?;
            hidden = next;
            collected = Some(match collected {
                Some(sum) => (sum + skip)?,
                None => skip,
            });
        }
        let collected = collected.expect("the network has at least one layer");
        let scaled = (collected / (self.layers.len() as f64).sqrt())?;
        self.end.forward(&scaled)
    }

    /// The sinusoidal embedding of one time, as `[1, TIME_EMB, 1]`.
    ///
    /// Small enough to build on the host: sixty-four frequencies spaced evenly
    /// in the **exponent**, from one to ten thousand, and the sine and cosine
    /// of the time at each.
    fn embed(&self, at: f64) -> Result<Tensor> {
        let at = at.clamp(0.0, 1.0);
        let half = TIME_EMB / 2;
        let mut values = vec![0f32; TIME_EMB];
        for index in 0..half {
            let exponent = f64::from(TIME_EMB_MAX_EXPONENT) * index as f64 /
                (half - 1) as f64;
            let angle = at * 10f64.powf(exponent);
            values[index] = angle.sin() as f32;
            values[half + index] = angle.cos() as f32;
        }
        Tensor::from_vec(values, (1, TIME_EMB, 1), &self.device)?
            .to_dtype(self.dtype)
    }
}

/// Normalization of every channel over time, without an affine — torch's
/// `InstanceNorm1d` as it is built by default.
fn instance_norm(xs: &Tensor) -> Result<Tensor> {
    let mean = xs.mean_keepdim(2)?;
    let centered = xs.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(2)?;
    centered.broadcast_div(&(variance + INSTANCE_NORM_EPS)?.sqrt()?)
}
