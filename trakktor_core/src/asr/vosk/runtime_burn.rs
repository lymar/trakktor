//! The burn-backed Vosk runtime.
//!
//! An alternative [`EncoderSeam`] implementation on [burn](burn), selectable
//! at run time next to the candle one. Backends: ndarray on the CPU (**f32
//! only** — the ndarray backend has no half-precision element), and wgpu with
//! MSL-compiled kernels on Metal (f16 and f32), with operator fusion and
//! autotuning.
//!
//! The transducer head is not here: it always runs in `f32` on the CPU (see
//! [`decode`](super::decode)), identical code and weights across runtimes, so
//! only the Zipformer2 encoder is ported to burn.

pub mod net;
#[cfg(test)]
mod tests;

use burn::{
    backend::{
        ndarray::{NdArray, NdArrayDevice},
        wgpu::{Metal, WgpuDevice},
    },
    tensor::{Tensor, TensorData, backend::Backend, f16},
};
use net::{NetState, ZipformerNet};

use super::{
    error::VoskError,
    feature::{FbankExtractor, Features, N_MELS},
    runtime::{EncoderSeam, Precision, StreamingSession},
    weights::{ModelWeights, ZipformerConfig},
};

/// A loaded Vosk model on a burn backend.
pub struct VoskBurnModel<B: Backend> {
    device: B::Device,
    feature: FbankExtractor,
    net: ZipformerNet<B>,
}

impl<B: Backend> VoskBurnModel<B> {
    /// Loads extracted weights onto `device`.
    pub fn load(
        weights: &ModelWeights,
        device: B::Device,
    ) -> Result<Self, VoskError> {
        if weights.config.feature_dim != N_MELS {
            return Err(VoskError::InvalidModel(format!(
                "model expects {} feature bins, the fbank front end produces \
                 {N_MELS}",
                weights.config.feature_dim
            )));
        }
        let net = ZipformerNet::load(weights, &device);
        Ok(Self {
            device,
            feature: FbankExtractor::new(),
            net,
        })
    }

    /// One chunk's features as the `[T, n_mels]` device tensor.
    fn features_tensor(&self, features: &Features) -> Tensor<B, 2> {
        Tensor::from_data(
            TensorData::new(
                features.data().to_vec(),
                [features.n_frames(), N_MELS],
            ),
            &self.device,
        )
    }
}

impl<B: Backend> EncoderSeam for VoskBurnModel<B> {
    fn config(&self) -> &ZipformerConfig { self.net.config() }

    fn feature(&self) -> &FbankExtractor { &self.feature }

    fn encode(
        &self,
        features: &Features,
    ) -> Result<(Vec<f32>, usize), VoskError> {
        let input = self.features_tensor(features);
        let encoded = self.net.forward(input);
        Ok(ZipformerNet::read_back(encoded))
    }

    fn start_stream(
        &self,
    ) -> Result<Box<dyn StreamingSession + '_>, VoskError> {
        let state = self.net.init_state();
        Ok(Box::new(BurnSession { model: self, state }))
    }
}

/// A burn streaming session: the device-resident caches plus the model.
struct BurnSession<'m, B: Backend> {
    model: &'m VoskBurnModel<B>,
    state: NetState<B>,
}

impl<B: Backend> StreamingSession for BurnSession<'_, B> {
    fn accept(
        &mut self,
        features: &Features,
    ) -> Result<(Vec<f32>, usize), VoskError> {
        let input = self.model.features_tensor(features);
        let encoded = self.model.net.forward_streaming(input, &mut self.state);
        Ok(ZipformerNet::read_back(encoded))
    }
}

/// Loads a model on the burn CPU backend (ndarray).
///
/// # Errors
///
/// [`VoskError::InvalidOptions`] for [`Precision::F16`] — the ndarray backend
/// computes in f32 only.
pub fn load_cpu(
    weights: &ModelWeights,
    precision: Precision,
) -> Result<Box<dyn EncoderSeam>, VoskError> {
    match precision {
        Precision::F32 => Ok(Box::new(VoskBurnModel::<NdArray<f32>>::load(
            weights,
            NdArrayDevice::Cpu,
        )?)),
        Precision::F16 => Err(VoskError::InvalidOptions(
            "the burn runtime computes in f32 on the CPU; use f32 precision \
             (or the metal device)"
                .into(),
        )),
    }
}

/// Loads a model on the burn Metal backend (wgpu with MSL-compiled kernels),
/// at the requested precision.
pub fn load_metal(
    weights: &ModelWeights,
    precision: Precision,
) -> Result<Box<dyn EncoderSeam>, VoskError> {
    let device = WgpuDevice::default();
    match precision {
        Precision::F16 => Ok(Box::new(VoskBurnModel::<Metal<f16>>::load(
            weights, device,
        )?)),
        Precision::F32 => Ok(Box::new(VoskBurnModel::<Metal<f32>>::load(
            weights, device,
        )?)),
    }
}
