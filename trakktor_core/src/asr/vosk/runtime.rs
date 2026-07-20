//! The Vosk runtime seam and its candle implementation.
//!
//! [`EncoderSeam`] is the narrow interface the rest of the engine sees: a
//! loaded model exposes the fbank extractor and the map from features to the
//! encoder output (already projected into the joint dimension) — a single
//! full-context pass for offline models, a stateful chunked
//! [`StreamingSession`] for streaming ones. Decoding stays off the device
//! either way: the transducer head runs on the CPU over the read-back
//! encoder output (see [`decode`](super::decode)), so a chunk costs exactly
//! one device synchronization on every runtime.
//!
//! [`VoskModel`] implements the seam on [candle](candle_core). Devices: CPU
//! always; Metal behind the `vosk-metal` cargo feature. An alternative
//! burn-backed implementation lives in `runtime_burn` (the `vosk-burn` cargo
//! feature).

pub mod net;
#[cfg(test)]
mod tests;

use candle_core::{DType, Device, Tensor};
use net::{NetState, ZipformerNet};

use super::{
    error::VoskError,
    feature::{FbankExtractor, Features, N_MELS},
    weights::{ModelWeights, ZipformerConfig},
};

/// Maps any backend failure onto the engine's model error.
pub(crate) fn model_err(context: &str, e: impl std::fmt::Display) -> VoskError {
    VoskError::InvalidModel(format!("{context}: {e}"))
}

/// Compute precision of the encoder.
///
/// [`F16`](Precision::F16) halves the memory and speeds up GPU matmuls;
/// [`F32`](Precision::F32) keeps full precision for bit-close reference
/// parity. The transducer head always computes in `f32` on the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Half precision (`f16`).
    #[default]
    F16,
    /// Full precision (`f32`).
    F32,
}

impl Precision {
    fn dtype(self) -> DType {
        match self {
            Self::F16 => DType::F16,
            Self::F32 => DType::F32,
        }
    }
}

/// A loaded model behind a runtime-agnostic seam: the fbank extractor plus
/// the encoder in both run modes. The transducer decoding, segmentation, and
/// tokenization are shared across runtimes.
pub trait EncoderSeam {
    /// The model geometry.
    fn config(&self) -> &ZipformerConfig;

    /// The fbank extractor.
    fn feature(&self) -> &FbankExtractor;

    /// Full-context encoding of one chunk's features: returns the projected
    /// encoder output as row-major `[frames, joiner_dim]` `f32` and the
    /// frame count. Offline models only.
    fn encode(
        &self,
        features: &Features,
    ) -> Result<(Vec<f32>, usize), VoskError>;

    /// Opens a chunked encoding session with fresh state. Streaming models
    /// only.
    fn start_stream(&self)
    -> Result<Box<dyn StreamingSession + '_>, VoskError>;
}

/// A stateful chunked encoder run over one audio stream.
pub trait StreamingSession {
    /// Encodes one feature window (`streaming.window_frames` rows),
    /// advancing the internal state; returns the block's projected output
    /// (row-major `[frames, joiner_dim]` `f32`) and its frame count.
    fn accept(
        &mut self,
        features: &Features,
    ) -> Result<(Vec<f32>, usize), VoskError>;
}

/// A loaded Vosk model on candle.
pub struct VoskModel {
    device: Device,
    dtype: DType,
    feature: FbankExtractor,
    net: ZipformerNet,
}

impl VoskModel {
    /// Loads extracted weights onto `device` at `precision`.
    pub fn load(
        weights: &ModelWeights,
        device: Device,
        precision: Precision,
    ) -> Result<Self, VoskError> {
        let dtype = precision.dtype();
        if weights.config.feature_dim != N_MELS {
            return Err(VoskError::InvalidModel(format!(
                "model expects {} feature bins, the fbank front end produces \
                 {N_MELS}",
                weights.config.feature_dim
            )));
        }
        let net = ZipformerNet::load(weights, &device, dtype)
            .map_err(|e| model_err("loading encoder weights", e))?;
        Ok(Self {
            device,
            dtype,
            feature: FbankExtractor::new(),
            net,
        })
    }

    /// [`load`](Self::load) on the CPU.
    pub fn load_cpu(
        weights: &ModelWeights,
        precision: Precision,
    ) -> Result<Self, VoskError> {
        Self::load(weights, Device::Cpu, precision)
    }

    /// [`load`](Self::load) on the first Metal device.
    #[cfg(feature = "vosk-metal")]
    pub fn load_metal(
        weights: &ModelWeights,
        precision: Precision,
    ) -> Result<Self, VoskError> {
        let device = Device::new_metal(0)
            .map_err(|e| model_err("creating the metal device", e))?;
        Self::load(weights, device, precision)
    }

    /// One chunk's features as the `[T, n_mels]` device tensor.
    fn features_tensor(
        &self,
        features: &Features,
    ) -> Result<Tensor, VoskError> {
        Tensor::from_slice(
            features.data(),
            (features.n_frames(), N_MELS),
            &self.device,
        )
        .and_then(|t| t.to_dtype(self.dtype))
        .map_err(|e| model_err("building feature tensor", e))
    }

    /// Reads an encoder output `[T', joiner_dim]` back as flat `f32`.
    fn read_back(
        &self,
        encoded: Tensor,
    ) -> Result<(Vec<f32>, usize), VoskError> {
        let frames = encoded.dim(0).map_err(|e| model_err("output", e))?;
        let flat = encoded
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| model_err("encoder read-back", e))?;
        Ok((flat, frames))
    }
}

impl EncoderSeam for VoskModel {
    fn config(&self) -> &ZipformerConfig { self.net.config() }

    fn feature(&self) -> &FbankExtractor { &self.feature }

    fn encode(
        &self,
        features: &Features,
    ) -> Result<(Vec<f32>, usize), VoskError> {
        let input = self.features_tensor(features)?;
        let encoded = self
            .net
            .forward(&input)
            .map_err(|e| model_err("encoder forward", e))?;
        self.read_back(encoded)
    }

    fn start_stream(
        &self,
    ) -> Result<Box<dyn StreamingSession + '_>, VoskError> {
        let state = self
            .net
            .init_state()
            .map_err(|e| model_err("initializing streaming state", e))?;
        Ok(Box::new(CandleSession { model: self, state }))
    }
}

/// A candle streaming session: the device-resident caches plus the model.
struct CandleSession<'m> {
    model: &'m VoskModel,
    state: NetState,
}

impl StreamingSession for CandleSession<'_> {
    fn accept(
        &mut self,
        features: &Features,
    ) -> Result<(Vec<f32>, usize), VoskError> {
        let input = self.model.features_tensor(features)?;
        let encoded = self
            .model
            .net
            .forward_streaming(&input, &mut self.state)
            .map_err(|e| model_err("streaming encoder forward", e))?;
        self.model.read_back(encoded)
    }
}
