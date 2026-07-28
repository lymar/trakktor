//! The vocoder: mel spectrogram in, waveform out.
//!
//! A small convolutional network (a hundredth of the DiT's size) predicts, per
//! frame, a magnitude and a phase for every frequency bin — that is, a complex
//! spectrum — and the waveform is the overlap-add of its inverse transform.
//! Only the network runs on the device: neither tensor backend has an FFT, so
//! the head's output goes back to the host and
//! [`MelBasis::istft`](crate::tts::espeech::mel::MelBasis::istft) finishes the
//! job. That split has a pleasant side effect — the inverse transform is
//! literally the same code on both runtimes.
//!
//! Ported from Vocos (MIT).

use candle_core::{D, Result, Tensor};
use candle_nn::{Conv1d, Linear, Module, VarBuilder};

use super::layers::{AffineNorm, ConvNeXtBlock, conv1d};
use crate::tts::espeech::config::VocoderConfig;

/// Epsilon of the vocoder's normalizations.
const NORM_EPS: f64 = 1e-6;

/// Ceiling on the predicted magnitudes — the reference's own guard against a
/// spectrum that would explode into a click.
const MAX_MAGNITUDE: f64 = 1e2;

/// The vocoder, loaded.
pub struct Vocoder {
    cfg: VocoderConfig,
    embed: Conv1d,
    norm: AffineNorm,
    blocks: Vec<ConvNeXtBlock>,
    final_norm: AffineNorm,
    out: Linear,
}

impl Vocoder {
    /// Loads the network with the geometry derived from its checkpoint.
    pub fn load(cfg: VocoderConfig, vb: VarBuilder) -> Result<Self> {
        let backbone = vb.pp("backbone");
        Ok(Self {
            cfg,
            embed: conv1d(cfg.mel_channels, cfg.dim, 7, backbone.pp("embed"))?,
            norm: AffineNorm::load(cfg.dim, NORM_EPS, backbone.pp("norm"))?,
            blocks: (0..cfg.layers)
                .map(|index| {
                    ConvNeXtBlock::load_scaled(
                        cfg.dim,
                        cfg.ff_inner,
                        backbone.pp("convnext").pp(index.to_string()),
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            final_norm: AffineNorm::load(
                cfg.dim,
                NORM_EPS,
                backbone.pp("final_layer_norm"),
            )?,
            out: candle_nn::linear(
                cfg.dim,
                cfg.n_fft + 2,
                vb.pp("head").pp("out"),
            )?,
        })
    }

    /// The geometry this was loaded with.
    pub fn config(&self) -> &VocoderConfig { &self.cfg }

    /// Runs the network over a mel spectrogram shaped `[1, frames, mel]` and
    /// returns the magnitudes and phases the inverse transform needs, each
    /// `[frames, n_fft / 2 + 1]` in row-major order.
    pub fn spectrum(&self, mel: &Tensor) -> Result<(Vec<f32>, Vec<f32>)> {
        // The network is channels-first; the mel arrives frames-first, the way
        // the DiT produces it.
        let hidden = self.embed.forward(&mel.transpose(1, 2)?.contiguous()?)?;
        let mut hidden =
            self.norm.forward(&hidden.transpose(1, 2)?.contiguous()?)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        let hidden = self.final_norm.forward(&hidden)?;
        let out = self.out.forward(&hidden)?;

        let bins = self.cfg.n_fft / 2 + 1;
        let magnitude = out
            .narrow(D::Minus1, 0, bins)?
            .exp()?
            .clamp(0f64, MAX_MAGNITUDE)?;
        let phase = out.narrow(D::Minus1, bins, bins)?;
        Ok((
            magnitude
                .flatten_all()?
                .to_dtype(candle_core::DType::F32)?
                .to_vec1()?,
            phase
                .flatten_all()?
                .to_dtype(candle_core::DType::F32)?
                .to_vec1()?,
        ))
    }
}
