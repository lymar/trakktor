//! The Silero-VAD v5 network on candle.
//!
//! A faithful, op-for-op port of the original model (the reference clean
//! re-implementation is `tinygrad_model.py`): a fixed windowed-DFT STFT
//! front-end as a 1-D convolution, four ReLU conv blocks, a single LSTM cell,
//! and a 1×1 conv classifier with a sigmoid. It emits one speech probability
//! per 512-sample window.
//!
//! The model is small and runs on the CPU in f32 (matching the original; more
//! precise than whisper.cpp's f16 conv path). Only the LSTM is inherently
//! sequential: the STFT and conv stack are evaluated for a whole batch of
//! windows at once (in chunks, to bound memory) on candle, and the recurrence
//! plus the classifier run in plain Rust over the encoder outputs — avoiding
//! tens of thousands of tiny compute graphs while reproducing the arithmetic
//! exactly.

#[cfg(test)]
mod tests;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};

use super::{
    CONTEXT_SIZE, WINDOW_SIZE, assets::SILERO_VAD_16K, error::VadError,
};

/// STFT filter length (analysis window), in samples.
const N_FFT: usize = 256;
/// STFT hop, in samples.
const STFT_STRIDE: usize = 128;
/// Right-side reflection padding applied to each window before the STFT.
const REFLECT_PAD: usize = 64;
/// STFT output channels: 129 real + 129 imaginary.
const STFT_CHANNELS: usize = 258;
/// Frequency bins kept after taking the magnitude (`N_FFT / 2 + 1`).
const N_FREQS: usize = 129;
/// LSTM hidden size (also the encoder output width and classifier input).
const HIDDEN: usize = 128;
/// LSTM gate matrix rows (`4 * HIDDEN`, order i, f, g, o).
const GATES: usize = 4 * HIDDEN;
/// Windows evaluated through the conv encoder per candle batch, bounding the
/// peak intermediate memory on long files.
const ENCODER_CHUNK: usize = 2048;

/// The loaded Silero-VAD model.
pub struct Vad {
    device: Device,
    stft: Conv1d,
    conv1: Conv1d,
    conv2: Conv1d,
    conv3: Conv1d,
    conv4: Conv1d,
    /// Input-projection weight `weight_ih` as `(GATES, HIDDEN)`, kept on the
    /// device for the batched projection.
    w_ih: Tensor,
    /// Input-projection bias `bias_ih`, `(GATES,)`.
    b_ih: Tensor,
    /// Recurrent weight `weight_hh`, row-major `GATES * HIDDEN`, for the Rust
    /// recurrence.
    w_hh: Vec<f32>,
    /// Recurrent bias `bias_hh`, `GATES`.
    b_hh: Vec<f32>,
    /// Classifier weight (the 1×1 conv), `HIDDEN` values.
    fc_w: Vec<f32>,
    /// Classifier bias (a scalar).
    fc_b: f32,
}

/// Maps any backend failure onto the VAD error.
fn vad_err(context: &str, e: impl std::fmt::Display) -> VadError {
    VadError(format!("{context}: {e}"))
}

impl Vad {
    /// Loads the embedded Silero-VAD model on the CPU.
    ///
    /// # Errors
    ///
    /// Returns [`VadError`] if the weights cannot be parsed or a tensor is
    /// missing or the wrong shape.
    pub fn load() -> Result<Self, VadError> {
        let device = Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(
            SILERO_VAD_16K.to_vec(),
            DType::F32,
            &device,
        )
        .map_err(|e| vad_err("loading VAD weights", e))?;

        let stft = load_conv(
            &vb,
            "stft_conv",
            (STFT_CHANNELS, 1, N_FFT),
            Conv1dConfig {
                stride: STFT_STRIDE,
                ..Default::default()
            },
            false,
        )?;
        let same_stride1 = Conv1dConfig {
            padding: 1,
            ..Default::default()
        };
        let stride2 = Conv1dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let conv1 =
            load_conv(&vb, "conv1", (HIDDEN, N_FREQS, 3), same_stride1, true)?;
        let conv2 = load_conv(&vb, "conv2", (64, HIDDEN, 3), stride2, true)?;
        let conv3 = load_conv(&vb, "conv3", (64, 64, 3), stride2, true)?;
        let conv4 =
            load_conv(&vb, "conv4", (HIDDEN, 64, 3), same_stride1, true)?;

        let lstm = vb.pp("lstm_cell");
        let w_ih = lstm
            .get((GATES, HIDDEN), "weight_ih")
            .map_err(|e| vad_err("loading weight_ih", e))?;
        let b_ih = lstm
            .get(GATES, "bias_ih")
            .map_err(|e| vad_err("loading bias_ih", e))?;
        let w_hh =
            to_vec(&lstm.get((GATES, HIDDEN), "weight_hh"), "weight_hh")?;
        let b_hh = to_vec(&lstm.get(GATES, "bias_hh"), "bias_hh")?;

        let fc = vb.pp("final_conv");
        let fc_w =
            to_vec(&fc.get((1, HIDDEN, 1), "weight"), "final_conv.weight")?;
        let fc_b = to_vec(&fc.get(1, "bias"), "final_conv.bias")?[0];

        Ok(Self {
            device,
            stft,
            conv1,
            conv2,
            conv3,
            conv4,
            w_ih,
            b_ih,
            w_hh,
            b_hh,
            fc_w,
            fc_b,
        })
    }

    /// Computes the per-window speech probabilities for 16 kHz mono PCM, one
    /// value in `[0, 1]` per 512-sample window (`ceil(len / 512)` values).
    ///
    /// # Errors
    ///
    /// Returns [`VadError`] on a backend failure.
    pub fn probabilities(&self, audio: &[f32]) -> Result<Vec<f32>, VadError> {
        let n_windows = audio.len().div_ceil(WINDOW_SIZE);
        if n_windows == 0 {
            return Ok(Vec::new());
        }

        // The window of index i is `padded[i*512 .. i*512 + 576]`: 64 samples
        // of the previous window's tail as context (zeros before the first
        // window) followed by 512 new samples (zero-filled past the end).
        let window_len = CONTEXT_SIZE + WINDOW_SIZE;
        let mut padded = vec![0f32; CONTEXT_SIZE + n_windows * WINDOW_SIZE];
        padded[CONTEXT_SIZE..CONTEXT_SIZE + audio.len()].copy_from_slice(audio);

        // Input-projection outputs `z_ih` for every window, gathered across
        // encoder chunks; the recurrence then only needs the (small) recurrent
        // matmul per step.
        let mut z_ih = Vec::with_capacity(n_windows * GATES);
        let mut start = 0;
        while start < n_windows {
            let batch = ENCODER_CHUNK.min(n_windows - start);
            let mut frames = vec![0f32; batch * window_len];
            for w in 0..batch {
                let src = (start + w) * WINDOW_SIZE;
                frames[w * window_len..(w + 1) * window_len]
                    .copy_from_slice(&padded[src..src + window_len]);
            }
            let batch_z = self.encode_chunk(&frames, batch)?;
            z_ih.extend_from_slice(&batch_z);
            start += batch;
        }

        Ok(self.recur_and_classify(&z_ih, n_windows))
    }

    /// Runs the STFT + conv encoder and the input projection for one batch of
    /// `batch` windows (`frames` is `batch * (CONTEXT_SIZE + WINDOW_SIZE)`
    /// row-major), returning `z_ih` as `batch * GATES` row-major.
    fn encode_chunk(
        &self,
        frames: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>, VadError> {
        let window_len = CONTEXT_SIZE + WINDOW_SIZE;
        let build = |t: candle_core::Result<Tensor>, what: &str| {
            t.map_err(|e| vad_err(what, e))
        };

        let x = build(
            Tensor::from_vec(
                frames.to_vec(),
                (batch, window_len),
                &self.device,
            ),
            "VAD frame tensor",
        )?;
        // Reflect-pad the right edge by 64 (mirror without repeating the edge
        // sample): append x[.., 574], x[.., 573], …, x[.., 511].
        let reflected = build(
            x.narrow(1, window_len - 1 - REFLECT_PAD, REFLECT_PAD)
                .and_then(|t| t.contiguous())
                .and_then(|t| t.flip(&[1])),
            "VAD reflect pad",
        )?;
        let x = build(Tensor::cat(&[&x, &reflected], 1), "VAD pad concat")?;
        let x = build(
            x.reshape((batch, 1, window_len + REFLECT_PAD)),
            "VAD reshape",
        )?;

        // STFT as a fixed conv, then magnitude over the 129 real/imag pairs.
        let stft = build(self.stft.forward(&x), "VAD stft")?;
        let real = build(stft.narrow(1, 0, N_FREQS), "VAD stft real")?;
        let imag = build(stft.narrow(1, N_FREQS, N_FREQS), "VAD stft imag")?;
        let mag = build(
            real.sqr()
                .and_then(|r| imag.sqr().and_then(|i| r.add(&i)))
                .and_then(|s| s.sqrt()),
            "VAD magnitude",
        )?;

        let enc = build(
            self.conv1.forward(&mag).and_then(|t| t.relu()),
            "VAD conv1",
        )?;
        let enc = build(
            self.conv2.forward(&enc).and_then(|t| t.relu()),
            "VAD conv2",
        )?;
        let enc = build(
            self.conv3.forward(&enc).and_then(|t| t.relu()),
            "VAD conv3",
        )?;
        let enc = build(
            self.conv4.forward(&enc).and_then(|t| t.relu()),
            "VAD conv4",
        )?;
        // Drop the single time frame → (batch, HIDDEN).
        let enc = build(enc.reshape((batch, HIDDEN)), "VAD encoder squeeze")?;

        // Input projection z_ih = enc · weight_ih^T + bias_ih → (batch, GATES).
        let z = build(
            self.w_ih
                .t()
                .and_then(|w| enc.matmul(&w))
                .and_then(|z| z.broadcast_add(&self.b_ih)),
            "VAD input projection",
        )?;
        z.flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| vad_err("VAD z_ih readout", e))
    }

    /// Steps the LSTM over all windows and applies the classifier, in plain
    /// f32 Rust. `z_ih` is the pre-computed input projection (`n * GATES`).
    ///
    /// The cell computes every gate from the previous `(h, c)`, so the new
    /// state is written to separate buffers and swapped in only after the whole
    /// step — a gate must never read a partially updated `h`.
    fn recur_and_classify(&self, z_ih: &[f32], n: usize) -> Vec<f32> {
        let mut h = vec![0f32; HIDDEN];
        let mut c = vec![0f32; HIDDEN];
        let mut next_h = vec![0f32; HIDDEN];
        let mut next_c = vec![0f32; HIDDEN];
        let mut probs = Vec::with_capacity(n);
        for t in 0..n {
            let z = &z_ih[t * GATES..(t + 1) * GATES];
            // Classifier accumulator: fc_b + Σ_k relu(h'_k) · fc_w_k.
            let mut logit = self.fc_b;
            for k in 0..HIDDEN {
                let ig = sigmoid(self.gate(z, &h, k));
                let fg = sigmoid(self.gate(z, &h, HIDDEN + k));
                let gg = self.gate(z, &h, 2 * HIDDEN + k).tanh();
                let og = sigmoid(self.gate(z, &h, 3 * HIDDEN + k));
                let cell = fg * c[k] + ig * gg;
                let hid = og * cell.tanh();
                next_c[k] = cell;
                next_h[k] = hid;
                logit += hid.max(0.0) * self.fc_w[k];
            }
            std::mem::swap(&mut h, &mut next_h);
            std::mem::swap(&mut c, &mut next_c);
            probs.push(sigmoid(logit));
        }
        probs
    }

    /// One LSTM gate pre-activation: `z_ih[row] + bias_hh[row] + weight_hh[row]
    /// · h`. `row` selects the gate block (i, f, g, o) and channel.
    fn gate(&self, z: &[f32], h: &[f32], row: usize) -> f32 {
        let weights = &self.w_hh[row * HIDDEN..(row + 1) * HIDDEN];
        let mut acc = z[row] + self.b_hh[row];
        for i in 0..HIDDEN {
            acc += weights[i] * h[i];
        }
        acc
    }
}

/// Loads a `Conv1d` (`weight` `(out, in, kernel)`, optional `bias`) under `vb`.
fn load_conv(
    vb: &VarBuilder,
    prefix: &str,
    shape: (usize, usize, usize),
    config: Conv1dConfig,
    bias: bool,
) -> Result<Conv1d, VadError> {
    let vb = vb.pp(prefix);
    let weight = vb
        .get(shape, "weight")
        .map_err(|e| vad_err(&format!("loading {prefix}.weight"), e))?;
    let bias = if bias {
        Some(
            vb.get(shape.0, "bias")
                .map_err(|e| vad_err(&format!("loading {prefix}.bias"), e))?,
        )
    } else {
        None
    };
    Ok(Conv1d::new(weight, bias, config))
}

/// Reads a tensor into a flat row-major `Vec<f32>`.
fn to_vec(
    tensor: &candle_core::Result<Tensor>,
    what: &str,
) -> Result<Vec<f32>, VadError> {
    tensor
        .as_ref()
        .map_err(|e| vad_err(&format!("loading {what}"), e))?
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| vad_err(&format!("reading {what}"), e))
}

/// The logistic sigmoid.
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
