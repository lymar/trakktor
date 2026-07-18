//! Embedded Silero-VAD weights.

/// Silero-VAD v5 (16 kHz) weights in safetensors format.
///
/// The model is tiny (~1.2 MB) and fixed, so it ships inside the binary — like
/// the Whisper tokenizer and mel-filter assets — rather than being downloaded.
/// The 15 tensors are named to match the module paths in [`super::model`], so
/// they load directly with no remapping.
pub const SILERO_VAD_16K: &[u8] =
    include_bytes!("assets/silero_vad_16k.safetensors");
