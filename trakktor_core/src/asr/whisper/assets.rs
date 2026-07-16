//! Embedded assets: mel filterbanks and BPE vocabularies.
//!
//! Whisper projects the STFT power spectrum onto a mel scale using fixed
//! filterbanks (the librosa/Slaney matrices). The exact matrices the model
//! ships are embedded here as raw little-endian f32 in row-major order: one row
//! per mel band, [`N_FREQS`] columns per row. The two byte-pair-encoding
//! vocabularies are embedded verbatim in their original text form (one
//! `base64(token) rank` pair per line).

use std::sync::OnceLock;

use super::feature::MelBands;

/// BPE vocabulary of the English-only models.
pub const GPT2_TIKTOKEN: &str = include_str!("assets/gpt2.tiktoken");

/// BPE vocabulary of the multilingual models.
pub const MULTILINGUAL_TIKTOKEN: &str =
    include_str!("assets/multilingual.tiktoken");

const MEL_80_BYTES: &[u8] = include_bytes!("assets/mel_80.bin");
const MEL_128_BYTES: &[u8] = include_bytes!("assets/mel_128.bin");

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Returns the mel filterbank for `bands` as a flat row-major slice.
///
/// The slice has `bands.count() * N_FREQS` elements; row `m` occupies
/// `[m * N_FREQS .. (m + 1) * N_FREQS]`.
pub fn mel_filters(bands: MelBands) -> &'static [f32] {
    static MEL_80: OnceLock<Vec<f32>> = OnceLock::new();
    static MEL_128: OnceLock<Vec<f32>> = OnceLock::new();
    match bands {
        MelBands::Mel80 => MEL_80.get_or_init(|| decode_f32(MEL_80_BYTES)),
        MelBands::Mel128 => MEL_128.get_or_init(|| decode_f32(MEL_128_BYTES)),
    }
}
