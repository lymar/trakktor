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

/// Cross-attention heads that track word-level alignment, per published
/// model, as `(decoder layer, head)` pairs — decoded once from the masks the
/// reference ships. The podlodka fine-tunes declare theirs in their
/// checkpoints' generation config: the same heads as their parent models.
pub const ALIGNMENT_HEADS: &[(&str, &[(usize, usize)])] = &[
    (
        "tiny.en",
        &[
            (1, 0),
            (2, 0),
            (2, 5),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
        ],
    ),
    ("tiny", &[(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)]),
    ("base.en", &[(3, 3), (4, 7), (5, 1), (5, 5), (5, 7)]),
    (
        "base",
        &[
            (3, 1),
            (4, 2),
            (4, 3),
            (4, 7),
            (5, 1),
            (5, 2),
            (5, 4),
            (5, 6),
        ],
    ),
    (
        "small.en",
        &[
            (6, 6),
            (7, 0),
            (7, 3),
            (7, 8),
            (8, 2),
            (8, 5),
            (8, 7),
            (9, 0),
            (9, 4),
            (9, 8),
            (9, 10),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 6),
            (10, 11),
            (11, 2),
            (11, 4),
        ],
    ),
    (
        "small",
        &[
            (5, 3),
            (5, 9),
            (8, 0),
            (8, 4),
            (8, 7),
            (8, 8),
            (9, 0),
            (9, 7),
            (9, 9),
            (10, 5),
        ],
    ),
    (
        "medium.en",
        &[
            (11, 4),
            (14, 1),
            (14, 12),
            (14, 14),
            (15, 4),
            (16, 0),
            (16, 4),
            (16, 9),
            (17, 12),
            (17, 14),
            (18, 7),
            (18, 10),
            (18, 15),
            (20, 0),
            (20, 3),
            (20, 9),
            (20, 14),
            (21, 12),
        ],
    ),
    (
        "medium",
        &[(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)],
    ),
    (
        "large-v1",
        &[
            (9, 19),
            (11, 2),
            (11, 4),
            (11, 17),
            (22, 7),
            (22, 11),
            (22, 17),
            (23, 2),
            (23, 15),
        ],
    ),
    (
        "large-v2",
        &[
            (10, 12),
            (13, 17),
            (16, 11),
            (16, 12),
            (16, 13),
            (17, 15),
            (17, 16),
            (18, 4),
            (18, 11),
            (18, 19),
            (19, 11),
            (21, 2),
            (21, 3),
            (22, 3),
            (22, 9),
            (22, 12),
            (23, 5),
            (23, 7),
            (23, 13),
            (25, 5),
            (26, 1),
            (26, 12),
            (27, 15),
        ],
    ),
    (
        "large-v3",
        &[
            (7, 0),
            (10, 17),
            (12, 18),
            (13, 12),
            (16, 1),
            (17, 14),
            (19, 11),
            (21, 4),
            (24, 1),
            (25, 6),
        ],
    ),
    (
        "large",
        &[
            (7, 0),
            (10, 17),
            (12, 18),
            (13, 12),
            (16, 1),
            (17, 14),
            (19, 11),
            (21, 4),
            (24, 1),
            (25, 6),
        ],
    ),
    (
        "large-v3-turbo",
        &[(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)],
    ),
    (
        "turbo",
        &[(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)],
    ),
    (
        "podlodka",
        &[
            (7, 0),
            (10, 17),
            (12, 18),
            (13, 12),
            (16, 1),
            (17, 14),
            (19, 11),
            (21, 4),
            (24, 1),
            (25, 6),
        ],
    ),
    (
        "podlodka-turbo",
        &[(2, 4), (2, 11), (3, 3), (3, 6), (3, 11), (3, 14)],
    ),
];

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
