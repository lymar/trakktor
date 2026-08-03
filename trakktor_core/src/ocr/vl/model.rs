//! The catalog: which checkpoint this engine runs and where it comes from.
//!
//! One entry, and that is the point. The three published versions of this model
//! share a **byte-identical** `config.json` and differ only in their weights —
//! and the difference is not a refinement but a capability: on the same images
//! version 1.0 answers a Tibetan block with Thai, with Korean, or with a run of
//! `5555…`, while 1.6 reads it. So "fall back to the older, it is the same
//! architecture" is not a fallback here, it silently switches a whole script
//! off. The version is pinned, and so is the repository revision it was taken
//! at: the weights live on a moving branch, and a re-publish must not change
//! what trakktor runs.

use crate::ocr::error::OcrError;

/// One file of the published model.
#[derive(Debug, Clone, Copy)]
pub struct File {
    pub name: &'static str,
    pub size: u64,
    pub blake3: &'static str,
}

/// The published checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct Model {
    /// Upstream's name, which is also the repository name and the directory
    /// the weights are cached under.
    pub name: &'static str,
    /// The repository revision the digests below were taken at.
    pub revision: &'static str,
    pub files: &'static [File],
}

impl Model {
    /// Total download size, in bytes.
    pub fn size(&self) -> u64 { self.files.iter().map(|f| f.size).sum() }
}

/// The checkpoint the engine runs.
pub const DEFAULT_MODEL: &str = "PaddleOCR-VL-1.6";

/// Everything this engine can run.
pub const MODELS: &[Model] = &[Model {
    name: "PaddleOCR-VL-1.6",
    revision: "66317acc4c9fc17bd154591ce650735cd2855f3e",
    // 1.93 GB, almost all of it the weights.
    files: &[
        File { name: "config.json", size: 2059, blake3: "bd249ac20e7b0458fe9d8c89662e3bd6ad0125864226bce912e0b62739b9a0f5" },
        File { name: "preprocessor_config.json", size: 641, blake3: "06a17b64a56e696acc447ca8002286dde7cc2900f57378e478178c39927cf70e" },
        File { name: "tokenizer.json", size: 11189060, blake3: "664e6c2425fd92e710a67a919753493657005ddcd1cb839737b6678db3edf3c3" },
        File { name: "model.safetensors", size: 1917255968, blake3: "4dc3ab13685a0c0a701f77f9c5ebafdc0004074247e00bde6b7e7b04279a41fc" },
    ],
}];

/// Looks a model up by name.
pub fn model(name: &str) -> Result<&'static Model, OcrError> {
    MODELS.iter().find(|m| m.name == name).ok_or_else(|| {
        OcrError::UnknownModel {
            name: name.to_string(),
            known: MODELS.iter().map(|m| m.name).collect::<Vec<_>>().join(", "),
        }
    })
}
