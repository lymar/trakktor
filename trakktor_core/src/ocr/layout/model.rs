//! The catalog: which layout model this stage runs and where it comes from.
//!
//! One entry, and the reason is measured rather than tidy. The published family
//! has three sizes, and the two small ones are a *different architecture*
//! (GFL/PicoDet rather than DETR) — so they would be a second port, not a
//! smaller download. On the pages this stage was measured against they also do
//! not find what the stage exists for: on a title page the medium model returns
//! no title, no abstract and no page number, and on a page of display formulas
//! it finds one of the nine the large model finds.
//!
//! The revision is pinned rather than tracked: `inference.json` +
//! `inference.pdiparams` is an internal Paddle contract, not a published
//! format, and it has changed once already.

use crate::ocr::error::OcrError;

/// One file of a published model.
#[derive(Debug, Clone, Copy)]
pub struct File {
    pub name: &'static str,
    pub size: u64,
    pub blake3: &'static str,
}

/// A published layout model.
#[derive(Debug, Clone, Copy)]
pub struct Model {
    /// Upstream's name, which is also the repository name and the directory
    /// the artifacts are cached under.
    pub name: &'static str,
    /// The repository revision the digests below were taken at.
    pub revision: &'static str,
    /// The square side the model was exported for. The graph declares it and
    /// the port checks it: this export has no dynamic input.
    pub input_side: usize,
    pub files: &'static [File],
}

impl Model {
    /// Total download size, in bytes.
    pub fn size(&self) -> u64 { self.files.iter().map(|f| f.size).sum() }
}

/// The model the stage runs unless told otherwise.
pub const DEFAULT_MODEL: &str = "PP-DocLayout_plus-L";

/// Everything this stage can run.
pub const MODELS: &[Model] = &[Model {
    name: "PP-DocLayout_plus-L",
    revision: "aa52b8528c84f9b1a34ac3a88fe0e576edb9d11d",
    input_side: 800,
    // 129 MB, almost all of it the weights.
    files: &[
        File { name: "config.json", size: 6074, blake3: "c63dcfb2e8a98868145b3704a7ad1aa9c6e67003091612e645a5a6af883731aa" },
        File { name: "inference.json", size: 1081621, blake3: "6d30c7f591f3e4c3ef8f095749fe85a0b9327fe322d664570223ab2ef9ef1c7b" },
        File { name: "inference.pdiparams", size: 129307978, blake3: "6ce856505eeb7e2aa4dead535950f633f825c1cbde1807724449bbc1be795f49" },
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
