//! The catalog: which models this stage runs and where they come from.
//!
//! Two entries, one per question the stage answers, and neither is a choice
//! the caller makes: upstream publishes exactly one model for each, and there
//! is no smaller or larger variant to prefer. Both are small next to the rest
//! of the domain — seven megabytes and thirty-two against a hundred and thirty
//! for the layout model.
//!
//! The revisions are pinned rather than tracked: `inference.json` +
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

/// What a model of this stage decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Which of the four right angles the page is at.
    Orientation,
    /// Where every pixel of the straightened page comes from.
    Unwarp,
}

/// A published model.
#[derive(Debug, Clone, Copy)]
pub struct Model {
    /// Upstream's name, which is also the repository name and the directory
    /// the artifacts are cached under.
    pub name: &'static str,
    pub kind: Kind,
    /// The repository revision the digests below were taken at.
    pub revision: &'static str,
    pub files: &'static [File],
}

impl Model {
    /// Total download size, in bytes.
    pub fn size(&self) -> u64 { self.files.iter().map(|f| f.size).sum() }
}

/// The page-orientation classifier: a PP-LCNet reading four classes.
pub const ORIENTATION: &str = "PP-LCNet_x1_0_doc_ori";

/// The unwarper.
pub const UNWARP: &str = "UVDoc";

/// Everything this stage can run.
pub const MODELS: &[Model] = &[
    Model {
        name: ORIENTATION,
        kind: Kind::Orientation,
        revision: "d3b95a6dff5fe8a94f2748e12b61cb26818a0df8",
        files: &[
            File { name: "config.json", size: 2561, blake3: "627cc45daf5abe5e1cfabd028916c999e5ecd0caa49093205e962a90f7c7e4b4" },
            File { name: "inference.json", size: 104441, blake3: "aab28e5a29da724ee318df5ee2dec843709166622b6fd68453393eb6145af704" },
            File { name: "inference.pdiparams", size: 6754166, blake3: "9fda64ac7c1a401504878f45df03aee6d101fad8b53b019a5f6e497bda516ab1" },
        ],
    },
    Model {
        name: UNWARP,
        kind: Kind::Unwarp,
        revision: "16c3f0ea9c2f0c6a57e24160f7eeaa7574613fa3",
        files: &[
            File { name: "config.json", size: 1489, blake3: "fae633d6a3feac1951ead4f9730b7015b9708ca5a57d608d036086d9621d047a" },
            File { name: "inference.json", size: 190986, blake3: "027fd03fe79b49f3a881dc7dde0d647923c212cd75e22a51354daa547decc15c" },
            File { name: "inference.pdiparams", size: 32054311, blake3: "35c23634d10cd38a10a8255e844410abcb8a1692d2303a9ea7de71661c07ef9f" },
        ],
    },
];

/// Looks a model up by name.
pub fn model(name: &str) -> Result<&'static Model, OcrError> {
    MODELS
        .iter()
        .find(|entry| entry.name == name)
        .ok_or_else(|| {
            let known: Vec<&str> = MODELS.iter().map(|m| m.name).collect();
            OcrError::UnknownModel {
                name: name.to_string(),
                known: known.join(", "),
            }
        })
}
