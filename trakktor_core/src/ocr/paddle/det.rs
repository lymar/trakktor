//! The text detector: one of two networks, chosen by the artifact itself.
//!
//! Either way the contract is the same. The network takes a page whose sides
//! are multiples of 32 and returns a single-channel probability map **at the
//! input resolution** — the head ends with two stride-2 transposed convolutions
//! that undo the pyramid's quarter scale, and the sigmoid is inside the
//! exported graph. Everything after that (thresholding, contours, boxes) is
//! geometry and lives in [`super::db`].
//!
//! What differs is size and what gets found. The [small](mobile) one is 4.7 MB
//! and reads a page in about a second; the [large](server) one is nineteen
//! times that and takes tens of seconds, and it finds the short lines and
//! superscripts the small one drops. Neither cares which script the page is in.
//!
//! Which of the two an artifact holds is not a property of its name — a
//! caller may hand `--det-model` a directory — so it is taken from the graph,
//! which names its own backbone.

#[cfg(test)]
mod tests;

pub mod mobile;
pub mod server;

use candle_core::Tensor;

use super::{artifact::Artifact, hgnet, net::Loader};
use crate::ocr::error::OcrError;

/// The side length the input must be a multiple of. Below the pyramid's
/// coarsest level the top-down additions would disagree by a row.
pub const SIZE_MULTIPLE: usize = hgnet::SIZE_MULTIPLE;

/// The backbone each network declares itself with.
const MOBILE_BACKBONE: &str = "PPLCNetV3";
const SERVER_BACKBONE: &str = "PPHGNetV2";

/// A loaded detector. Both variants are boxed: a network holds its own weights,
/// so the two differ in size by more than an enum should carry.
#[derive(Debug)]
pub enum Detector {
    Mobile(Box<mobile::Net>),
    Server(Box<server::Net>),
}

impl Detector {
    /// Loads whichever of the two the artifact holds.
    pub fn load(loader: &Loader) -> Result<Self, OcrError> {
        let artifact = loader.artifact();
        if artifact.has_module(MOBILE_BACKBONE) {
            Ok(Self::Mobile(Box::new(mobile::Net::load(loader)?)))
        } else if artifact.has_module(SERVER_BACKBONE) {
            Ok(Self::Server(Box::new(server::Net::load(loader)?)))
        } else {
            Err(unknown_backbone(artifact))
        }
    }

    /// Runs the network over one normalized page, returning the probability
    /// map as `[1, 1, height, width]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let (_, _, height, width) = x.dims4()?;
        if height % SIZE_MULTIPLE != 0 || width % SIZE_MULTIPLE != 0 {
            return Err(OcrError::Runtime(format!(
                "the detector needs sides that are multiples of \
                 {SIZE_MULTIPLE}, got {height}x{width}"
            )));
        }
        match self {
            Self::Mobile(net) => net.forward(x),
            Self::Server(net) => net.forward(x),
        }
    }
}

/// A detector this port does not implement — a newer generation, most likely,
/// since the two here share their published names with everything else in the
/// family.
fn unknown_backbone(artifact: &Artifact) -> OcrError {
    OcrError::Artifact(format!(
        "this detector is built on {}, and trakktor runs {MOBILE_BACKBONE} \
         and {SERVER_BACKBONE}",
        artifact
            .modules()
            .first()
            .map_or("no module the graph names", String::as_str),
    ))
}
