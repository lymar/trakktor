//! The large recognizer's backbone: PP-HGNetV2 in its recognition shape.
//!
//! It is the [same backbone](crate::ocr::paddle::hgnet) the large detector and
//! the layout model carry — the same stem, the same four stages, the same
//! channel counts — asked to read a text line instead of a page, so the height
//! falls 48 → 3 while the width stops at a quarter. Nothing else about it is
//! new, which is the whole reason this recognizer costs a file rather than a
//! port.
//!
//! Where it does differ from the detector is the numbering above it: the
//! backbone claims eighty-one convolutions and the head starts at the
//! eighty-*third*. The one in between is the classification layer every
//! PP-HGNetV2 declares and only a classifier runs; the exporter took its
//! number and dropped its weights. The same gap opens one lower in the
//! normalizations, and one in the projections, where an unused fully connected
//! layer took `linear_0`.

use candle_core::Tensor;

use super::head::Naming;
use crate::ocr::{
    error::OcrError,
    paddle::{hgnet, net::Loader},
};

/// The width this backbone hands to the head, and where the head's own weights
/// are numbered from.
pub const NAMING: Naming = Naming {
    width: 2048,
    conv: 82,
    norm: 81,
    linear: 1,
};

/// The convolutions the backbone itself claims, counted so that the gap above
/// it is a stated fact rather than an accident of a counter.
const BACKBONE_CONVS: usize = 81;

/// The backbone.
#[derive(Debug)]
pub struct Net {
    backbone: hgnet::Backbone,
}

impl Net {
    pub fn load(loader: &Loader) -> Result<Self, OcrError> {
        let mut next = 0usize;
        let backbone =
            hgnet::Backbone::load(loader, &mut next, 0, hgnet::Shape::Line)?;
        if next != BACKBONE_CONVS {
            return Err(OcrError::Artifact(format!(
                "the recognition backbone claimed {next} convolutions, not \
                 {BACKBONE_CONVS}, so the head above it is numbered elsewhere"
            )));
        }
        Ok(Self { backbone })
    }

    /// Runs the backbone over `[batch, 3, 48, width]` and returns the pooled
    /// feature map, `[batch, 2048, 1, width / 8]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        // Only the coarsest stage is wanted; the finer three are what the
        // detector reads off the same backbone, and they go here.
        let last = self.backbone.forward(x)?.pop().ok_or_else(|| {
            OcrError::Runtime("the backbone produced no stages".into())
        })?;
        // Three rows into one and two columns into one: the feature map
        // becomes a single row, and the sequence gets its length.
        Ok(last.avg_pool2d((3, 2))?)
    }
}
