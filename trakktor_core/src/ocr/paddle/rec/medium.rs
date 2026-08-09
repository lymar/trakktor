//! The newest recognizer's backbone: PP-LCNetV4 in its recognition shape.
//!
//! The [same network](crate::ocr::paddle::lcnetv4) the newest detector carries,
//! with the table that reads a text line instead of a page: the stem quarters
//! both axes, and after it only the height is spent, 12 → 6 → 3, while the
//! width stays where the stem left it. The closing pooling turns the three rows
//! into one and halves the width once more, so a crop 320 columns wide becomes
//! forty steps — the same sequence length the older recognizers produce, which
//! is what lets the pipeline hold any of them without knowing which.
//!
//! Its weights are taken in the order the graph reads them, for the reason
//! given in [`lcnetv4`](crate::ocr::paddle::lcnetv4).

use candle_core::Tensor;

use crate::ocr::{
    error::OcrError,
    paddle::{
        lcnetv4::{self, BlockSpec, block},
        net::{Loader, Order},
    },
};

/// The width this backbone hands to the head.
pub const WIDTH: usize = 768;

/// The stem's channels: half-width in the middle, full width out.
const STEM: (usize, usize) = (64, 128);

/// The four stages of the recognition table. Only the head of a stage strides,
/// and it strides the height alone.
const STAGE1: [BlockSpec; 1] = [block(128, 128, (1, 1), true)];
const STAGE2: [BlockSpec; 3] = [
    block(128, 256, (1, 1), false),
    block(256, 256, (1, 1), false),
    block(256, 256, (1, 1), true),
];
const STAGE3: [BlockSpec; 7] = [
    block(256, 512, (2, 1), false),
    block(512, 512, (1, 1), true),
    block(512, 512, (1, 1), false),
    block(512, 512, (1, 1), true),
    block(512, 512, (1, 1), false),
    block(512, 512, (1, 1), true),
    block(512, 512, (1, 1), false),
];
const STAGE4: [BlockSpec; 3] = [
    block(512, WIDTH, (2, 1), false),
    block(WIDTH, WIDTH, (1, 1), true),
    block(WIDTH, WIDTH, (1, 1), false),
];

/// The backbone.
#[derive(Debug)]
pub struct Net {
    backbone: lcnetv4::Backbone,
}

impl Net {
    pub fn load(loader: &Loader, order: &mut Order) -> Result<Self, OcrError> {
        let stages: [&[BlockSpec]; 4] = [&STAGE1, &STAGE2, &STAGE3, &STAGE4];
        Ok(Self {
            backbone: lcnetv4::Backbone::load(loader, order, STEM, 2, &stages)?,
        })
    }

    /// Runs the backbone over `[batch, 3, 48, width]` and returns the pooled
    /// feature map, `[batch, 768, 1, width / 8]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, OcrError> {
        let last = self.backbone.forward(x)?.pop().ok_or_else(|| {
            OcrError::Runtime("the backbone produced no stages".into())
        })?;
        // Three rows into one and two columns into one: the feature map
        // becomes a single row, and the sequence gets its length.
        Ok(last.avg_pool2d((3, 2))?)
    }
}
