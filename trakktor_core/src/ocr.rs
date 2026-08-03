//! OCR: reading the text off a page image.
//!
//! The domain is organized like [`crate::asr`]: a set of engines, each a
//! self-contained port of a published pipeline, behind one common result
//! shape. A run takes one or more page images and returns, per page, the text
//! lines it found — each with its quadrangle, its confidence and the order it
//! reads in — plus the assembled text.

pub mod error;
pub mod figures;
pub mod layout;
pub mod markdown;
pub mod paddle;
pub mod page;
pub mod vl;

pub use error::OcrError;
pub use page::{Line, Page, Quad};
