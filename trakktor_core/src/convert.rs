//! Conversion: turning a document into text you can work with.
//!
//! A PDF made from a layout program already carries its text — every character
//! is in the file, with the font code that draws it. Nothing needs recognizing:
//! the answer comes out letter for letter and three orders of magnitude faster
//! than reading the page as a picture. This module is that path; a PDF that
//! came out of a scanner belongs to [`crate::ocr`] instead, and the two say so
//! about each other.
//!
//! Today the domain holds one format. `pdf` is where a second one would go.

pub mod error;
pub mod pdf;

pub use error::ConvertError;
