//! `ocr paddle` — a port of PaddleOCR's PP-OCRv5 pipeline.
//!
//! The pipeline is two networks and a layer of geometry between them: a
//! differentiable-binarization detector turns the page into a probability map
//! and then into text-line quadrangles, each quadrangle is straightened into a
//! crop, and a CTC recognizer reads the crop. A tiny classifier in between
//! decides whether a crop is upside down.
//!
//! trakktor runs the published inference artifacts directly — the weights and
//! the graph as PaddleOCR publishes them — so there is no conversion step and
//! no ONNX runtime; see [`artifact`].

pub mod artifact;
pub mod cls;
pub mod config;
pub mod crop;
pub mod db;
pub mod det;
pub mod download;
pub mod image;
pub mod model;
pub mod net;
pub mod pipeline;
pub mod rec;
