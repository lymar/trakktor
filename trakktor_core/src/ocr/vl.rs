//! `ocr vl` — a port of PaddleOCR-VL 1.6.
//!
//! Where the classic engine classifies characters out of a fixed dictionary,
//! this one **writes** the answer: a vision tower turns the picture into
//! tokens, a projector folds them into the decoder's width, and an ERNIE-4.5
//! decoder generates the text one token at a time. That buys three things the
//! classic pipeline cannot have at any effort — a single model for every
//! writing system it knows (including Tibetan, which has no dictionary
//! anywhere in the classic line), tables and formulas as structure rather than
//! as lines, and a reader that is not told in advance what language it is
//! looking at.
//!
//! It costs 1.92 GB of weights and tens of seconds a page, so it is a choice,
//! not a default.
//!
//! The model is driven **block by block**, and that is the engine's one real
//! design decision: measured, a whole page collapses into a repeat loop and a
//! single mixed-script strip hallucinates alphabets, while a coherent block of
//! lines is read well. Blocks are assembled from the classic detector's boxes
//! — see [`blocks`].
//!
//! # Credits
//!
//! Ported from **PaddleOCR-VL 1.6** by the PaddlePaddle authors (Apache-2.0):
//! the vision tower, the projector, the image preprocessing and the prompt
//! format, over the **ERNIE-4.5** decoder by Baidu (Apache-2.0). Where the
//! checkpoint's own published code and the native implementation in
//! **Transformers** by Hugging Face (Apache-2.0) disagree — the interpolation
//! of the vision tower's learned position grid — this port follows
//! Transformers, because that is the reference every measurement was taken
//! against. **oar-ocr** (Apache-2.0) was consulted as prior art for running the
//! same checkpoint on candle.

#[cfg(test)]
mod tests;

pub mod blocks;
pub mod config;
pub mod download;
pub mod ernie;
pub mod generate;
pub mod image;
pub mod model;
pub mod pipeline;
pub mod vision;

pub use generate::{Answer, Limits, Task};
pub use pipeline::{Engine, Options, Read};
