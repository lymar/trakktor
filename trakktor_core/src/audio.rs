//! Built-in audio decoding and PCM shaping.
//!
//! Turns a media file into PCM of a requested shape. Decoding is delegated to
//! [symphonia]; the sample-rate/channel/format conversion pipeline
//! reimplements the numeric behavior of ffmpeg's libswresample so that the
//! result is faithful to the widely used
//! `ffmpeg -i <file> -f s16le -ac 1 -ar 16000 -` reference chain: identical
//! for lossless inputs, within ±1 least-significant bit on a small fraction
//! of samples for lossy codecs (where the decoders themselves differ).
//!
//! [symphonia]: https://github.com/pdeljanov/Symphonia

mod convert;
mod decode;
pub mod edit;
pub mod encode;
mod error;
mod pipeline;
mod rematrix;
mod resample;

pub use decode::{
    DecodedAudio, decode_file, decode_to_mono_f32, decode_to_mono_s16,
};
pub use error::AudioError;
pub use pipeline::SampleFormat;
