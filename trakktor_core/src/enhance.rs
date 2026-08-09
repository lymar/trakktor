//! Speech enhancement: a recording in, a cleaner recording out.
//!
//! The youngest of the speech domains, and the only one whose input and output
//! are both audio. Where [`asr`](crate::asr) turns a recording into text and
//! [`tts`](crate::tts) text into a recording, this one repairs the recording
//! itself: room noise, reverberation, a telephone band, the holes a dropped
//! packet leaves in a call.
//!
//! It has two consumers, and they want different things from it. A person wants
//! a file back — the same recording, cleaner. A recogniser wants a
//! preprocessing stage, and cares only about the 16 kHz mono signal the
//! acoustic model will see. Both are served by the same engine; the second one
//! simply stops before the bandwidth is put back.
//!
//! Engines live one per module, the way they do under [`asr`](crate::asr) and
//! [`tts`](crate::tts):
//!
//! - [`unipase`] — a four-network generative pipeline around a fine-tuned WavLM
//!   encoder.

pub mod unipase;
