//! Error type of the audio subsystem.

use std::path::PathBuf;

use thiserror::Error;

/// Errors produced while decoding or converting audio.
#[derive(Debug, Error)]
pub enum AudioError {
    /// The file could not be opened or read.
    #[error("cannot read '{path}': {source}")]
    Io {
        /// The file that failed to open or read.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// The container or codec is not supported by the built-in decoder.
    #[error("unsupported media format ({detail}); supported: {supported}")]
    UnsupportedFormat {
        /// What was detected (or why detection failed).
        detail: String,
        /// A short human-readable list of supported formats.
        supported: &'static str,
    },

    /// The container holds no decodable audio track.
    #[error("no decodable audio track")]
    NoAudioTrack,

    /// The stream is damaged beyond recovery.
    #[error("audio decode failed: {0}")]
    Decode(String),

    /// The channel layout cannot be mixed to the requested output.
    #[error("unsupported channel layout: {0}")]
    UnsupportedLayout(String),
}

/// The human-readable support list used in [`AudioError::UnsupportedFormat`].
pub const SUPPORTED_FORMATS: &str =
    "mp3, aac (LC), vorbis, flac, alac, adpcm, pcm in wav/aiff/caf/ogg/mp4/mkv";
