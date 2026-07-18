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

    /// The requested output encoding cannot represent this audio losslessly
    /// (for example FLAC cannot hold float PCM or more than 24 bits).
    #[error("cannot encode losslessly: {0}")]
    UnsupportedEncoding(String),

    /// Writing the encoded output failed.
    #[error("cannot write '{path}': {message}")]
    Encode {
        /// The file that could not be written.
        path: PathBuf,
        /// The underlying encoder or I/O error, rendered.
        message: String,
    },

    /// Encoding through an external `ffmpeg` process failed (not installed, or
    /// it rejected the format or exited with an error).
    #[error("ffmpeg encoding failed: {0}")]
    Ffmpeg(String),
}

/// The human-readable support list used in [`AudioError::UnsupportedFormat`].
pub const SUPPORTED_FORMATS: &str =
    "mp3, aac (LC), vorbis, flac, alac, adpcm, pcm in wav/aiff/caf/ogg/mp4/mkv";
