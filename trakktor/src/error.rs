//! Bin-boundary error type.
//!
//! `trakktor_core` returns typed, feature-specific errors; here, at the CLI
//! boundary, they are unified and mapped to the output contract — a stable
//! string `code` plus exit code 1. Changing a feature's `#[error]` message must
//! never change its external `code`.

use trakktor_core::{
    asr::{vad::VadError, whisper::WhisperError},
    feed::FeedError,
    http::HttpError,
    skill::SkillError,
};

/// Any runtime/validation error surfaced by a command (exit code 1).
#[derive(Debug)]
pub enum CliError {
    /// An error from the `asr` feature.
    Asr(WhisperError),
    /// An error from the VAD preprocessing stage.
    Vad(VadError),
    /// An error from the `feed` feature.
    Feed(FeedError),
    /// An error from the `skill` feature.
    Skill(SkillError),
}

impl CliError {
    /// The stable error `code` for the JSON error contract. The variant →
    /// `code` mapping lives here, at the bin boundary.
    pub fn code(&self) -> &'static str {
        match self {
            CliError::Asr(err) => asr_code(err),
            CliError::Vad(_) => "vad_failed",
            CliError::Feed(err) => feed_code(err),
            CliError::Skill(err) => skill_code(err),
        }
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliError::Asr(err) => write!(f, "{err}"),
            CliError::Vad(err) => write!(f, "{err}"),
            CliError::Feed(err) => write!(f, "{err}"),
            CliError::Skill(err) => write!(f, "{err}"),
        }
    }
}

impl From<WhisperError> for CliError {
    fn from(err: WhisperError) -> Self { CliError::Asr(err) }
}

impl From<VadError> for CliError {
    fn from(err: VadError) -> Self { CliError::Vad(err) }
}

impl From<FeedError> for CliError {
    fn from(err: FeedError) -> Self { CliError::Feed(err) }
}

impl From<SkillError> for CliError {
    fn from(err: SkillError) -> Self { CliError::Skill(err) }
}

/// Maps a [`WhisperError`] to its stable `code`.
fn asr_code(err: &WhisperError) -> &'static str {
    match err {
        WhisperError::AudioDecode(_) => "audio_decode_failed",
        WhisperError::UnsupportedLanguage(_) => "unsupported_language",
        WhisperError::InvalidModel(_) | WhisperError::ModelDownload(_) => {
            "model_unavailable"
        },
        WhisperError::InvalidOptions(_) => "invalid_options",
        WhisperError::Io(_) => "io_error",
        WhisperError::HomeDirUnknown => "no_home_dir",
    }
}

/// Maps a [`FeedError`] to its stable `code`.
fn feed_code(err: &FeedError) -> &'static str {
    match err {
        FeedError::Http(http) => match http {
            HttpError::InvalidUrl(_) => "invalid_url",
            HttpError::Fetch(_) => "fetch_failed",
            HttpError::Status(_) => "http_error",
            HttpError::TooLarge { .. } => "too_large",
        },
        FeedError::ParseFailed(_) => "parse_failed",
        FeedError::FeedNotFound => "feed_not_found",
        FeedError::InvalidUid(_) => "invalid_uid",
        FeedError::InvalidField(_) => "invalid_field",
        FeedError::Io(_) => "io_error",
    }
}

/// Maps a [`SkillError`] to its stable `code`.
fn skill_code(err: &SkillError) -> &'static str {
    match err {
        SkillError::HomeDirUnknown => "no_home_dir",
        SkillError::AgentDirMissing(_) => "agent_dir_missing",
        SkillError::Io(_) => "io_error",
    }
}
