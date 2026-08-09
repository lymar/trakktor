//! Bin-boundary error type.
//!
//! `trakktor_core` returns typed, feature-specific errors; here, at the CLI
//! boundary, they are unified and mapped to the output contract — a stable
//! string `code` plus exit code 1. Changing a feature's `#[error]` message must
//! never change its external `code`.

use trakktor_core::{
    asr::{gigaam::GigaamError, vosk::VoskError, whisper::WhisperError},
    audio::AudioError,
    enhance::unipase::UnipaseError,
    feed::FeedError,
    http::HttpError,
    ocr::OcrError,
    punctuate::PunctuateError,
    skill::SkillError,
    stress::StressError,
    structify::StructifyError,
    tts::{
        espeech::EspeechError, qwen3_tts::Qwen3TtsError, silero::SileroError,
    },
    vad::VadError,
};

/// Any runtime/validation error surfaced by a command (exit code 1).
#[derive(Debug)]
pub enum CliError {
    /// An error from the `asr` feature.
    Asr(WhisperError),
    /// An error from the GigaAM engine.
    Gigaam(GigaamError),
    /// An error from the Vosk engine.
    Vosk(VoskError),
    /// An error from voice-activity detection (the model).
    Vad(VadError),
    /// An error from decoding or encoding audio (the `vad` feature).
    Audio(AudioError),
    /// An error from the `feed` feature.
    Feed(FeedError),
    /// An error from the `skill` feature.
    Skill(SkillError),
    /// An error from the `text structify` feature.
    Structify(StructifyError),
    /// An error from the `text punctuate` feature.
    Punctuate(PunctuateError),
    /// An error from the OCR feature.
    Ocr(OcrError),
    /// An error from the `text stress` feature.
    Stress(StressError),
    /// An error from the Qwen3-TTS engine.
    Qwen3Tts(Qwen3TtsError),
    /// An error from the ESpeech engine.
    Espeech(EspeechError),
    /// An error from the Silero engine.
    Silero(SileroError),
    /// An error from the UniPASE speech-enhancement engine.
    Unipase(UnipaseError),
}

impl CliError {
    /// The stable error `code` for the JSON error contract. The variant →
    /// `code` mapping lives here, at the bin boundary.
    pub fn code(&self) -> &'static str {
        match self {
            CliError::Asr(err) => asr_code(err),
            CliError::Gigaam(err) => gigaam_code(err),
            CliError::Vosk(err) => vosk_code(err),
            CliError::Vad(_) => "vad_failed",
            CliError::Audio(err) => audio_code(err),
            CliError::Feed(err) => feed_code(err),
            CliError::Skill(err) => skill_code(err),
            CliError::Structify(err) => structify_code(err),
            CliError::Punctuate(err) => punctuate_code(err),
            CliError::Ocr(err) => ocr_code(err),
            CliError::Stress(err) => stress_code(err),
            CliError::Qwen3Tts(err) => qwen3_tts_code(err),
            CliError::Espeech(err) => espeech_code(err),
            CliError::Silero(err) => silero_code(err),
            CliError::Unipase(err) => unipase_code(err),
        }
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CliError::Asr(err) => write!(f, "{err}"),
            CliError::Gigaam(err) => write!(f, "{err}"),
            CliError::Vosk(err) => write!(f, "{err}"),
            CliError::Vad(err) => write!(f, "{err}"),
            CliError::Audio(err) => write!(f, "{err}"),
            CliError::Feed(err) => write!(f, "{err}"),
            CliError::Skill(err) => write!(f, "{err}"),
            CliError::Structify(err) => write!(f, "{err}"),
            CliError::Punctuate(err) => write!(f, "{err}"),
            CliError::Ocr(err) => write!(f, "{err}"),
            CliError::Stress(err) => write!(f, "{err}"),
            CliError::Qwen3Tts(err) => write!(f, "{err}"),
            CliError::Espeech(err) => write!(f, "{err}"),
            CliError::Silero(err) => write!(f, "{err}"),
            CliError::Unipase(err) => write!(f, "{err}"),
        }
    }
}

impl From<WhisperError> for CliError {
    fn from(err: WhisperError) -> Self { CliError::Asr(err) }
}

impl From<GigaamError> for CliError {
    fn from(err: GigaamError) -> Self { CliError::Gigaam(err) }
}

impl From<VoskError> for CliError {
    fn from(err: VoskError) -> Self { CliError::Vosk(err) }
}

impl From<VadError> for CliError {
    fn from(err: VadError) -> Self { CliError::Vad(err) }
}

impl From<AudioError> for CliError {
    fn from(err: AudioError) -> Self { CliError::Audio(err) }
}

impl From<FeedError> for CliError {
    fn from(err: FeedError) -> Self { CliError::Feed(err) }
}

impl From<SkillError> for CliError {
    fn from(err: SkillError) -> Self { CliError::Skill(err) }
}

impl From<StructifyError> for CliError {
    fn from(err: StructifyError) -> Self { CliError::Structify(err) }
}

impl From<PunctuateError> for CliError {
    fn from(err: PunctuateError) -> Self { CliError::Punctuate(err) }
}

impl From<OcrError> for CliError {
    fn from(err: OcrError) -> Self { CliError::Ocr(err) }
}

impl From<StressError> for CliError {
    fn from(err: StressError) -> Self { CliError::Stress(err) }
}

impl From<EspeechError> for CliError {
    fn from(err: EspeechError) -> Self { CliError::Espeech(err) }
}

impl From<SileroError> for CliError {
    fn from(err: SileroError) -> Self { CliError::Silero(err) }
}

impl From<UnipaseError> for CliError {
    fn from(err: UnipaseError) -> Self { CliError::Unipase(err) }
}

impl From<Qwen3TtsError> for CliError {
    fn from(err: Qwen3TtsError) -> Self { CliError::Qwen3Tts(err) }
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

/// Maps a [`GigaamError`] to its stable `code`.
fn gigaam_code(err: &GigaamError) -> &'static str {
    match err {
        GigaamError::AudioDecode(_) => "audio_decode_failed",
        GigaamError::InvalidModel(_) | GigaamError::ModelDownload(_) => {
            "model_unavailable"
        },
        GigaamError::InvalidOptions(_) => "invalid_options",
        GigaamError::UnsupportedLanguage(_) => "unsupported_language",
        GigaamError::Vad(_) => "vad_failed",
        GigaamError::Io(_) => "io_error",
        GigaamError::HomeDirUnknown => "no_home_dir",
    }
}

/// Maps a [`VoskError`] to its stable `code`.
fn vosk_code(err: &VoskError) -> &'static str {
    match err {
        VoskError::AudioDecode(_) => "audio_decode_failed",
        VoskError::InvalidModel(_) | VoskError::ModelDownload(_) => {
            "model_unavailable"
        },
        VoskError::InvalidOptions(_) => "invalid_options",
        VoskError::Vad(_) => "vad_failed",
        VoskError::Io(_) => "io_error",
        VoskError::HomeDirUnknown => "no_home_dir",
    }
}

/// Maps an [`AudioError`] to its stable `code`.
fn audio_code(err: &AudioError) -> &'static str {
    match err {
        AudioError::UnsupportedFormat { .. } |
        AudioError::NoAudioTrack |
        AudioError::Decode(_) |
        AudioError::UnsupportedLayout(_) => "audio_decode_failed",
        AudioError::UnsupportedEncoding(_) => "unsupported_encoding",
        AudioError::Io { .. } | AudioError::Encode { .. } => "io_error",
        AudioError::Ffmpeg(_) => "ffmpeg_failed",
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

/// Maps a [`StructifyError`] to its stable `code`.
fn structify_code(err: &StructifyError) -> &'static str {
    match err {
        StructifyError::InvalidModel(_) |
        StructifyError::ModelDownload(_) |
        StructifyError::Tokenizer(_) => "model_unavailable",
        StructifyError::InvalidOptions(_) => "invalid_options",
        StructifyError::Io(_) => "io_error",
        StructifyError::HomeDirUnknown => "no_home_dir",
    }
}

/// Maps an [`OcrError`] to its stable `code`.
fn ocr_code(err: &OcrError) -> &'static str {
    match err {
        OcrError::ImageRead { .. } |
        OcrError::NoPages |
        OcrError::PageNotFound { .. } |
        OcrError::EmptyPage { .. } => "invalid_input",
        OcrError::UnsupportedLanguage { .. } => "unsupported_language",
        OcrError::UnknownModel { .. } |
        OcrError::ModelDownload(_) |
        OcrError::Artifact(_) |
        OcrError::ModelFile { .. } => "model_unavailable",
        OcrError::InvalidOptions(_) => "invalid_options",
        OcrError::Write { .. } => "io_error",
        OcrError::Runtime(_) => "runtime_error",
    }
}

/// Maps a [`PunctuateError`] to its stable `code`.
fn punctuate_code(err: &PunctuateError) -> &'static str {
    match err {
        PunctuateError::InvalidModel(_) |
        PunctuateError::ModelDownload(_) |
        PunctuateError::Tokenizer(_) => "model_unavailable",
        PunctuateError::InvalidOptions(_) => "invalid_options",
        PunctuateError::Io(_) => "io_error",
        PunctuateError::HomeDirUnknown => "no_home_dir",
    }
}

/// Maps a [`StressError`] to its stable `code`.
fn stress_code(err: &StressError) -> &'static str {
    match err {
        StressError::InvalidModel(_) |
        StressError::ModelDownload(_) |
        StressError::Tokenizer(_) => "model_unavailable",
        StressError::InvalidOptions(_) | StressError::Dictionary(_) => {
            "invalid_options"
        },
        StressError::Io(_) => "io_error",
        StressError::HomeDirUnknown => "no_home_dir",
    }
}

/// Maps an [`EspeechError`] to its stable `code`.
fn espeech_code(err: &EspeechError) -> &'static str {
    match err {
        EspeechError::TextEmpty => "text_empty",
        EspeechError::RefTextRequired => "ref_text_required",
        EspeechError::RefAudioDecode(_) => "ref_audio_decode_failed",
        EspeechError::RefAudioTooShort(_) => "ref_audio_too_short",
        EspeechError::UnsupportedLanguage(_) => "unsupported_language",
        EspeechError::InvalidModel(_) |
        EspeechError::ModelDownload(_) |
        EspeechError::Checkpoint(_) => "model_unavailable",
        EspeechError::InvalidOptions(_) => "invalid_options",
        EspeechError::Io(_) => "io_error",
    }
}

/// Maps a [`SileroError`] to its stable `code`.
fn silero_code(err: &SileroError) -> &'static str {
    match err {
        SileroError::TextEmpty => "text_empty",
        SileroError::TextTooLong { .. } => "text_too_long",
        SileroError::LicenseRestricted { .. } => "model_license_restricted",
        SileroError::InvalidModel(_) |
        SileroError::ModelDownload(_) |
        SileroError::Checkpoint(_) => "model_unavailable",
        SileroError::UnknownVoice { .. } => "unsupported_voice",
        SileroError::InvalidOptions(_) => "invalid_options",
        SileroError::Io(_) => "io_error",
    }
}

/// Maps a [`Qwen3TtsError`] to its stable `code`.
fn qwen3_tts_code(err: &Qwen3TtsError) -> &'static str {
    match err {
        Qwen3TtsError::TextEmpty => "text_empty",
        Qwen3TtsError::InvalidModel(_) |
        Qwen3TtsError::ModelDownload(_) |
        Qwen3TtsError::Tokenizer(_) => "model_unavailable",
        Qwen3TtsError::UnsupportedVoice(_) => "unsupported_voice",
        Qwen3TtsError::UnsupportedLanguage(_) => "unsupported_language",
        Qwen3TtsError::InvalidOptions(_) => "invalid_options",
        Qwen3TtsError::Io(_) => "io_error",
        Qwen3TtsError::HomeDirUnknown => "no_home_dir",
    }
}

/// Maps a [`UnipaseError`] to its stable `code`.
fn unipase_code(err: &UnipaseError) -> &'static str {
    match err {
        UnipaseError::InvalidModel(_) => "invalid_model",
        UnipaseError::ModelDownload(_) => "model_download_failed",
        UnipaseError::Checkpoint(_) => "checkpoint_invalid",
        UnipaseError::InvalidOptions(_) => "invalid_options",
        UnipaseError::Decode(_) => "audio_decode_failed",
        UnipaseError::Empty => "audio_empty",
        UnipaseError::Compute(_) => "enhance_failed",
        UnipaseError::Io(_) => "io_error",
    }
}
