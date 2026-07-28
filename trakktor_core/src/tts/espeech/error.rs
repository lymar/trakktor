//! Failures of the ESpeech engine, in the shape the output contract needs.

/// What can go wrong in one synthesis run.
#[derive(Debug, thiserror::Error)]
pub enum EspeechError {
    /// The text to speak is empty.
    #[error("the text to speak is empty")]
    TextEmpty,

    /// The reference transcript is missing or blank.
    #[error(
        "the reference recording needs its transcript: pass --ref-text (or \
         --ref-text-file) with what is said in it"
    )]
    RefTextRequired,

    /// The reference recording could not be decoded.
    #[error("cannot read the reference recording: {0}")]
    RefAudioDecode(String),

    /// The reference recording holds too little speech to condition on.
    #[error(
        "the reference recording is too short ({0:.2} s of audio): it sets \
         both the voice and the speech rate, so give it a few seconds of \
         speech"
    )]
    RefAudioTooShort(f64),

    /// The requested language is not one the checkpoints speak.
    #[error("{0}")]
    UnsupportedLanguage(String),

    /// The model name is neither a published variant nor a checkpoint
    /// directory.
    #[error("{0}")]
    InvalidModel(String),

    /// Fetching or converting a checkpoint failed.
    #[error("cannot fetch the model: {0}")]
    ModelDownload(String),

    /// The checkpoint is not what the engine expects.
    #[error("the checkpoint cannot be loaded: {0}")]
    Checkpoint(String),

    /// Options that cannot be combined.
    #[error("{0}")]
    InvalidOptions(String),

    /// Reading the input text or writing the output failed.
    #[error("{0}")]
    Io(String),
}
