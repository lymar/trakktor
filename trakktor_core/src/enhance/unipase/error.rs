//! Failures of the UniPASE engine, in the shape the output contract needs.

/// What can go wrong in one enhancement run.
#[derive(Debug, thiserror::Error)]
pub enum UnipaseError {
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

    /// The recording could not be decoded.
    #[error("cannot read the recording: {0}")]
    Decode(String),

    /// The recording holds no audio at all.
    #[error("the recording is empty")]
    Empty,

    /// Running the network failed.
    #[error("enhancement failed: {0}")]
    Compute(String),

    /// Writing the enhanced audio failed.
    #[error("{0}")]
    Io(String),
}

impl From<crate::download::DownloadError> for UnipaseError {
    fn from(error: crate::download::DownloadError) -> Self {
        Self::ModelDownload(error.to_string())
    }
}

impl From<crate::audio::AudioError> for UnipaseError {
    fn from(error: crate::audio::AudioError) -> Self {
        Self::Decode(error.to_string())
    }
}
