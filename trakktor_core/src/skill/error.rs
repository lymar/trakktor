//! Typed errors for the skill feature.
//!
//! Each variant maps to a stable error `code` at the CLI boundary (see
//! `conventions/error-handling.md` and skill design.md §8); changing an
//! `#[error]` message must never change the external `code`.

use std::path::PathBuf;

/// Errors returned by skill operations.
#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    /// A `--global` install was requested but the home directory could not be
    /// determined (`no_home_dir`).
    #[error("could not determine the home directory for a global install")]
    HomeDirUnknown,

    /// A global install was requested but the agent's directory does not exist
    /// (`agent_dir_missing`). A global install never creates the agent's home
    /// directory itself (design.md §5); the user must create it first.
    #[error(
        "the agent directory {} does not exist; a global install does not \
         create it",
        .0.display()
    )]
    AgentDirMissing(PathBuf),

    /// A filesystem error while writing the stub (`io_error`).
    #[error("filesystem error: {0}")]
    Io(#[from] std::io::Error),
}
