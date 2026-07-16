//! File-based read-state store.
//!
//! Layout under the trakktor working directory:
//!
//! ```text
//! .trakktor/feed/<uid[0..2]>/<uid[2..3]>
//! ```
//!
//! 256 directories (first two hex chars) × 16 files (third char). Each file is
//! plain text: one uid per line. Membership — not count — is what matters, so
//! duplicate lines are harmless and deduplicated on read.
//!
//! `mark-read` appends in `O_APPEND` mode (atomic for a line far smaller than
//! `PIPE_BUF`); it never rewrites the whole file, which would lose updates when
//! different uids race into the same shard.

use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
};

use crate::feed::{error::FeedError, model::MarkReadSummary};

/// The read-state store rooted at `<work_dir>/feed`.
pub struct ReadStore {
    root: PathBuf,
}

impl ReadStore {
    /// Creates a handle to the store under the given working directory.
    ///
    /// The directory is not touched until [`ReadStore::mark_read`] writes to
    /// it; `read` never creates the store.
    #[must_use]
    pub fn new(work_dir: &Path) -> Self {
        Self {
            root: work_dir.join("feed"),
        }
    }

    /// Shard path for a uid. The uid must already be validated as 64 hex chars.
    fn shard_path(&self, uid: &str) -> PathBuf {
        self.root.join(&uid[0..2]).join(&uid[2..3])
    }

    /// Returns whether `uid` is recorded as read.
    ///
    /// A missing shard file means "not read"; the store is not created.
    ///
    /// # Errors
    ///
    /// Returns [`FeedError::Io`] on filesystem errors other than "not found".
    pub fn is_read(&self, uid: &str) -> Result<bool, FeedError> {
        match fs::read_to_string(self.shard_path(uid)) {
            Ok(contents) => Ok(contents.lines().any(|line| line.trim() == uid)),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
            Err(e) => Err(FeedError::Io(e)),
        }
    }

    /// Marks each uid as read; idempotent.
    ///
    /// All uids are validated up front, so a malformed argument fails the whole
    /// call without writing anything. Counts are best-effort under concurrency.
    ///
    /// # Errors
    ///
    /// - [`FeedError::InvalidUid`] if any argument is not a valid uid.
    /// - [`FeedError::Io`] on filesystem errors.
    pub fn mark_read(
        &self,
        uids: &[String],
    ) -> Result<MarkReadSummary, FeedError> {
        for uid in uids {
            validate_uid(uid)?;
        }

        let mut marked = 0;
        let mut already_read = 0;
        // Track uids written in this call so repeated arguments are counted as
        // already-read rather than appended twice.
        let mut written = HashSet::new();

        for uid in uids {
            if written.contains(uid.as_str()) || self.is_read(uid)? {
                already_read += 1;
                continue;
            }

            let path = self.shard_path(uid);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file =
                OpenOptions::new().create(true).append(true).open(&path)?;
            writeln!(file, "{uid}")?;

            written.insert(uid.clone());
            marked += 1;
        }

        Ok(MarkReadSummary {
            marked,
            already_read,
        })
    }
}

/// Validates a uid: exactly 64 lowercase hex characters.
///
/// # Errors
///
/// Returns [`FeedError::InvalidUid`] otherwise.
pub fn validate_uid(uid: &str) -> Result<(), FeedError> {
    let valid = uid.len() == 64 &&
        uid.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if valid {
        Ok(())
    } else {
        Err(FeedError::InvalidUid(uid.to_string()))
    }
}

#[cfg(test)]
mod tests;
