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
mod tests {
    use super::*;

    // A valid-looking uid (64 lowercase hex chars).
    const UID_A: &str =
        "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899";
    const UID_B: &str =
        "0011223344556677889900112233445566778899001122334455667788990011";

    #[test]
    fn validate_uid_accepts_64_lower_hex() {
        assert!(validate_uid(UID_A).is_ok());
    }

    #[test]
    fn validate_uid_rejects_bad_values() {
        assert!(validate_uid("").is_err());
        assert!(validate_uid("abc").is_err());
        // Uppercase is rejected (must be lowercase).
        assert!(validate_uid(&UID_A.to_uppercase()).is_err());
        // Wrong length.
        assert!(validate_uid(&UID_A[..63]).is_err());
        // Non-hex character.
        let mut bad = UID_A.to_string();
        bad.replace_range(0..1, "g");
        assert!(validate_uid(&bad).is_err());
    }

    #[test]
    fn mark_read_is_idempotent_and_counts() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReadStore::new(dir.path());

        // Nothing is read initially; the store is not created by `is_read`.
        assert!(!store.is_read(UID_A).unwrap());
        assert!(!dir.path().join("feed").exists());

        let s1 = store.mark_read(&[UID_A.to_string()]).unwrap();
        assert_eq!(s1.marked, 1);
        assert_eq!(s1.already_read, 0);
        assert!(store.is_read(UID_A).unwrap());

        // Re-marking is a no-op.
        let s2 = store.mark_read(&[UID_A.to_string()]).unwrap();
        assert_eq!(s2.marked, 0);
        assert_eq!(s2.already_read, 1);
    }

    #[test]
    fn mark_read_dedups_repeated_arguments() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReadStore::new(dir.path());
        let summary = store
            .mark_read(&[
                UID_A.to_string(),
                UID_A.to_string(),
                UID_B.to_string(),
            ])
            .unwrap();
        assert_eq!(summary.marked, 2);
        assert_eq!(summary.already_read, 1);
    }

    #[test]
    fn mark_read_validates_before_writing() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReadStore::new(dir.path());
        // A bad uid in the batch fails the whole call without side effects.
        let err = store
            .mark_read(&[UID_A.to_string(), "bad".to_string()])
            .unwrap_err();
        assert!(matches!(err, FeedError::InvalidUid(_)));
        assert!(!store.is_read(UID_A).unwrap());
    }

    #[test]
    fn shard_path_uses_documented_layout() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReadStore::new(dir.path());
        let expected = dir.path().join("feed").join("aa").join("b");
        assert_eq!(
            store.shard_path(
                "aab0000000000000000000000000000000000000000000000000000000000000"
            ),
            expected
        );
    }
}
