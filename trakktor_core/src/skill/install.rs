//! Installing the skill stub into an agent's skills directory.
//!
//! This layer computes the canonical paths and writes the stub. *Where* to
//! install — the agent layout (`claude`/`agents`) and whether it is a project
//! or a global install — is decided by the CLI crate. A project install
//! creates the whole path; a global install requires the agent's directory to
//! already exist and never creates it (see [`global_skill_path`]).

use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::skill::error::SkillError;

/// An install target: which agent directory family to write into. `.claude`
/// is the confirmed layout; `.agents` is preliminary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// `.claude` (confirmed layout).
    Claude,
    /// `.agents` (preliminary — format not yet verified).
    Agents,
}

impl Target {
    /// Every target, in display order.
    #[must_use]
    pub fn all() -> &'static [Target] { &[Target::Claude, Target::Agents] }

    /// The on-disk agent directory name (e.g. `.claude`).
    #[must_use]
    pub fn dir_name(self) -> &'static str {
        match self {
            Target::Claude => ".claude",
            Target::Agents => ".agents",
        }
    }

    /// The `--target` token (e.g. `claude`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Target::Claude => "claude",
            Target::Agents => "agents",
        }
    }
}

/// The agent directory for `target` under `base` (e.g. `<base>/.claude`).
///
/// Used for detection: a target is "present" when this directory exists.
#[must_use]
pub fn agent_dir(base: &Path, target: Target) -> PathBuf {
    base.join(target.dir_name())
}

/// The `SKILL.md` path for `target` under `base`
/// (`<base>/.claude/skills/trakktor/SKILL.md`).
///
/// The `trakktor` directory leaf is the skill name and must match the stub's
/// frontmatter `name` — both are the binary's own name.
#[must_use]
pub fn skill_path(base: &Path, target: Target) -> PathBuf {
    agent_dir(base, target)
        .join("skills")
        .join("trakktor")
        .join("SKILL.md")
}

/// The `SKILL.md` path for a *global* install of `target` under `home`,
/// requiring the agent directory (e.g. `~/.claude`) to already exist.
///
/// A global install must never create an agent's home directory itself: if
/// [`agent_dir`] does not exist, this returns
/// [`SkillError::AgentDirMissing`]. The intermediate `skills/trakktor/`
/// directories are created later, when the stub is written ([`write_stub`]).
///
/// # Errors
///
/// Returns [`SkillError::AgentDirMissing`] when the agent directory is absent.
pub fn global_skill_path(
    home: &Path,
    target: Target,
) -> Result<PathBuf, SkillError> {
    let dir = agent_dir(home, target);
    if !dir.is_dir() {
        return Err(SkillError::AgentDirMissing(dir));
    }
    Ok(skill_path(home, target))
}

/// Result of attempting to write one stub.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteOutcome {
    /// The stub was written (created or overwritten).
    Written,
    /// A `SKILL.md` already existed and `force` was not set; left untouched.
    Skipped,
}

/// Writes the stub to `path`, creating parent directories.
///
/// If `path` already exists and `force` is false the file is left untouched and
/// [`WriteOutcome::Skipped`] is returned — the caller decides whether to prompt
/// for overwrite.
///
/// # Errors
///
/// Returns [`SkillError::Io`] if a directory or the file cannot be written.
pub fn write_stub(
    path: &Path,
    content: &str,
    force: bool,
) -> Result<WriteOutcome, SkillError> {
    if path.exists() && !force {
        return Ok(WriteOutcome::Skipped);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content)?;
    Ok(WriteOutcome::Written)
}

#[cfg(test)]
mod tests;
