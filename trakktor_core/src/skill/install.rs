//! Installing the skill stub into an agent's skills directory (design.md §5).
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

/// An install target: which agent directory family to write into (design.md
/// §5). `.claude` is the confirmed layout; `.agents` is preliminary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// `.claude` (confirmed layout, research.md).
    Claude,
    /// `.agents` (preliminary — format not yet verified, design.md §5).
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
/// (`<base>/.claude/skills/trakktor/SKILL.md`, design.md §5).
///
/// The `trakktor` directory leaf is the skill name and must match the stub's
/// frontmatter `name` (design.md §6) — both are the binary's own name.
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
/// A global install must never create an agent's home directory itself
/// (design.md §5): if [`agent_dir`] does not exist, this returns
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

/// Writes the stub to `path`, creating parent directories (design.md §5.3).
///
/// If `path` already exists and `force` is false the file is left untouched and
/// [`WriteOutcome::Skipped`] is returned — the caller decides whether to prompt
/// for overwrite (design.md §5.4).
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
mod tests {
    use super::*;

    #[test]
    fn skill_path_follows_the_design_layout() {
        let path = skill_path(Path::new("."), Target::Claude);
        assert_eq!(
            path,
            Path::new("./.claude/skills/trakktor/SKILL.md").to_path_buf()
        );
        assert_eq!(
            skill_path(Path::new("/home/u"), Target::Agents),
            Path::new("/home/u/.agents/skills/trakktor/SKILL.md").to_path_buf()
        );
    }

    #[test]
    fn write_creates_dirs_then_skips_without_force() {
        let dir = tempfile::tempdir().unwrap();
        let path = skill_path(dir.path(), Target::Claude);

        assert_eq!(
            write_stub(&path, "first", false).unwrap(),
            WriteOutcome::Written
        );
        assert_eq!(fs::read_to_string(&path).unwrap(), "first");

        // Existing file, no force → skipped, content untouched.
        assert_eq!(
            write_stub(&path, "second", false).unwrap(),
            WriteOutcome::Skipped
        );
        assert_eq!(fs::read_to_string(&path).unwrap(), "first");

        // force → overwritten.
        assert_eq!(
            write_stub(&path, "second", true).unwrap(),
            WriteOutcome::Written
        );
        assert_eq!(fs::read_to_string(&path).unwrap(), "second");
    }

    #[test]
    fn global_skill_path_requires_an_existing_agent_dir() {
        let home = tempfile::tempdir().unwrap();

        // Missing `.claude` → error, no path produced.
        match global_skill_path(home.path(), Target::Claude) {
            Err(SkillError::AgentDirMissing(dir)) => {
                assert_eq!(dir, agent_dir(home.path(), Target::Claude));
            },
            other => panic!("expected AgentDirMissing, got {other:?}"),
        }

        // Once `.claude` exists, the usual layout is returned.
        fs::create_dir(agent_dir(home.path(), Target::Claude)).unwrap();
        assert_eq!(
            global_skill_path(home.path(), Target::Claude).unwrap(),
            skill_path(home.path(), Target::Claude)
        );
    }
}
