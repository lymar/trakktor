//! `trakktor skill`: print the generated guide and install the on-disk stub.
//!
//! Thin orchestration over `trakktor_core::skill`: the core renders content
//! from the live `clap::Command` and writes files; this layer decides *where*
//! to install — the agent layout and the project-vs-global scope, both stated
//! explicitly on the command line — and formats the result.

use std::path::{Path, PathBuf};

use clap::CommandFactory;
use trakktor_core::skill::{self, SkillError, Target};

use crate::{cli::Cli, error::CliError, output};

/// The project (CWD-relative) base for project-scope installs. Reported paths
/// then read as `./.claude/skills/trakktor/SKILL.md`.
const PROJECT_BASE: &str = ".";

/// `trakktor skill show [--full]`. Content is generated from the live command,
/// so it always matches this binary version.
pub fn show(full: bool, json: bool, pretty: bool) -> Result<(), CliError> {
    let command = Cli::command();
    let content = skill::render_guide(&command, full);
    output::print_skill_show(&content, json, pretty);
    Ok(())
}

/// Parsed `skill install` options, kept independent of clap types.
pub struct InstallOptions {
    /// Agent directory layout to install into.
    pub target: Target,
    /// `--global`: install under the home directory instead of the project.
    pub global: bool,
    /// `--force`: overwrite an existing `SKILL.md` instead of skipping it.
    pub force: bool,
}

/// `trakktor skill install <target> [--global] [--force]`.
///
/// Resolves the single destination, writes the stub there (creating the path
/// for a project install), and prints the outcome.
pub fn install(
    opts: &InstallOptions,
    json: bool,
    pretty: bool,
) -> Result<(), CliError> {
    let stub = skill::render_stub(&Cli::command());
    let path = resolve_path(opts)?;
    let outcome = skill::write_stub(&path, &stub, opts.force)?;
    output::print_skill_install(&path, outcome, json, pretty);
    Ok(())
}

/// The `SKILL.md` path for the requested destination.
///
/// A project install (the default) creates the whole path when the stub is
/// written. A global install (`--global`) targets `~/<agent-dir>` and requires
/// that directory to already exist — [`skill::global_skill_path`] returns an
/// error otherwise, since we never create an agent's home directory ourselves.
fn resolve_path(opts: &InstallOptions) -> Result<PathBuf, CliError> {
    if opts.global {
        let home = std::env::home_dir().ok_or(SkillError::HomeDirUnknown)?;
        Ok(skill::global_skill_path(&home, opts.target)?)
    } else {
        Ok(skill::skill_path(Path::new(PROJECT_BASE), opts.target))
    }
}
