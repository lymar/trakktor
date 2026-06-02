//! Skill: generate and install the trakktor skill for coding agents.
//!
//! Full specification: `../trakktor_project/docs/features/skill/design.md`.
//! Key decision: the content is generated from the live `clap::Command` — the
//! single source of truth for commands and flags — so it always matches the
//! installed binary, and the on-disk stub only points back at the binary
//! (ADR-0003). This module renders that content and writes the stub; the CLI
//! crate decides *where* to install and formats the result.

pub mod error;
pub mod install;
pub mod render;

pub use error::SkillError;
pub use install::{
    Target, WriteOutcome, agent_dir, global_skill_path, skill_path, write_stub,
};
pub use render::{render_guide, render_stub};
