//! Command-line interface: grammar, global options, and dispatch.
//!
//! Mirrors `conventions/cli.md`. Global options are declared `global = true`
//! so they may appear before or after the subcommand. Structural argument
//! errors are reported by clap with exit code 2; value-validation and runtime
//! errors return exit code 1 and respect `--json` (see `output.md`).

use std::path::PathBuf;

use clap::{
    Args, CommandFactory, Parser, Subcommand, ValueEnum, error::ErrorKind,
};
use trakktor_core::{feed, skill::Target};

use crate::{error::CliError, output};

/// Default working directory when neither `--work-dir` nor `TRAKKTOR_DIR` is
/// set (working-directory.md).
const DEFAULT_WORK_DIR: &str = ".trakktor";

// NOTE: every `///` on a clap item in this file is user-facing — clap renders
// it into `--help`, and `trakktor skill show` reproduces it verbatim for coding
// agents. Keep internal doc references (ADR-xxxx, design.md, §-sections) OUT of
// this prose; cite them in `//` comments or module docs instead. (Why the skill
// is generated from the binary at all: ADR-0003.)
/// Helper commands for coding agents: predictable, machine-readable building
/// blocks (RSS/Atom/JSON feeds today, more later) that an agent runs as a
/// tool. Use trakktor when a task needs one of these helpers — for example,
/// fetching and tracking the unread items of a feed.
///
/// Output is human-readable text by default and structured JSON with `--json`
/// (`--pretty` indents it); results go to stdout, errors to stderr. Exit codes
/// are stable: 0 success, 1 a runtime/validation error, 2 a usage error.
///
/// trakktor is self-documenting: run `trakktor skill show` for a guide to what
/// it does and when, and `trakktor skill show --full` for the complete,
/// always-current reference of every command, flag, and value — both generated
/// from this binary so they always match this version.
#[derive(Parser)]
#[command(name = "trakktor", version)]
pub struct Cli {
    #[command(flatten)]
    global: GlobalOpts,

    #[command(subcommand)]
    command: Command,
}

/// Options that apply to every command (cli.md).
#[derive(Args)]
struct GlobalOpts {
    /// Working directory for local state (default: ./.trakktor).
    #[arg(long, global = true, env = "TRAKKTOR_DIR", value_name = "path")]
    work_dir: Option<PathBuf>,

    /// Emit machine-readable JSON instead of text.
    #[arg(long, global = true)]
    json: bool,

    /// Pretty-print JSON (only with --json).
    #[arg(long, global = true)]
    pretty: bool,
}

impl GlobalOpts {
    /// Resolves the working directory: `--work-dir` > `TRAKKTOR_DIR` > default
    /// (working-directory.md). clap applies the flag-over-env precedence.
    fn work_dir(&self) -> PathBuf {
        self.work_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(DEFAULT_WORK_DIR))
    }
}

#[derive(Subcommand)]
enum Command {
    /// Work with RSS/Atom/JSON feeds: discover, read, and track read state.
    ///
    /// Typical workflow: `trakktor feed discover <page-url>` finds the feeds a
    /// page declares; `trakktor feed read <url>` returns the unread items (a
    /// feed URL or a regular page — autodiscovery applies), each carrying a
    /// stable `uid`; then `trakktor feed mark-read <uid>...` records them as
    /// read so the next `read` omits them. Read state lives under the working
    /// directory and needs no database.
    Feed {
        #[command(subcommand)]
        command: FeedCommand,
    },

    /// Generate and install the trakktor skill for coding agents.
    ///
    /// A skill tells an agent what trakktor does, when to use it, and how to
    /// call it, in the agent's native format. The installed file is a thin
    /// discovery stub that stays valid across releases because it only points
    /// back at the binary: an agent runs `trakktor skill show` for the guide
    /// and `trakktor skill show --full` for the complete reference, both
    /// generated from this exact version.
    Skill {
        #[command(subcommand)]
        command: SkillCommand,
    },
}

#[derive(Subcommand)]
enum FeedCommand {
    /// Find the feeds declared on a web page.
    Discover {
        /// URL of the web page to inspect.
        #[arg(value_name = "page-url")]
        page_url: String,
    },

    /// Read a feed and return its publications.
    ///
    /// Accepts a feed URL or a regular page (autodiscovery applies). By
    /// default only unread publications are returned.
    Read {
        /// URL of the feed or page.
        #[arg(value_name = "url")]
        url: String,

        /// Include already-read publications.
        #[arg(long)]
        all: bool,

        /// Fields to show (comma-separated), or `minimal`/`all`.
        ///
        /// Available fields: uid, is_read, title, link, published, updated,
        /// summary, content, authors. Special values: `minimal` (the default,
        /// = uid,title,link) and `all` (every field).
        #[arg(long, default_value = "minimal", value_name = "list")]
        fields: String,
    },

    /// Mark publications as read by uid (idempotent).
    MarkRead {
        /// One or more uids to mark as read.
        #[arg(required = true, value_name = "uid")]
        uids: Vec<String>,
    },
}

#[derive(Subcommand)]
enum SkillCommand {
    /// Print the trakktor skill to stdout as Markdown.
    ///
    /// Without `--full`, prints the narrative guide: what trakktor is, when to
    /// use it, and the typical workflows. With `--full`, also prints the
    /// complete reference of every command, flag, allowed value, and default.
    /// The content is generated from this binary, so it always matches the
    /// installed version. Under `--json` the Markdown is wrapped as an object
    /// `{ "content": "…" }`.
    Show {
        /// Also print the full command/flag/value reference.
        #[arg(long)]
        full: bool,
    },

    /// Install the skill stub into an agent's skills directory.
    ///
    /// The destination is stated explicitly. `trakktor skill install claude`
    /// writes `./.claude/skills/trakktor/SKILL.md`, `trakktor skill install
    /// agents` writes `./.agents/skills/trakktor/SKILL.md`, and `trakktor skill
    /// install claude --global` writes `~/.claude/skills/trakktor/SKILL.md`. A
    /// project install creates the whole path. A global install requires the
    /// `~/.claude` directory to already exist — it is never created, and the
    /// command fails if it is missing. `--global` is only valid with `claude`.
    /// An existing SKILL.md is left untouched, and reported as skipped, unless
    /// `--force` is given.
    Install {
        /// Agent directory layout to install into: claude or agents.
        #[arg(value_name = "target")]
        target: TargetArg,

        /// Install into the home directory (~/.claude) instead of the project;
        /// only valid with `claude`, and that directory must already exist.
        #[arg(long)]
        global: bool,

        /// Overwrite an existing SKILL.md instead of skipping it.
        #[arg(long)]
        force: bool,
    },
}

/// The `skill install <target>` value (maps to
/// [`trakktor_core::skill::Target`]).
#[derive(Clone, Copy, ValueEnum)]
enum TargetArg {
    /// `.claude` directory layout.
    Claude,
    /// `.agents` directory layout.
    Agents,
}

impl TargetArg {
    fn to_core(self) -> Target {
        match self {
            TargetArg::Claude => Target::Claude,
            TargetArg::Agents => Target::Agents,
        }
    }
}

/// Parses arguments and runs the requested command.
///
/// Returns the process exit code: 0 on success, 1 on a runtime/validation
/// error. (clap exits with 2 itself on structural argument errors.)
pub fn run() -> i32 {
    let cli = Cli::parse();
    match dispatch(&cli) {
        Ok(()) => 0,
        Err(err) => {
            output::emit_error(&err, cli.global.json, cli.global.pretty);
            1
        },
    }
}

/// Executes the parsed command, printing successful output to stdout. Feature
/// errors are converted to [`CliError`] at the `?` boundary
/// (error-handling.md).
fn dispatch(cli: &Cli) -> Result<(), CliError> {
    let global = &cli.global;
    match &cli.command {
        Command::Feed { command } => match command {
            FeedCommand::Discover { page_url } => {
                let feeds = feed::discover(page_url)?;
                output::print_discover(&feeds, global.json, global.pretty);
                Ok(())
            },
            FeedCommand::Read { url, all, fields } => {
                // Validate --fields before any network I/O (fail fast).
                let selection = feed::parse_fields(fields)?;
                let publications = feed::read(url, *all, &global.work_dir())?;
                output::print_read(
                    &publications,
                    &selection,
                    global.json,
                    global.pretty,
                );
                Ok(())
            },
            FeedCommand::MarkRead { uids } => {
                let summary = feed::mark_read(uids, &global.work_dir())?;
                output::print_mark_read(&summary, global.json, global.pretty);
                Ok(())
            },
        },
        Command::Skill { command } => match command {
            SkillCommand::Show { full } => {
                crate::skill::show(*full, global.json, global.pretty)
            },
            SkillCommand::Install {
                target,
                global: to_home,
                force,
            } => {
                // A global install targets the home Claude directory, which
                // only exists for `claude`; reject `agents --global` as a usage
                // error (exit 2), like clap's own argument-conflict errors.
                if *to_home && !matches!(target, TargetArg::Claude) {
                    Cli::command()
                        .error(
                            ErrorKind::ArgumentConflict,
                            "`--global` is only valid with the `claude` target",
                        )
                        .exit();
                }
                let opts = crate::skill::InstallOptions {
                    target: target.to_core(),
                    global: *to_home,
                    force: *force,
                };
                crate::skill::install(&opts, global.json, global.pretty)
            },
        },
    }
}
