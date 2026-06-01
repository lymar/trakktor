//! Command-line interface: grammar, global options, and dispatch.
//!
//! Mirrors `conventions/cli.md`. Global options are declared `global = true`
//! so they may appear before or after the subcommand. Structural argument
//! errors are reported by clap with exit code 2; value-validation and runtime
//! errors return exit code 1 and respect `--json` (see `output.md`).

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use trakktor_core::feed;

use crate::output;

/// Default working directory when neither `--work-dir` nor `TRAKKTOR_DIR` is
/// set (working-directory.md).
const DEFAULT_WORK_DIR: &str = ".trakktor";

/// Helper commands for coding agents: feeds, and more to come.
#[derive(Parser)]
#[command(name = "trakktor", version, about, long_about = None)]
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
    /// Work with RSS/Atom/JSON feeds.
    Feed {
        #[command(subcommand)]
        command: FeedCommand,
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

/// Executes the parsed command, printing successful output to stdout.
fn dispatch(cli: &Cli) -> Result<(), feed::FeedError> {
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
    }
}
