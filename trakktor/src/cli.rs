//! Command-line interface: grammar, global options, and dispatch.
//!
//! Global options are declared `global = true` so they may appear before or
//! after the subcommand. Structural argument errors are reported by clap with
//! exit code 2; value-validation and runtime errors return exit code 1 and are
//! formatted like normal output (JSON by default, text under `--text`).

use std::path::PathBuf;

use clap::{
    Args, CommandFactory, Parser, Subcommand, ValueEnum, error::ErrorKind,
};
use trakktor_core::{feed, skill::Target};

use crate::{error::CliError, output};

/// Default working directory when neither `--work-dir` nor `TRAKKTOR_DIR` is
/// set.
const DEFAULT_WORK_DIR: &str = ".trakktor";

/// Helper commands for coding agents: predictable, machine-readable building
/// blocks (web feeds and speech-to-text today, more later) that an agent runs
/// as a tool. Use trakktor when a task needs one of these helpers — for
/// example, fetching the unread items of a feed, or transcribing an audio file
/// to timestamped text.
///
/// Output is JSON by default (`--pretty` indents it); pass `--text` for
/// human-readable text. Results go to stdout, errors to stderr. Exit codes are
/// stable: 0 success, 1 a runtime/validation error, 2 a usage error.
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

/// Options that apply to every command.
#[derive(Args)]
struct GlobalOpts {
    /// Working directory for local state (default: ./.trakktor).
    #[arg(long, global = true, env = "TRAKKTOR_DIR", value_name = "path")]
    work_dir: Option<PathBuf>,

    /// Print human-readable text instead of the default JSON.
    #[arg(long, global = true)]
    text: bool,

    /// Pretty-print (indent) the JSON output; ignored with --text.
    #[arg(long, global = true)]
    pretty: bool,
}

impl GlobalOpts {
    /// Resolves the working directory: `--work-dir` > `TRAKKTOR_DIR` > default.
    /// clap applies the flag-over-env precedence.
    fn work_dir(&self) -> PathBuf {
        self.work_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(DEFAULT_WORK_DIR))
    }

    /// Whether to emit JSON. JSON is the default; `--text` opts out.
    fn json(&self) -> bool { !self.text }
}

#[derive(Subcommand)]
enum Command {
    /// Transcribe speech from audio (ASR).
    ///
    /// Speech recognition is organized as a set of engines, each with its own
    /// capabilities and flags; pick one as the subcommand. The result is JSON
    /// with the full text, the detected or given language, and timestamped
    /// segments (`--text` prints readable `[start --> end] text` lines).
    Asr {
        #[command(subcommand)]
        command: AsrCommand,
    },

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
pub(crate) enum AsrCommand {
    /// Transcribe audio with a Whisper model.
    ///
    /// The audio file is decoded by the built-in decoder — mp3, aac (LC),
    /// vorbis, flac, alac, and pcm audio in wav/aiff/caf/ogg/mp4/mkv
    /// containers — and transcribed window by window with a fallback policy
    /// that guards against repetition loops. The first use of a model
    /// downloads its checkpoint into the working directory; later runs
    /// reuse it.
    Whisper(WhisperArgs),
}

/// Flags of `asr whisper`.
#[derive(Args)]
pub(crate) struct WhisperArgs {
    /// Path to the audio file to transcribe.
    #[arg(value_name = "audio")]
    pub(crate) audio: PathBuf,

    /// Language of the audio: a code like `en` or `ru`, or an English name
    /// like `russian`. Detected from the first 30 seconds when omitted.
    #[arg(long, value_name = "lang")]
    pub(crate) language: Option<String>,

    /// Timestamp granularity of the output.
    #[arg(
        long,
        value_enum,
        default_value_t = TimestampsArg::Segment,
        value_name = "granularity"
    )]
    pub(crate) timestamps: TimestampsArg,

    /// Model: a published name, downloaded on first use, or a path to a
    /// checkpoint directory. Names: tiny, tiny.en, base, base.en, small,
    /// small.en, medium, medium.en, large-v1, large-v2, large-v3, large,
    /// turbo, large-v3-turbo. Larger models are slower and more accurate.
    #[arg(long, default_value = "tiny", value_name = "name|dir")]
    pub(crate) model: String,

    /// Compute device. `metal` needs a build with the `metal` feature
    /// enabled and is only available on macOS.
    #[arg(
        long,
        value_enum,
        default_value_t = DeviceArg::Cpu,
        value_name = "device"
    )]
    pub(crate) device: DeviceArg,

    /// Compute precision. `f16` (the default) uses about half the memory and
    /// is faster; `f32` runs in full precision for reproducible results, at
    /// twice the memory.
    #[arg(
        long,
        value_enum,
        default_value_t = PrecisionArg::F16,
        value_name = "precision"
    )]
    pub(crate) precision: PrecisionArg,

    /// Transcribe in the source language, or translate into English.
    #[arg(
        long,
        value_enum,
        default_value_t = TaskArg::Transcribe,
        value_name = "task"
    )]
    pub(crate) task: TaskArg,

    /// Sampling temperature the fallback schedule starts at.
    #[arg(long, default_value_t = 0.0, value_name = "float")]
    pub(crate) temperature: f32,

    /// Step between fallback temperatures up to 1.0, or `none` to always use
    /// the single starting temperature.
    #[arg(long, default_value = "0.2", value_name = "float|none")]
    pub(crate) temperature_increment_on_fallback: OrNone<f64>,

    /// Independent sampling trajectories at non-zero temperatures, or
    /// `none`.
    #[arg(long, default_value = "5", value_name = "int|none")]
    pub(crate) best_of: OrNone<usize>,

    /// Beam width at zero temperature, or `none` for greedy decoding.
    #[arg(long, default_value = "5", value_name = "int|none")]
    pub(crate) beam_size: OrNone<usize>,

    /// Beam-search patience (how many finished candidates to collect,
    /// relative to the beam width), or `none` (equivalent to 1.0).
    #[arg(long, default_value = "none", value_name = "float|none")]
    pub(crate) patience: OrNone<f64>,

    /// Length-penalty alpha in 0..=1, or `none` for plain length
    /// normalization when ranking candidates.
    #[arg(long, default_value = "none", value_name = "float|none")]
    pub(crate) length_penalty: OrNone<f64>,

    /// Comma-separated token ids to suppress during sampling; `-1` expands
    /// to a built-in set of non-speech tokens. An empty value disables
    /// suppression.
    #[arg(
        long,
        default_value = "-1",
        allow_hyphen_values = true,
        value_name = "csv"
    )]
    pub(crate) suppress_tokens: String,

    /// Text prompt for the first window — for example domain vocabulary or
    /// proper nouns the audio is likely to contain.
    #[arg(long, value_name = "text")]
    pub(crate) initial_prompt: Option<String>,

    /// Prepend the initial prompt to every window, not just the first.
    #[arg(long)]
    pub(crate) carry_initial_prompt: bool,

    /// Feed the previous output as context for the next window; `false`
    /// reduces the chance of failure loops at some cost to consistency.
    #[arg(
        long,
        default_value_t = true,
        action = clap::ArgAction::Set,
        value_name = "bool"
    )]
    pub(crate) condition_on_previous_text: bool,

    /// Treat a window as failed (and retry hotter) when its text compresses
    /// better than this ratio — the repetition detector. `none` disables it.
    #[arg(long, default_value = "2.4", value_name = "float|none")]
    pub(crate) compression_ratio_threshold: OrNone<f64>,

    /// Treat a window as failed when its average log-probability falls below
    /// this. `none` disables it.
    #[arg(
        long,
        default_value = "-1.0",
        allow_hyphen_values = true,
        value_name = "float|none"
    )]
    pub(crate) logprob_threshold: OrNone<f64>,

    /// Consider a window silent (and skip it) when the no-speech probability
    /// exceeds this while the confidence stays below the log-probability
    /// threshold. `none` disables it.
    #[arg(long, default_value = "0.6", value_name = "float|none")]
    pub(crate) no_speech_threshold: OrNone<f64>,

    /// Punctuation marks merged with the following word (with
    /// `--timestamps word`).
    #[arg(long, default_value = "\"'“¿([{-", value_name = "chars")]
    pub(crate) prepend_punctuations: String,

    /// Punctuation marks merged with the previous word (with
    /// `--timestamps word`).
    #[arg(
        long,
        default_value = "\"'.。,，!！?？:：”)]}、",
        value_name = "chars"
    )]
    pub(crate) append_punctuations: String,

    /// Comma-separated `start,end,start,end,...` offsets in seconds of the
    /// clips to transcribe; the last end defaults to the end of the audio.
    #[arg(long, default_value = "0", value_name = "csv")]
    pub(crate) clip_timestamps: String,

    /// With `--timestamps word`: skip silent stretches longer than this many
    /// seconds when a probable hallucination is detected, or `none`.
    #[arg(long, default_value = "none", value_name = "float|none")]
    pub(crate) hallucination_silence_threshold: OrNone<f64>,
}

/// The `--timestamps` granularity of `asr` output.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum TimestampsArg {
    /// Text only, without the segment list.
    None,
    /// Segment start/end times (the default).
    Segment,
    /// Segment times plus per-word timings.
    Word,
}

/// The `--device` value of `asr whisper`.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum DeviceArg {
    /// The CPU (the default).
    Cpu,
    /// The GPU via Metal, on macOS.
    Metal,
}

/// The `--precision` value of `asr whisper`.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum PrecisionArg {
    /// Half precision: less memory, faster (the default).
    F16,
    /// Full precision: reproducible, at twice the memory.
    F32,
}

impl PrecisionArg {
    pub(crate) fn to_core(self) -> trakktor_core::asr::whisper::Precision {
        match self {
            PrecisionArg::F16 => trakktor_core::asr::whisper::Precision::F16,
            PrecisionArg::F32 => trakktor_core::asr::whisper::Precision::F32,
        }
    }
}

/// The `--task` value of `asr whisper`.
#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum TaskArg {
    /// Transcribe in the source language.
    Transcribe,
    /// Translate into English.
    Translate,
}

impl TaskArg {
    pub(crate) fn to_core(self) -> trakktor_core::asr::whisper::Task {
        match self {
            TaskArg::Transcribe => {
                trakktor_core::asr::whisper::Task::Transcribe
            },
            TaskArg::Translate => trakktor_core::asr::whisper::Task::Translate,
        }
    }
}

/// A flag value that is either a number or the literal `none`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OrNone<T>(pub(crate) Option<T>);

impl<T: std::str::FromStr> std::str::FromStr for OrNone<T>
where
    T::Err: std::fmt::Display,
{
    type Err = String;

    fn from_str(value: &str) -> Result<Self, String> {
        if value.eq_ignore_ascii_case("none") {
            return Ok(OrNone(None));
        }
        value
            .parse::<T>()
            .map(|parsed| OrNone(Some(parsed)))
            .map_err(|e| format!("expected a value or `none`: {e}"))
    }
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

        /// Display fields to show (comma-separated), or `minimal`/`all`.
        ///
        /// `uid` is always included — it is each publication's primary key,
        /// the id you pass to `feed mark-read`, so it is never
        /// dropped. `--fields` selects only the additional fields:
        /// is_read, title, link, published, updated, summary, content,
        /// authors. Special values: `minimal` (the default, =
        /// title,link) and `all` (every field).
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
    /// installed version. By default the Markdown is wrapped as an object
    /// `{ "content": "…" }`; pass `--text` to print the raw Markdown.
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
            output::emit_error(&err, cli.global.json(), cli.global.pretty);
            1
        },
    }
}

/// Executes the parsed command, printing successful output to stdout. Feature
/// errors are converted to [`CliError`] at the `?` boundary.
fn dispatch(cli: &Cli) -> Result<(), CliError> {
    let global = &cli.global;
    match &cli.command {
        Command::Asr { command } => match command {
            AsrCommand::Whisper(args) => crate::asr::run_whisper(
                args,
                &global.work_dir(),
                global.json(),
                global.pretty,
            ),
        },
        Command::Feed { command } => match command {
            FeedCommand::Discover { page_url } => {
                let feeds = feed::discover(page_url)?;
                output::print_discover(&feeds, global.json(), global.pretty);
                Ok(())
            },
            FeedCommand::Read { url, all, fields } => {
                // Validate --fields before any network I/O (fail fast).
                let selection = feed::parse_fields(fields)?;
                let publications = feed::read(url, *all, &global.work_dir())?;
                output::print_read(
                    &publications,
                    &selection,
                    global.json(),
                    global.pretty,
                );
                Ok(())
            },
            FeedCommand::MarkRead { uids } => {
                let summary = feed::mark_read(uids, &global.work_dir())?;
                output::print_mark_read(&summary, global.json(), global.pretty);
                Ok(())
            },
        },
        Command::Skill { command } => match command {
            SkillCommand::Show { full } => {
                crate::skill::show(*full, global.json(), global.pretty)
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
                crate::skill::install(&opts, global.json(), global.pretty)
            },
        },
    }
}
