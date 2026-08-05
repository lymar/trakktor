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
use trakktor_core::{asr::whisper::WhisperError, feed, skill::Target};

use crate::{error::CliError, output};

/// Default working directory when neither `--work-dir` nor `TRAKKTOR_DIR` is
/// set.
const DEFAULT_WORK_DIR: &str = ".trakktor";

/// Home-relative default for the model directory (`~/.trakktor`), used when
/// neither `--model-dir` nor `TRAKKTOR_MODEL_DIR` is set. It shares the
/// basename with the working directory by design — both hold trakktor's data,
/// split by location (the home directory vs the current project).
const DEFAULT_MODEL_DIR_NAME: &str = ".trakktor";

/// A predictable, automation-friendly CLI toolbox for coding agents (Claude
/// Code, OpenCode, etc.): speech-to-text and text-to-speech, voice-activity
/// audio editing, text recognition on page images, feeds, text structuring,
/// punctuation and Russian stress marking, and more — machine-readable output,
/// stable flags, and meaningful exit codes. Reach for it when a task needs one
/// of these helpers, such as fetching a feed's unread items, transcribing an
/// audio file to timestamped text, reading a text or Markdown file aloud into
/// an audio file — in a preset voice or in one cloned from a sample
/// recording — cutting the silence out of a recording, reading the text off a
/// scan or a screenshot page by page, marking a page up into labelled blocks
/// (title, heading, paragraph, footnote, table, formula, picture), restoring
/// punctuation to a raw transcript, splitting a transcript into readable
/// paragraphs, or marking where the stress falls in Russian text.
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
    /// Working directory for per-project local state such as feed read-state
    /// (default: ./.trakktor). Model weights live under `--model-dir` instead.
    #[arg(long, global = true, env = "TRAKKTOR_DIR", value_name = "path")]
    work_dir: Option<PathBuf>,

    /// Directory for downloaded model weights (default: ~/.trakktor), shared
    /// across projects rather than kept in the working directory. Also set by
    /// TRAKKTOR_MODEL_DIR.
    #[arg(
        long,
        global = true,
        env = "TRAKKTOR_MODEL_DIR",
        value_name = "path"
    )]
    model_dir: Option<PathBuf>,

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

    /// Resolves the model directory: `--model-dir` > `TRAKKTOR_MODEL_DIR` >
    /// `~/.trakktor`. Only ASR uses it, so a missing home directory is an error
    /// (`no_home_dir`) rather than a silent fallback — feed never calls this.
    fn model_dir(&self) -> Result<PathBuf, WhisperError> {
        if let Some(dir) = &self.model_dir {
            return Ok(dir.clone());
        }
        std::env::home_dir()
            .map(|home| home.join(DEFAULT_MODEL_DIR_NAME))
            .ok_or(WhisperError::HomeDirUnknown)
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

    /// Synthesize speech from text (TTS).
    ///
    /// Speech synthesis is organized as a set of engines, each with its own
    /// voices and flags; pick one as the subcommand. Text of any length is
    /// read aloud into a single audio file — plain text or Markdown, from an
    /// argument, a file, or standard input. The format follows the output's
    /// extension (wav and flac directly, others through ffmpeg), and the
    /// result on stdout is JSON with the path, the duration, and the voice
    /// used.
    Tts {
        #[command(subcommand)]
        command: TtsCommand,
    },

    /// Edit audio by voice-activity detection: report, cut, or split on speech.
    ///
    /// Silero voice-activity detection finds the speech in an audio file, and a
    /// subcommand acts on it: `timeline` reports the speech spans as JSON,
    /// `cut` writes a new file with non-speech removed (or kept, with `--keep
    /// non-speech`), and `split` writes one file per detected span. Cutting
    /// works on the decoded original at full quality — its own sample rate,
    /// channels, and bit depth, not the 16 kHz mono the detector uses — so
    /// nothing is downsampled. Output is WAV by default, or FLAC with `--format
    /// flac`. Detection is tunable directly or through `--preset`.
    Vad {
        #[command(subcommand)]
        command: VadCommand,
    },

    /// Structure and transform text.
    ///
    /// A group of text-processing operations, each with a local model, fully
    /// offline: `structify` turns an unstructured wall of text — for example a
    /// speech transcript whose line breaks fall on segments rather than meaning
    /// — into readable paragraphs; `punctuate` restores punctuation and
    /// capitalization in raw lowercase text, such as the output of the Vosk and
    /// GigaAM speech engines; `stress` marks the stressed vowel of every
    /// Russian word (and restores the letter ё), which is what a speech
    /// synthesizer needs to read the text correctly.
    Text {
        #[command(subcommand)]
        command: TextCommand,
    },

    /// Read the text off page images (OCR).
    ///
    /// Text recognition is organized as a set of engines, each with its own
    /// models and languages; pick one as the subcommand. The input is one or
    /// more page images and the result is one page per image, in the order
    /// given, so a scanned document is read by passing its pages in sequence.
    /// Each page carries its text lines with their quadrangles and confidence;
    /// `--format md` assembles them into Markdown instead, with paragraphs and
    /// a reading order worked out from the geometry.
    ///
    /// Which engine: `paddle` for a page in one writing system it covers —
    /// 13 MB and a second or two per page, and the right default. `vl` for a
    /// page whose script it does not cover, for one that mixes scripts, or
    /// when a table or a formula is wanted as structure rather than as lines —
    /// about 1.9 GB downloaded once, and tens of seconds per page.
    ///
    /// Both engines also run a layout model, which labels the blocks of the
    /// page — document title, section heading, paragraph, abstract, footnote,
    /// running head, page number, table, formula, picture, caption — so that
    /// the structure is read rather than guessed. It costs 129 MB, downloaded
    /// once, and about a second a page; `--no-layout` skips it when only the
    /// lines are wanted. `ocr layout` runs the same model on its own and
    /// answers "what is on this page" without reading a word of it.
    Ocr {
        #[command(subcommand)]
        command: OcrCommand,
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
pub(crate) enum OcrCommand {
    /// Read pages with the PaddleOCR PP-OCRv5 pipeline.
    ///
    /// A detector finds the text lines on the page, each line is straightened
    /// out of it, and a recognizer reads the line. Twelve recognizers cover
    /// the scripts between them and `--lang` picks one; the models download on
    /// first use into the model directory (~/.trakktor by default) and later
    /// runs reuse them. The full working set for a language is about 13 MB.
    Paddle(Box<OcrPaddleArgs>),

    /// Read pages with the PaddleOCR-VL document model.
    ///
    /// A generative model that writes out the text it sees instead of picking
    /// characters from a dictionary. It works out the writing system by
    /// itself — so there is no `--lang` here — reads scripts the classic
    /// pipeline has no model for at all, and can return a table as markup or a
    /// formula as LaTeX. It finds the lines with the same detector `ocr paddle`
    /// uses, groups them into blocks, and reads a block at a time: handed a
    /// whole page it tends to get stuck repeating itself, and handed a single
    /// line it has too little context to settle on a script.
    ///
    /// The price is the model: about 1.9 GB downloaded once, and tens of
    /// seconds a page. Prefer `ocr paddle` unless you need what this buys.
    Vl(Box<OcrVlArgs>),

    /// Mark up the structure of a page without reading it.
    ///
    /// A layout model returns the blocks a page is made of, each with a label:
    /// document title, section heading, paragraph, abstract, footnote, running
    /// head, page number, table, formula, picture, caption, stamp, chart. It
    /// reads no text — this is the answer to "what is on this page", not "what
    /// does it say".
    ///
    /// The same model runs inside `ocr paddle` and `ocr vl` unless
    /// `--no-layout` turns it off, where the labels drive the Markdown and,
    /// for `ocr vl`, the blocks the model is asked to read. It downloads once,
    /// about 129 MB, and takes about a second a page.
    Layout(Box<OcrLayoutArgs>),
}

#[derive(Args)]
pub(crate) struct OcrPaddleArgs {
    /// Page images to read, in reading order: each file is one page of the
    /// result. PNG, JPEG, TIFF, WebP, BMP and GIF are understood.
    #[arg(value_name = "page")]
    pub(crate) pages: Vec<PathBuf>,

    /// Language of the page, which chooses the recognizer. Pass `list` to
    /// print every code. A recognizer is trained on one script and can only
    /// emit characters from its own alphabet, so a page in a script other than
    /// the one selected comes back empty or as nonsense even though its lines
    /// were found.
    #[arg(long, default_value = "ru", value_name = "code")]
    pub(crate) lang: String,

    /// Form of the text result: `lines` prints the recognized lines as they
    /// were found, `md` assembles Markdown — paragraphs, reading order,
    /// headings and footnotes worked out from the geometry of the page.
    #[arg(
        long,
        value_enum,
        default_value_t = OcrFormatArg::Lines,
        value_name = "form"
    )]
    pub(crate) format: OcrFormatArg,

    /// Write the text result to this file as well as reporting it. With
    /// `--format md` any illustrations found on the page are written next to
    /// it, in an `imgs` directory, and linked from the Markdown.
    #[arg(long, value_name = "path")]
    pub(crate) out: Option<PathBuf>,

    /// Write every recognized line as its own image into this directory — the
    /// straightened crop the recognizer actually read. Useful for telling a
    /// detection mistake from a recognition one.
    #[arg(long, value_name = "dir")]
    pub(crate) crops: Option<PathBuf>,

    /// Longest side, in pixels, the page is scaled to before detection. This
    /// is the setting that decides whether small type is found at all: an A4
    /// page scanned at 300 dpi is 3508 pixels tall, so the default shrinks it
    /// nearly fourfold and footnote-sized text stops being detected. Raising
    /// it to 1536 or 2048 finds those lines, and costs time in proportion to
    /// the area.
    #[arg(long, default_value_t = 960, value_name = "px")]
    pub(crate) limit_side_len: usize,

    /// Text detection model. Defaults to the small one, which is what makes a
    /// page take seconds rather than a minute.
    #[arg(long, value_name = "name|dir")]
    pub(crate) det_model: Option<String>,

    /// Text recognition model, overriding the one `--lang` would choose.
    #[arg(long, value_name = "name|dir")]
    pub(crate) rec_model: Option<String>,

    /// Probability above which a pixel counts as text when the detector's map
    /// is thresholded.
    #[arg(long, default_value_t = 0.3, value_name = "p")]
    pub(crate) thresh: f32,

    /// Mean probability a detected box must reach to be kept. Lower it to
    /// recover faint lines, at the price of boxes over background.
    #[arg(long, default_value_t = 0.6, value_name = "p")]
    pub(crate) box_thresh: f32,

    /// How far a detected box is expanded before the line is cut out. The
    /// detector marks a shrunken core of each line, so some expansion is
    /// always needed; more of it captures tall letters and accents, and
    /// eventually the neighbouring line.
    #[arg(long, default_value_t = 1.5, value_name = "ratio")]
    pub(crate) unclip_ratio: f32,

    /// Lines the recognizer read with less confidence than this are dropped.
    /// Zero keeps everything, which is what to use when a line is missing and
    /// you want to know whether it was found at all.
    #[arg(long, default_value_t = 0.5, value_name = "p")]
    pub(crate) drop_score: f32,

    /// Also run the text-line orientation classifier, which turns a line
    /// around when it was set upside down. Costs a little per line and rarely
    /// fires on a scanned book.
    #[arg(long)]
    pub(crate) textline_orientation: bool,

    /// Skip the layout model, which otherwise labels the blocks of the page —
    /// heading, paragraph, footnote, running head, page number, table,
    /// formula, picture. Without those labels the structure is guessed from
    /// the geometry, and the guess fails where it matters most: a title set in
    /// capitals makes *shorter* boxes than the text below it, so it is not
    /// found by size at all.
    ///
    /// Use this when only the lines are wanted and their structure is not, or
    /// to avoid the 129 MB model. It saves about a second a page.
    #[arg(long)]
    pub(crate) no_layout: bool,

    /// Layout model to run.
    #[arg(long, value_name = "name|dir")]
    pub(crate) layout_model: Option<String>,

    /// Inference runtime executing the models. This engine serves `candle`
    /// only; of the OCR engines, `ocr vl` is the one with a burn runtime.
    #[arg(
        long,
        value_enum,
        default_value_t = RuntimeArg::Candle,
        value_name = "engine"
    )]
    pub(crate) runtime: RuntimeArg,

    /// Device to run on.
    #[arg(
        long,
        value_enum,
        default_value_t = DeviceArg::Cpu,
        value_name = "device"
    )]
    pub(crate) device: DeviceArg,
}

#[derive(Args)]
pub(crate) struct OcrVlArgs {
    /// Page images to read, in reading order: each file is one page of the
    /// result. PNG, JPEG, TIFF, WebP, BMP and GIF are understood.
    #[arg(value_name = "page")]
    pub(crate) pages: Vec<PathBuf>,

    /// What to ask the model for. `ocr` reads the text and is the only one
    /// that works block by block; the other three are asked of the page as a
    /// whole, because a table or a formula is itself one block and cutting it
    /// up destroys the structure that makes the question worth asking.
    #[arg(
        long,
        value_enum,
        default_value_t = OcrTaskArg::Ocr,
        value_name = "task"
    )]
    pub(crate) task: OcrTaskArg,

    /// Form of the text result: `lines` prints the recognized lines as they
    /// were read, `md` assembles Markdown — paragraphs, reading order,
    /// headings and footnotes worked out from the geometry of the page.
    #[arg(
        long,
        value_enum,
        default_value_t = OcrFormatArg::Lines,
        value_name = "form"
    )]
    pub(crate) format: OcrFormatArg,

    /// Write the text result to this file as well as reporting it. With
    /// `--format md` any illustrations found on the page are written next to
    /// it, in an `imgs` directory, and linked from the Markdown.
    #[arg(long, value_name = "path")]
    pub(crate) out: Option<PathBuf>,

    /// Write every block as its own image into this directory — exactly the
    /// picture the model was handed. The unit here is the block, not the line:
    /// when a reading goes wrong, this is what shows whether the model was
    /// given something coherent to read.
    #[arg(long, value_name = "dir")]
    pub(crate) crops: Option<PathBuf>,

    /// Longest the answer for one block may get. It has to be generous, and
    /// how generous depends on the script: Tibetan costs about five times more
    /// tokens per character than English, so a ceiling that fits a Latin page
    /// will cut a Tibetan one in half.
    #[arg(long, default_value_t = 1024, value_name = "n")]
    pub(crate) max_tokens: usize,

    /// Blocks the model read with less mean token probability than this are
    /// dropped. Zero keeps everything, which is what to use when text is
    /// missing and you want to know whether it was read at all.
    #[arg(long, default_value_t = 0.3, value_name = "p")]
    pub(crate) drop_score: f32,

    /// Read each page in a single call instead of block by block. Right when
    /// the page already is one block; a way into a repetition loop when it is
    /// not.
    #[arg(long)]
    pub(crate) whole_page: bool,

    /// Longest side, in pixels, the page is scaled to before the lines are
    /// detected. A line the detector misses is not merely unreported here — it
    /// is a block the model never sees.
    #[arg(long, default_value_t = 1440, value_name = "px")]
    pub(crate) limit_side_len: usize,

    /// The checkpoint to read with, or a directory holding one.
    #[arg(long, value_name = "name|dir")]
    pub(crate) model: Option<String>,

    /// Text detection model, as in `ocr paddle`.
    #[arg(long, value_name = "name|dir")]
    pub(crate) det_model: Option<String>,

    /// Probability above which a pixel counts as text when the detector's map
    /// is thresholded.
    #[arg(long, default_value_t = 0.3, value_name = "p")]
    pub(crate) thresh: f32,

    /// Mean probability a detected box must reach to be kept.
    #[arg(long, default_value_t = 0.6, value_name = "p")]
    pub(crate) box_thresh: f32,

    /// How far a detected box is expanded before the block is cut out.
    #[arg(long, default_value_t = 1.5, value_name = "ratio")]
    pub(crate) unclip_ratio: f32,

    /// Widest horizontal gap, in line heights, that still joins two detected
    /// boxes into one line of text. The detector returns a line with wide word
    /// spaces as several boxes, and a line handed to the model in halves is
    /// read far worse than a whole one — but the same measurement is what
    /// tells a word space from a column gutter, so raising this merges
    /// columns.
    #[arg(long, default_value_t = 1.2, value_name = "ratio")]
    pub(crate) block_row_gap: f32,

    /// Vertical gap, in line heights, that ends a block.
    #[arg(long, default_value_t = 0.8, value_name = "ratio")]
    pub(crate) block_gap: f32,

    /// How much of the narrower box two lines must share horizontally to
    /// belong to one block. This is what keeps two columns apart.
    #[arg(long, default_value_t = 0.3, value_name = "ratio")]
    pub(crate) block_overlap: f32,

    /// How far a line's height may differ from its block's, either way. This
    /// is the setting that decides how a page mixing two writing systems is
    /// read: a line of a stacking script runs taller than an alphabetic line
    /// of the same size, and the ratio is the only sign the geometry has that
    /// the two are different kinds of text. Raise it and they join into one
    /// block, which reads markedly worse.
    #[arg(long, default_value_t = 1.5, value_name = "ratio")]
    pub(crate) block_height: f32,

    /// Most lines one block may hold. The answer grows with the block, and so
    /// does the chance of the model losing its place in it.
    #[arg(long, default_value_t = 12, value_name = "n")]
    pub(crate) block_lines: usize,

    /// How far, in line heights, the cut is grown past the detected boxes.
    /// The detector marks a shrunken core of each line; without a margin the
    /// tall letters go, and in a stacking script whole tiers of marks go with
    /// them.
    #[arg(long, default_value_t = 0.35, value_name = "ratio")]
    pub(crate) block_padding: f32,

    /// Skip the layout model, and go back to building the blocks out of line
    /// geometry alone. The labels are what lets a block be cut along the
    /// structure of the page rather than across it: with them a table goes to
    /// the model whole, as a table, and a formula goes as a formula; without
    /// them a table comes back as strips with its columns doubled.
    ///
    /// It saves a 129 MB model and about a second a page — next to nothing
    /// beside this engine's own price.
    #[arg(long)]
    pub(crate) no_layout: bool,

    /// Layout model to run.
    #[arg(long, value_name = "name|dir")]
    pub(crate) layout_model: Option<String>,

    /// Inference runtime executing the model. Both read the pages the same,
    /// at the same precision (f16 on Metal, f32 on the CPU); `burn` needs a
    /// build with the `burn` feature enabled. The detection stage runs on
    /// candle either way.
    #[arg(
        long,
        value_enum,
        default_value_t = RuntimeArg::Candle,
        value_name = "engine"
    )]
    pub(crate) runtime: RuntimeArg,

    /// Device to run on. This model is nearly two gigabytes of weights and an
    /// autoregressive decode loop, so a GPU is the difference between a page
    /// in seconds and a page in minutes — hence the default, where the build
    /// has one.
    #[arg(
        long,
        value_enum,
        default_value_t = DEFAULT_VL_DEVICE,
        value_name = "device"
    )]
    pub(crate) device: DeviceArg,
}

#[derive(Args)]
pub(crate) struct OcrLayoutArgs {
    /// Page images to mark up, in reading order: each file is one page of the
    /// result. PNG, JPEG, TIFF, WebP, BMP and GIF are understood.
    #[arg(value_name = "page")]
    pub(crate) pages: Vec<PathBuf>,

    /// Layout model to run.
    #[arg(long, value_name = "name|dir")]
    pub(crate) model: Option<String>,

    /// One score floor for every block kind, replacing the per-kind defaults.
    ///
    /// The defaults are not a formality: the model is as sure of a paragraph
    /// as the writing system is familiar — around 0.98 for English, around
    /// 0.45 for Tibetan — so a flat half hides most of a page in an unfamiliar
    /// script. Lower this to see what the model nearly said; raise it to keep
    /// only what it is certain of.
    #[arg(long, value_name = "p")]
    pub(crate) threshold: Option<f32>,

    /// Write every block as its own image into this directory — the crop the
    /// label describes.
    #[arg(long, value_name = "dir")]
    pub(crate) crops: Option<PathBuf>,

    /// Inference runtime executing the model. Both mark the pages up the same,
    /// in f32 on either device; `burn` needs a build with the `burn` feature.
    #[arg(
        long,
        value_enum,
        default_value_t = RuntimeArg::Candle,
        value_name = "engine"
    )]
    pub(crate) runtime: RuntimeArg,

    /// Device to run on.
    #[arg(
        long,
        value_enum,
        default_value_t = DeviceArg::Cpu,
        value_name = "device"
    )]
    pub(crate) device: DeviceArg,
}

/// The device `ocr vl` runs on unless told otherwise.
#[cfg(feature = "metal")]
const DEFAULT_VL_DEVICE: DeviceArg = DeviceArg::Metal;
#[cfg(not(feature = "metal"))]
const DEFAULT_VL_DEVICE: DeviceArg = DeviceArg::Cpu;

/// What `ocr vl` asks the model for.
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
pub(crate) enum OcrTaskArg {
    /// Read the text.
    Ocr,
    /// Read a table, as markup rather than as lines.
    Table,
    /// Read a formula as LaTeX.
    Formula,
    /// Describe a chart.
    Chart,
}

/// Form of the text an OCR run reports.
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
pub(crate) enum OcrFormatArg {
    /// The recognized lines, one per line of the page.
    Lines,
    /// Markdown, with paragraphs and a reading order.
    Md,
}

#[derive(Subcommand)]
pub(crate) enum AsrCommand {
    /// Transcribe audio with a Whisper model.
    ///
    /// The audio file is decoded by the built-in decoder — mp3, aac (LC),
    /// vorbis, flac, alac, and pcm audio in wav/aiff/caf/ogg/mp4/mkv
    /// containers — and transcribed window by window with a fallback policy
    /// that guards against repetition loops. The first use of a model
    /// downloads its checkpoint into the model directory (~/.trakktor by
    /// default; see --model-dir), and later runs reuse it.
    Whisper(WhisperArgs),

    /// Transcribe audio with a GigaAM model.
    ///
    /// GigaAM is a family of Conformer acoustic models with CTC or RNN-T
    /// (transducer) decoding — mainly for Russian and, with the
    /// multilingual checkpoints, several more languages. The default model
    /// emits punctuated, capitalized Russian; see --model for the
    /// alternatives. The audio is decoded by the built-in decoder, split
    /// along detected speech into chunks, and each chunk transcribed in a
    /// single pass. The first use of a model downloads its checkpoint into
    /// the model directory (~/.trakktor by default; see --model-dir), and
    /// later runs reuse it.
    Gigaam(GigaamArgs),

    /// Transcribe audio with a Vosk model.
    ///
    /// Vosk's current model line is a family of Zipformer2 transducers
    /// (Alpha Cephei / k2-fsa) — mainly for Russian, with more languages
    /// available. The models emit lowercase text without punctuation.
    /// Offline models are split along detected speech and each chunk
    /// transcribed in one pass; streaming models run a low-latency chunked
    /// pipeline. The default model is the large Russian one; see --model for
    /// the alternatives, or pass a directory with a compatible export. The
    /// first use of a named model downloads it into the model directory
    /// (~/.trakktor by default; see --model-dir), and later runs reuse it.
    Vosk(VoskArgs),
}

#[derive(Subcommand)]
pub(crate) enum TtsCommand {
    /// Synthesize speech with a Qwen3-TTS model.
    ///
    /// Reads the text as an argument, from a file with --text-file, or from
    /// standard input (--text-file -), and writes spoken audio at 24 kHz. Text
    /// of any length works: it is split into paragraphs (plain text or
    /// Markdown, see --text-format), spoken paragraph by paragraph, and joined
    /// into one file with a configurable pause between them (--pause-ms). A
    /// paragraph too long for one utterance is split further with a local
    /// sentence model, downloaded on first need. The voice is one of the
    /// model's preset speakers (see --voice) and the language is set
    /// independently of it (see --language), so any voice can speak any of the
    /// supported languages, Russian included. Generation samples by default,
    /// which makes each run differ slightly; pass --seed to repeat a run
    /// exactly, or --greedy for deterministic output. The first use of a model
    /// downloads its checkpoint into the model directory (~/.trakktor by
    /// default; see --model-dir), and later runs reuse it.
    #[command(name = "qwen3-tts")]
    Qwen3Tts(Qwen3TtsArgs),

    /// Synthesize Russian speech in a cloned voice with an ESpeech model.
    ///
    /// The voice is not chosen from a list: it is taken from a recording you
    /// pass with --ref-audio, together with a transcript of what is said in it
    /// (--ref-text). A few seconds of clean speech are enough, and more than
    /// twelve are not used. Reads the text as an argument, from a file with
    /// --text-file, or from standard input (--text-file -), and writes spoken
    /// audio at 24 kHz. Text of any length works: it is split into paragraphs
    /// (plain text or Markdown, see --text-format), and a paragraph too long
    /// for one utterance is split further — with a local sentence model,
    /// downloaded on first need, and at punctuation where that is not enough.
    /// Russian stress is written with `+` before the stressed vowel (`з+амок`
    /// is a lock, `зам+ок` a castle), and the model reads it as a real input.
    /// By default the engine marks the text — and the reference transcript —
    /// itself before speaking (see --stress); a `+` you wrote yourself is
    /// never moved, so marking the odd word by hand and leaving the rest to
    /// the model is the normal way to work. The reading is repeatable:
    /// --seed fixes it, and the same seed gives the same file. The first use
    /// of a model downloads its checkpoint into the model directory
    /// (~/.trakktor by default; see --model-dir), and later runs reuse it.
    Espeech(EspeechArgs),

    /// Synthesize speech with a Silero model, in a preset voice.
    ///
    /// Sixty voices across twenty languages in the default model: Russian (29
    /// of them), Bashkir, Belarusian, Ukrainian, Kazakh, Tatar, Tajik, Khakas,
    /// Kalmyk, Armenian, Azerbaijani, Chuvash, Erzya, Georgian, Kabardian,
    /// Kyrgyz, Moksha, Udmurt, Uzbek and Yakut. The language is not a flag but
    /// the voice: each speaker was trained for one, and the prefix of its name
    /// is that language's code (see --voice, and `--voice list` to print
    /// them). Georgian, Armenian and Latin-written Azerbaijani or Uzbek are
    /// transliterated into the model's own alphabet first. **There is no
    /// Latin-script language and so no English**: the alphabet is Cyrillic,
    /// Latin letters are dropped rather than read, and a text with nothing
    /// else in it is an error — use `tts qwen3-tts` for those. Reads the text
    /// as an argument, from a file with --text-file, or from standard input
    /// (--text-file -), and
    /// writes spoken audio at 48 kHz — the model's own rate, with 24 kHz and 8
    /// kHz derived from it by its own filterbank (--sample-rate). Text of any
    /// length works: it is split into paragraphs (plain text or Markdown, see
    /// --text-format), spoken piece by piece and joined into one file. Nothing
    /// in this model is sampled, so the same command always produces the same
    /// file; --rate and --pitch change the reading. Russian stress is written
    /// with `+` before the stressed vowel (`з+амок` is a lock, `зам+ок` a
    /// castle) and is a real input to the model, not a hint: by default the
    /// engine marks the text itself before speaking when the voice is a
    /// Russian one (see --stress), and a `+` you wrote yourself is never
    /// moved. The default model is
    /// published under MIT; two others are not, and need
    /// --allow-non-commercial-models (see --model). The first use of a model
    /// downloads it into the model directory (~/.trakktor by default; see
    /// --model-dir), converts it once, and later runs reuse it.
    Silero(SileroArgs),
}

/// Flags of `tts silero`.
#[derive(Args)]
// At most one source of text. Unlike the other engines the group is not
// required: `--voice list` asks about the model rather than speaks, and
// demanding a text for it would be a riddle. Anything else without a text is
// still a usage error, raised in clap's own style.
#[command(group(clap::ArgGroup::new("silero_source").args(["speech", "text_file"])))]
pub(crate) struct SileroArgs {
    /// The text to speak. Omit it when reading the text from a file with
    /// --text-file.
    // The id must differ from the global `--text` flag, which clap would
    // otherwise take this for.
    #[arg(id = "speech", value_name = "text")]
    pub(crate) text: Option<String>,

    /// Read the text to speak from a UTF-8 file instead of the argument, or
    /// from standard input with `-`. Text of any length works: it is spoken
    /// piece by piece and joined into one file (see --text-format and
    /// --pause-ms).
    #[arg(long, value_name = "path|-")]
    pub(crate) text_file: Option<PathBuf>,

    /// How to read the input: where paragraphs end and whether it carries
    /// markup. `txt` takes one paragraph per line; `md` separates paragraphs
    /// with blank lines and strips Markdown markup (headings, list and quote
    /// markers, emphasis, links — the text of a table or a code block is
    /// kept). `auto` decides by the file extension, then by the text itself,
    /// and falls back to `md`.
    #[arg(
        long,
        value_enum,
        default_value_t = TextFormatArg::Auto,
        value_name = "format"
    )]
    pub(crate) text_format: TextFormatArg,

    /// Voice to read in, as the model names its speakers — and with it the
    /// language, since each speaker was trained for exactly one and the prefix
    /// of the name says which (`ru_` Russian, `ukr_` Ukrainian, `kaz_` Kazakh,
    /// `kat_` Georgian, and so on). Pass `list` to print the voices of the
    /// selected model and exit.
    #[arg(long, default_value = "ru_zhadyra", value_name = "name|list")]
    pub(crate) voice: String,

    /// Model: a published name, downloaded on first use, or a path to a
    /// converted model directory. `cis-base` (the default) and
    /// `cis-base-nostress` are published under MIT; `cis-ext` and `ru-classic`
    /// — the five long-standing Russian voices — are under CC BY-NC-SA 4.0 and
    /// need --allow-non-commercial-models. The license of the model that was
    /// used is reported in the output.
    #[arg(long, default_value = "cis-base", value_name = "name|dir")]
    pub(crate) model: String,

    /// Allow the models published under CC BY-NC-SA 4.0 (non-commercial,
    /// share-alike). Without it, asking for one is an error before anything is
    /// downloaded. The flag grants no license — it records that the choice was
    /// deliberate, and makes it reproducible in a script.
    #[arg(long)]
    pub(crate) allow_non_commercial_models: bool,

    /// Sample rate of the output. 48000 is what the model synthesizes; the
    /// lower two are its own filterbank applied to that, not a resampling, so
    /// they are what the model itself would produce.
    #[arg(long, default_value_t = 48_000, value_name = "hz")]
    pub(crate) sample_rate: u32,

    /// Speech rate as a multiplier: below 1 speaks slower, above 1 faster. It
    /// divides the length the model gave each symbol, so it changes the pacing
    /// of the reading rather than replaying it at a different speed.
    #[arg(long, default_value_t = 1.0, value_name = "float")]
    pub(crate) rate: f32,

    /// Pitch as a multiplier: below 1 lowers the voice, above 1 raises it. The
    /// shift is scaled by the speaker's own range, so a voice moved this way
    /// still sounds like itself; useful values sit between 0.75 and 1.25.
    #[arg(long, default_value_t = 1.0, value_name = "float")]
    pub(crate) pitch: f32,

    /// Whether to mark the stress in the text before speaking it. This model
    /// reads `+` before a stressed vowel as a real input — it is a symbol of
    /// its alphabet — and the published models do not place it themselves, so
    /// `auto` (the default) marks the text using the same model `text stress`
    /// uses, downloaded on first use. Marks you wrote yourself are never
    /// moved. The marker is Russian, so with a voice for another language
    /// (any name prefix but `ru_`) the text is spoken as written, whatever
    /// this flag says. `off` speaks the text exactly as given.
    #[arg(
        long,
        value_enum,
        default_value_t = StressArg::Auto,
        value_name = "mode"
    )]
    pub(crate) stress: StressArg,

    /// Silence inserted between paragraphs, in milliseconds. Each paragraph is
    /// first trimmed of the silence the model leaves at its edges (a few tens
    /// of milliseconds are kept as breathing room), so the gap is this value
    /// rather than whatever the model happened to add. Pieces of a single
    /// split paragraph get half of it.
    #[arg(long, default_value_t = 500, value_name = "ms")]
    pub(crate) pause_ms: u32,

    /// What to do with the loudness of each piece. `match` (the default)
    /// brings the pieces to a common level before joining, `keep` leaves them
    /// exactly as synthesized.
    #[arg(
        long,
        value_enum,
        default_value_t = LevelsArg::Match,
        value_name = "what"
    )]
    pub(crate) levels: LevelsArg,

    /// Where to write the audio; the extension picks the format. `.wav` and
    /// `.flac` are written directly, anything ffmpeg knows (mp3, m4a, opus,
    /// ogg, …) through it — see --audio-encoder.
    #[arg(long, short, default_value = "speech.wav", value_name = "path")]
    pub(crate) output: PathBuf,

    /// Encoder for the output file. `auto` (the default) writes wav and flac
    /// with the built-in pure-Rust encoder and hands any other extension to an
    /// installed `ffmpeg`; `builtin` refuses anything but wav/flac; `ffmpeg`
    /// always shells out.
    #[arg(
        long,
        value_enum,
        default_value_t = AudioEncoderArg::Auto,
        value_name = "encoder"
    )]
    pub(crate) audio_encoder: AudioEncoderArg,

    /// Target bitrate for lossy ffmpeg formats, such as `192k` or `320k` (sets
    /// ffmpeg's `-b:a`). Ignored by the built-in encoder; omit for ffmpeg's
    /// own default.
    #[arg(long, value_name = "rate")]
    pub(crate) bitrate: Option<String>,

    /// Inference runtime executing the model.
    #[arg(
        long,
        value_enum,
        default_value_t = RuntimeArg::Candle,
        value_name = "runtime"
    )]
    pub(crate) runtime: RuntimeArg,

    /// Compute device. This model is small enough that the CPU is the intended
    /// place to run it rather than a fallback; `metal` needs a build with the
    /// `metal` feature enabled, and is only available on macOS.
    #[arg(
        long,
        value_enum,
        default_value_t = DeviceArg::Cpu,
        value_name = "device"
    )]
    pub(crate) device: DeviceArg,
}

/// Flags of `tts espeech`.
#[derive(Args)]
// Exactly one source of text, and the error names both when neither is given.
#[command(group(clap::ArgGroup::new("espeech_source").required(true).args(["speech", "text_file"])))]
#[command(group(clap::ArgGroup::new("espeech_ref_text").required(true).args(["ref_text", "ref_text_file"])))]
pub(crate) struct EspeechArgs {
    /// The text to speak. Omit it when reading the text from a file with
    /// --text-file.
    // The id must differ from the global `--text` flag, which clap would
    // otherwise take this for.
    #[arg(id = "speech", value_name = "text")]
    pub(crate) text: Option<String>,

    /// Recording of the voice to speak in. Any audio file the built-in decoder
    /// reads; a few seconds of clean, uninterrupted speech work best. Silence
    /// at the edges is trimmed, and anything past twelve seconds is left
    /// unused — the model conditions on no more than that.
    #[arg(long, value_name = "path")]
    pub(crate) ref_audio: PathBuf,

    /// What is said in --ref-audio, word for word. The model aligns the
    /// recording against this text, so a wrong transcript costs quality; mark
    /// stress in it with `+` as you would in the text to speak.
    #[arg(long, value_name = "text")]
    pub(crate) ref_text: Option<String>,

    /// Read the reference transcript from a UTF-8 file instead of --ref-text.
    #[arg(long, value_name = "path")]
    pub(crate) ref_text_file: Option<PathBuf>,

    /// Read the text to speak from a UTF-8 file instead of the argument, or
    /// from standard input with `-`. Text of any length works: it is spoken
    /// piece by piece and joined into one file (see --text-format and
    /// --pause-ms).
    #[arg(long, value_name = "path|-")]
    pub(crate) text_file: Option<PathBuf>,

    /// How to read the input: where paragraphs end and whether it carries
    /// markup. `txt` takes one paragraph per line; `md` separates paragraphs
    /// with blank lines and strips Markdown markup (headings, list and quote
    /// markers, emphasis, links — the text of a table or a code block is
    /// kept). `auto` decides by the file extension, then by the text itself,
    /// and falls back to `md`.
    #[arg(
        long,
        value_enum,
        default_value_t = TextFormatArg::Auto,
        value_name = "format"
    )]
    pub(crate) text_format: TextFormatArg,

    /// Silence inserted between paragraphs, in milliseconds. Each paragraph is
    /// first trimmed of the silence the model leaves at its edges (a few tens
    /// of milliseconds are kept as breathing room), so the gap is this value
    /// rather than whatever the model happened to add. Pieces of a single
    /// split paragraph get half of it.
    #[arg(long, default_value_t = 500, value_name = "ms")]
    pub(crate) pause_ms: u32,

    /// What to do with the loudness of each piece. Every piece is spoken as
    /// its own utterance, so a long text can drift from one to the next;
    /// `match` (the default) brings the pieces to a common level before
    /// joining, `keep` leaves them exactly as synthesized.
    #[arg(
        long,
        value_enum,
        default_value_t = LevelsArg::Match,
        value_name = "what"
    )]
    pub(crate) levels: LevelsArg,

    /// Where to write the audio; the extension picks the format. `.wav` and
    /// `.flac` are written directly, anything ffmpeg knows (mp3, m4a, opus,
    /// ogg, …) through it — see --audio-encoder.
    #[arg(long, short, default_value = "speech.wav", value_name = "path")]
    pub(crate) output: PathBuf,

    /// Encoder for the output file. `auto` (the default) writes wav and flac
    /// with the built-in pure-Rust encoder and hands any other extension to an
    /// installed `ffmpeg`; `builtin` refuses anything but wav/flac; `ffmpeg`
    /// always shells out.
    #[arg(
        long,
        value_enum,
        default_value_t = AudioEncoderArg::Auto,
        value_name = "encoder"
    )]
    pub(crate) audio_encoder: AudioEncoderArg,

    /// Target bitrate for lossy ffmpeg formats, such as `192k` or `320k` (sets
    /// ffmpeg's `-b:a`). Ignored by the built-in encoder; omit for ffmpeg's
    /// own default.
    #[arg(long, value_name = "rate")]
    pub(crate) bitrate: Option<String>,

    /// Language to speak in. These checkpoints are Russian only, so the value
    /// is accepted for symmetry with the other engines and rejected unless it
    /// spells Russian (`russian`, `ru`, `rus`) or is `auto`.
    #[arg(long, default_value = "auto", value_name = "lang")]
    pub(crate) language: String,

    /// Model: a published name, downloaded on first use, or a path to a
    /// checkpoint directory. Names: rl-v2 (the default), rl-v1, sft-256k,
    /// sft-95k, podcaster (trained on podcast delivery). All are the same size
    /// and differ in training, so the choice is one of manner, not of quality
    /// against speed.
    #[arg(long, default_value = "rl-v2", value_name = "name|dir")]
    pub(crate) model: String,

    /// Steps the solver takes per piece, and the time scales with it exactly:
    /// measured 33, 62 and 91 seconds for 16, 32 and 48 steps. Intelligibility
    /// does not scale with it — all three transcribe back identically — so 16
    /// halves the time honestly, and past 48 there is nothing measurable to
    /// gain.
    #[arg(long, default_value_t = 32, value_name = "int")]
    pub(crate) nfe_step: usize,

    /// How strongly the reading is pushed toward the text and the reference
    /// voice. Zero switches off the second, unguided pass — exactly twice as
    /// fast, and not worth it for speech: measured on the test phrase, words
    /// come out mangled and half-swallowed. Lower it only if you know why.
    #[arg(long, default_value_t = 2.0, value_name = "float")]
    pub(crate) cfg_strength: f32,

    /// Speech rate as a multiplier: below 1 speaks slower, above 1 faster. It
    /// works by deciding how much time the words are given, so extreme values
    /// crowd or stretch the reading rather than only changing its pace.
    #[arg(long, default_value_t = 1.0, value_name = "float")]
    pub(crate) speed: f32,

    /// Seed of the noise the reading starts from, making a run repeatable.
    /// Different seeds give different readings of the same text in the same
    /// voice.
    #[arg(long, default_value_t = 0, value_name = "int")]
    pub(crate) seed: u64,

    /// Inference runtime executing the model.
    #[arg(
        long,
        value_enum,
        default_value_t = RuntimeArg::Candle,
        value_name = "runtime"
    )]
    pub(crate) runtime: RuntimeArg,

    /// Compute device. `metal` needs a build with the `metal` feature enabled,
    /// and is only available on macOS.
    #[arg(
        long,
        value_enum,
        default_value_t = DeviceArg::Cpu,
        value_name = "device"
    )]
    pub(crate) device: DeviceArg,

    /// Compute precision of the model. `f32` (the default) is the reference
    /// point every comparison is made against; `f16` halves the memory traffic
    /// and is what the reference implementation itself runs on a GPU. The
    /// vocoder and the spectrogram always run in full precision.
    #[arg(
        long,
        value_enum,
        default_value_t = EspeechPrecisionArg::F32,
        value_name = "precision"
    )]
    pub(crate) precision: EspeechPrecisionArg,

    /// Whether to mark the stress in the text before speaking it. This model
    /// reads `+` before a stressed vowel as a real input, and without it reads
    /// rare words wrong often enough to hear, so `auto` (the default) marks
    /// the text — and the reference transcript with it — using the same model
    /// `text stress` uses, downloaded on first use. Marks you wrote yourself
    /// are never moved. `off` speaks the text exactly as given.
    #[arg(
        long,
        value_enum,
        default_value_t = StressArg::Auto,
        value_name = "mode"
    )]
    pub(crate) stress: StressArg,
}

/// The `--stress` value of `tts espeech`.
#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum StressArg {
    /// Mark the stress before speaking.
    Auto,
    /// Speak the text exactly as given.
    Off,
}

/// The `--precision` value of `tts espeech`.
///
/// Deliberately not the shared [`PrecisionArg`] and not the other engine's
/// either: this checkpoint is stored in `f32` and its reference runs `f16` on a
/// GPU, so those are the two values that mean anything here.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum EspeechPrecisionArg {
    /// Half precision: less memory traffic, and what the reference runs on GPU.
    F16,
    /// Full precision (the default).
    F32,
}

/// Flags of `tts qwen3-tts`.
#[derive(Args)]
// Exactly one source of text, and the error names both when neither is given.
#[command(group(clap::ArgGroup::new("source").required(true).args(["speech", "text_file"])))]
pub(crate) struct Qwen3TtsArgs {
    /// The text to speak. Omit it when reading the text from a file with
    /// --text-file.
    // The id must differ from the global `--text` flag, which clap would
    // otherwise take this for.
    #[arg(id = "speech", value_name = "text")]
    pub(crate) text: Option<String>,

    /// Read the text to speak from a UTF-8 file instead of the argument, or
    /// from standard input with `-`. Text of any length works: it is spoken
    /// paragraph by paragraph and joined into one file (see --text-format and
    /// --pause-ms).
    #[arg(long, value_name = "path|-")]
    pub(crate) text_file: Option<PathBuf>,

    /// How to read the input: where paragraphs end and whether it carries
    /// markup. `txt` takes one paragraph per line; `md` separates paragraphs
    /// with blank lines and strips Markdown markup (headings, list and quote
    /// markers, emphasis, links — the text of a table or a code block is
    /// kept). `auto` decides by the file extension, then by the text
    /// itself, and falls back to `md`.
    #[arg(
        long,
        value_enum,
        default_value_t = TextFormatArg::Auto,
        value_name = "format"
    )]
    pub(crate) text_format: TextFormatArg,

    /// Silence inserted between paragraphs, in milliseconds. Each paragraph is
    /// first trimmed of the silence the model leaves at its edges (a few tens
    /// of milliseconds are kept as breathing room), so the gap is this value
    /// rather than whatever the model happened to add. Pieces of a single
    /// split paragraph get half of it.
    #[arg(long, default_value_t = 500, value_name = "ms")]
    pub(crate) pause_ms: u32,

    /// What to do with the loudness of each paragraph. Every paragraph is
    /// spoken as its own utterance and the model picks a level for it anew, so
    /// a long text drifts by several decibels from one to the next; `match`
    /// (the default) brings them to a common level before joining, `keep`
    /// leaves them exactly as synthesized.
    #[arg(
        long,
        value_enum,
        default_value_t = LevelsArg::Match,
        value_name = "what"
    )]
    pub(crate) levels: LevelsArg,

    /// Where to write the audio; the extension picks the format. `.wav` and
    /// `.flac` are written directly, anything ffmpeg knows (mp3, m4a, opus,
    /// ogg, …) through it — see --audio-encoder.
    #[arg(long, short, default_value = "speech.wav", value_name = "path")]
    pub(crate) output: PathBuf,

    /// Encoder for the output file. `auto` (the default) writes wav and flac
    /// with the built-in pure-Rust encoder and hands any other extension to an
    /// installed `ffmpeg`; `builtin` refuses anything but wav/flac; `ffmpeg`
    /// always shells out.
    #[arg(
        long,
        value_enum,
        default_value_t = AudioEncoderArg::Auto,
        value_name = "encoder"
    )]
    pub(crate) audio_encoder: AudioEncoderArg,

    /// Target bitrate for lossy ffmpeg formats, such as `192k` or `320k` (sets
    /// ffmpeg's `-b:a`). Ignored by the built-in encoder; omit for ffmpeg's
    /// own default.
    #[arg(long, value_name = "rate")]
    pub(crate) bitrate: Option<String>,

    /// Language to speak in, as an English name: russian, english, german,
    /// spanish, chinese, japanese, french, korean, italian, or portuguese.
    /// Omit (or pass `auto`) to let the model decide from the text.
    #[arg(long, default_value = "auto", value_name = "lang")]
    pub(crate) language: String,

    /// Preset voice: serena, vivian, uncle_fu, ryan, aiden, ono_anna, sohee,
    /// eric, or dylan. The last two are dialect voices: with the language
    /// left to `auto` (or set to `chinese`) eric speaks Sichuan and dylan
    /// Beijing Mandarin; any other explicit --language overrides that.
    #[arg(long, default_value = "serena", value_name = "name")]
    pub(crate) voice: String,

    /// Model: a published name, downloaded on first use, or a path to a
    /// checkpoint directory. Names: 0.6b-customvoice (the default),
    /// 1.7b-customvoice (larger and slower).
    #[arg(long, default_value = "0.6b-customvoice", value_name = "name|dir")]
    pub(crate) model: String,

    /// Seed for sampling, making a run repeatable.
    #[arg(long, default_value_t = 0, value_name = "int")]
    pub(crate) seed: u64,

    /// Sampling temperature; higher is more varied.
    #[arg(long, default_value_t = 0.9, value_name = "float")]
    pub(crate) temperature: f32,

    /// Candidates kept before sampling.
    #[arg(long, default_value_t = 50, value_name = "int")]
    pub(crate) top_k: usize,

    /// Penalty discouraging codes already generated.
    #[arg(long, default_value_t = 1.05, value_name = "float")]
    pub(crate) repetition_penalty: f32,

    /// Always take the most likely code instead of sampling. Deterministic,
    /// and usually flatter; mainly for reproducible comparisons. The
    /// checkpoint's own repetition penalty still applies.
    #[arg(long, conflicts_with_all = [
        "seed", "temperature", "top_k", "repetition_penalty"
    ])]
    pub(crate) greedy: bool,

    /// Inference runtime executing the model. The burn runtime computes in
    /// f32 only, so pair it with `--precision f32`; it is also several times
    /// slower here, especially on a long text and on its first run on a
    /// machine, where it compiles and tunes its GPU kernels. Prefer the
    /// default for anything longer than a phrase.
    #[arg(
        long,
        value_enum,
        default_value_t = RuntimeArg::Candle,
        value_name = "runtime"
    )]
    pub(crate) runtime: RuntimeArg,

    /// Compute device. `metal` needs a build with the `metal` feature enabled,
    /// and is only available on macOS.
    #[arg(
        long,
        value_enum,
        default_value_t = DeviceArg::Cpu,
        value_name = "device"
    )]
    pub(crate) device: DeviceArg,

    /// Compute precision of the speech model. The default depends on the
    /// runtime and device: `bf16` on candle with Metal (about half the
    /// memory, the format the weights are stored in, and what keeps the
    /// larger model within a 16 GB machine), and `f32` everywhere else — the
    /// burn runtime and candle's CPU backend compute in `f32` only. Pass
    /// `f32` for full precision and reproducible results. The codec always
    /// runs in full precision either way.
    #[arg(long, value_enum, value_name = "precision")]
    pub(crate) precision: Option<TtsPrecisionArg>,
}

/// The `--levels` value of `tts`.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum LevelsArg {
    /// Bring the paragraphs to a common loudness (the default).
    Match,
    /// Leave each paragraph at the level the model gave it.
    Keep,
}

/// The `--text-format` value of `tts`.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum TextFormatArg {
    /// Decide from the file extension, then from the text (the default).
    Auto,
    /// Plain text: one paragraph per line.
    Txt,
    /// Markdown: paragraphs separated by blank lines, markup stripped.
    Md,
}

impl TextFormatArg {
    /// The core-side format this stands for.
    pub(crate) fn to_core(self) -> trakktor_core::tts::text::TextFormat {
        use trakktor_core::tts::text::TextFormat;
        match self {
            TextFormatArg::Auto => TextFormat::Auto,
            TextFormatArg::Txt => TextFormat::Plain,
            TextFormatArg::Md => TextFormat::Markdown,
        }
    }
}

/// The `--precision` value of `tts qwen3-tts`.
///
/// Deliberately not the shared [`PrecisionArg`]: this engine's half-precision
/// option is `bf16`, the format its weights are stored in. `f16` is not
/// offered because its narrower exponent range overflows partway through
/// generation and the run degenerates into babble.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum TtsPrecisionArg {
    /// Half precision, as the weights are stored (the default on candle with
    /// Metal, the one backend that serves it).
    Bf16,
    /// Full precision: reproducible, at twice the memory (the default
    /// everywhere else).
    F32,
}

/// Flags of `asr gigaam`.
#[derive(Args)]
pub(crate) struct GigaamArgs {
    /// Path to the audio file to transcribe.
    #[arg(value_name = "audio")]
    pub(crate) audio: PathBuf,

    /// Model: a published name, downloaded on first use, or a path to a
    /// checkpoint `.ckpt` file. Names: v3_e2e_rnnt (the default — punctuated,
    /// capitalized Russian; the more accurate of the two punctuated models),
    /// v3_e2e_ctc (the same punctuated output from a CTC decoder: it ends
    /// sentences more often, at some cost in word accuracy, and is the
    /// lighter choice for CPU-only runs), v3_ctc (Russian, lowercase, no
    /// punctuation), v3_rnnt (like v3_ctc but with a transducer decoder —
    /// usually the most accurate raw text on Russian), multilingual_ctc,
    /// multilingual_large_ctc (the largest multilingual). Larger models are
    /// slower.
    #[arg(long, default_value = "v3_e2e_rnnt", value_name = "name|file")]
    pub(crate) model: String,

    /// Timestamp granularity of the output.
    #[arg(
        long,
        value_enum,
        default_value_t = TimestampsArg::Segment,
        value_name = "granularity"
    )]
    pub(crate) timestamps: TimestampsArg,

    /// Inference runtime executing the model. Both produce the same
    /// transcription; `burn` needs a build with the `burn` feature enabled,
    /// and on the CPU computes in f32 only.
    #[arg(
        long,
        value_enum,
        default_value_t = RuntimeArg::Candle,
        value_name = "runtime"
    )]
    pub(crate) runtime: RuntimeArg,

    /// Compute device. `metal` needs a build with the `metal` feature (for
    /// the candle runtime) or the `burn` feature (for the burn runtime)
    /// enabled, and is only available on macOS.
    #[arg(
        long,
        value_enum,
        default_value_t = DeviceArg::Cpu,
        value_name = "device"
    )]
    pub(crate) device: DeviceArg,

    /// Compute precision. `f16` (the default) uses about half the memory and
    /// is faster on GPU, matching how the reference runs on GPU; `f32` runs in
    /// full precision for reproducible results, at twice the memory.
    #[arg(
        long,
        value_enum,
        default_value_t = PrecisionArg::F16,
        value_name = "precision"
    )]
    pub(crate) precision: PrecisionArg,

    /// Audio decoder. `builtin` (the default) is pure Rust and needs no
    /// external tools; `ffmpeg` shells out to an installed `ffmpeg` and adds
    /// input formats the built-in decoder does not cover, such as opus, wma,
    /// and amr.
    #[arg(
        long,
        value_enum,
        default_value_t = AudioDecoderArg::Builtin,
        value_name = "decoder"
    )]
    pub(crate) audio_decoder: AudioDecoderArg,

    /// Language label to report in the output (a code like `ru`). GigaAM does
    /// not detect the language; this only annotates the result.
    #[arg(long, value_name = "lang")]
    pub(crate) language: Option<String>,

    /// Also write the transcript to files in this format: txt, vtt, srt, tsv,
    /// json, or `all`. Files are named after the audio and written into
    /// `--output-dir`; stdout still prints the result as usual.
    #[arg(long, value_enum, value_name = "format")]
    pub(crate) output_format: Option<OutputFormatArg>,

    /// Directory for files written by `--output-format`, created if missing
    /// (default: the current directory).
    #[arg(long, default_value = ".", value_name = "dir")]
    pub(crate) output_dir: PathBuf,
}

/// Flags of `asr vosk`.
#[derive(Args)]
pub(crate) struct VoskArgs {
    /// Path to the audio file to transcribe.
    #[arg(value_name = "audio")]
    pub(crate) audio: PathBuf,

    /// Model: a published name, downloaded on first use, or a path to a
    /// directory holding a compatible export (encoder.onnx, decoder.onnx,
    /// joiner.onnx, tokens.txt). Names: ru (large Russian, offline; the
    /// default), small-ru (small Russian, offline), streaming-ru (large
    /// Russian, streaming), small-streaming-ru (small Russian, streaming),
    /// small-streaming-bn (Bengali, streaming), tg (Tajik, offline). All emit
    /// lowercase text without punctuation.
    #[arg(long, default_value = "ru", value_name = "name|dir")]
    pub(crate) model: String,

    /// Transducer search. `beam` (the default) is modified beam search, as
    /// used by the reference; `greedy` is faster and usually slightly less
    /// accurate.
    #[arg(
        long,
        value_enum,
        default_value_t = DecodingArg::Beam,
        value_name = "search"
    )]
    pub(crate) decoding: DecodingArg,

    /// Timestamp granularity of the output.
    #[arg(
        long,
        value_enum,
        default_value_t = TimestampsArg::Segment,
        value_name = "granularity"
    )]
    pub(crate) timestamps: TimestampsArg,

    /// Inference runtime executing the model. Both produce the same
    /// transcription; `burn` needs a build with the `burn` feature enabled,
    /// and on the CPU computes in f32 only.
    #[arg(
        long,
        value_enum,
        default_value_t = RuntimeArg::Candle,
        value_name = "runtime"
    )]
    pub(crate) runtime: RuntimeArg,

    /// Compute device. `metal` needs a build with the `metal` feature (for
    /// the candle runtime) or the `burn` feature (for the burn runtime)
    /// enabled, and is only available on macOS.
    #[arg(
        long,
        value_enum,
        default_value_t = DeviceArg::Cpu,
        value_name = "device"
    )]
    pub(crate) device: DeviceArg,

    /// Compute precision of the encoder. `f16` (the default) uses about half
    /// the memory and is faster on GPU; `f32` runs in full precision for
    /// reproducible results, at twice the memory. The transducer decoder
    /// always runs in f32.
    #[arg(
        long,
        value_enum,
        default_value_t = PrecisionArg::F16,
        value_name = "precision"
    )]
    pub(crate) precision: PrecisionArg,

    /// Audio decoder. `builtin` (the default) is pure Rust and needs no
    /// external tools; `ffmpeg` shells out to an installed `ffmpeg` and adds
    /// input formats the built-in decoder does not cover, such as opus, wma,
    /// and amr.
    #[arg(
        long,
        value_enum,
        default_value_t = AudioDecoderArg::Builtin,
        value_name = "decoder"
    )]
    pub(crate) audio_decoder: AudioDecoderArg,

    /// Language label to report in the output (a code like `ru`). Vosk does
    /// not detect the language; this only annotates the result.
    #[arg(long, value_name = "lang")]
    pub(crate) language: Option<String>,

    /// Also write the transcript to files in this format: txt, vtt, srt, tsv,
    /// json, or `all`. Files are named after the audio and written into
    /// `--output-dir`; stdout still prints the result as usual.
    #[arg(long, value_enum, value_name = "format")]
    pub(crate) output_format: Option<OutputFormatArg>,

    /// Directory for files written by `--output-format`, created if missing
    /// (default: the current directory).
    #[arg(long, default_value = ".", value_name = "dir")]
    pub(crate) output_dir: PathBuf,
}

/// The `--decoding` value of `asr vosk`.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum DecodingArg {
    /// Modified beam search (the reference default).
    Beam,
    /// Greedy search: one token per frame, faster.
    Greedy,
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

    /// Also write the transcript to files in this format: txt (one line per
    /// segment), vtt or srt (subtitles), tsv (start/end in milliseconds plus
    /// text), json (the full result), or `all` for every format. Files are
    /// named after the audio and written into `--output-dir`; stdout still
    /// prints the result as usual.
    #[arg(long, value_enum, value_name = "format")]
    pub(crate) output_format: Option<OutputFormatArg>,

    /// Directory for files written by `--output-format`, created if missing
    /// (default: the current directory).
    #[arg(long, default_value = ".", value_name = "dir")]
    pub(crate) output_dir: PathBuf,

    /// Model: a published name, downloaded on first use, or a path to a
    /// checkpoint directory. Names: tiny, tiny.en, base, base.en, small,
    /// small.en, medium, medium.en, large-v1, large-v2, large-v3, large,
    /// turbo, large-v3-turbo, plus podlodka and podlodka-turbo — fine-tunes
    /// of large-v3 and turbo specialized for Russian. Larger models are
    /// slower and more accurate.
    #[arg(long, default_value = "tiny", value_name = "name|dir")]
    pub(crate) model: String,

    /// Inference runtime executing the model. Both produce the same
    /// transcription; `burn` needs a build with the `burn` feature enabled,
    /// and on the CPU computes in f32 only.
    #[arg(
        long,
        value_enum,
        default_value_t = RuntimeArg::Candle,
        value_name = "runtime"
    )]
    pub(crate) runtime: RuntimeArg,

    /// Compute device. `metal` needs a build with the `metal` feature (for
    /// the candle runtime) or the `burn` feature (for the burn runtime)
    /// enabled, and is only available on macOS.
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

    /// Audio decoder. `builtin` (the default) is pure Rust and needs no
    /// external tools; `ffmpeg` shells out to an installed `ffmpeg` and adds
    /// input formats the built-in decoder does not cover, such as opus, wma,
    /// and amr.
    #[arg(
        long,
        value_enum,
        default_value_t = AudioDecoderArg::Builtin,
        value_name = "decoder"
    )]
    pub(crate) audio_decoder: AudioDecoderArg,

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

    /// Start transcribing at this offset: seconds or a `[[HH:]MM:]SS[.mmm]`
    /// clock (for example `90`, `1:30`, or `1:02:03.250`), to millisecond
    /// precision. Used alone, it runs to the end of the audio. Cannot be
    /// combined with `--clip-timestamps`.
    #[arg(long, value_name = "time", conflicts_with = "clip_timestamps")]
    pub(crate) start: Option<Timecode>,

    /// Stop transcribing at this offset, in the same format as `--start`. Used
    /// alone, it runs from the beginning. Cannot be combined with
    /// `--clip-timestamps`.
    #[arg(long, value_name = "time", conflicts_with = "clip_timestamps")]
    pub(crate) end: Option<Timecode>,

    /// With `--timestamps word`: skip silent stretches longer than this many
    /// seconds when a probable hallucination is detected, or `none`.
    #[arg(long, default_value = "none", value_name = "float|none")]
    pub(crate) hallucination_silence_threshold: OrNone<f64>,

    /// Detect speech with Silero voice-activity detection and drop non-speech
    /// (silence, music, noise) before transcribing — fewer repetition loops
    /// and faster on sparse audio. Off by default. Cannot be combined with
    /// `--clip-timestamps`; combined with `--start`/`--end` it detects speech
    /// within that range.
    #[arg(long, conflicts_with = "clip_timestamps")]
    pub(crate) vad: bool,

    /// VAD speech-probability threshold in 0..=1; higher detects less speech.
    #[arg(long, default_value_t = 0.5, value_name = "float")]
    pub(crate) vad_threshold: f32,

    /// VAD: drop detected speech shorter than this many milliseconds.
    #[arg(long, default_value_t = 250, value_name = "ms")]
    pub(crate) vad_min_speech_duration_ms: u32,

    /// VAD: a silence shorter than this many milliseconds does not split a
    /// speech segment — brief pauses are bridged.
    #[arg(long, default_value_t = 100, value_name = "ms")]
    pub(crate) vad_min_silence_duration_ms: u32,

    /// VAD: padding added to each side of a detected speech segment, in
    /// milliseconds.
    #[arg(long, default_value_t = 30, value_name = "ms")]
    pub(crate) vad_speech_pad_ms: u32,

    /// VAD: force-split speech longer than this many seconds, or `none` to
    /// never split.
    #[arg(long, default_value = "none", value_name = "float|none")]
    pub(crate) vad_max_speech_duration_s: OrNone<f64>,
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
    /// The CPU.
    Cpu,
    /// The GPU via Metal, on macOS.
    Metal,
}

/// The `--runtime` value of the model-backed commands (`asr` engines and
/// `text structify`).
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum RuntimeArg {
    /// The candle runtime (the default).
    Candle,
    /// The burn runtime; needs a build with the `burn` feature.
    Burn,
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

    pub(crate) fn to_gigaam(self) -> trakktor_core::asr::gigaam::Precision {
        match self {
            PrecisionArg::F16 => trakktor_core::asr::gigaam::Precision::F16,
            PrecisionArg::F32 => trakktor_core::asr::gigaam::Precision::F32,
        }
    }

    pub(crate) fn to_vosk(self) -> trakktor_core::asr::vosk::Precision {
        match self {
            PrecisionArg::F16 => trakktor_core::asr::vosk::Precision::F16,
            PrecisionArg::F32 => trakktor_core::asr::vosk::Precision::F32,
        }
    }

    pub(crate) fn to_structify(self) -> trakktor_core::structify::Precision {
        match self {
            PrecisionArg::F16 => trakktor_core::structify::Precision::F16,
            PrecisionArg::F32 => trakktor_core::structify::Precision::F32,
        }
    }

    pub(crate) fn to_punctuate(self) -> trakktor_core::punctuate::Precision {
        match self {
            PrecisionArg::F16 => trakktor_core::punctuate::Precision::F16,
            PrecisionArg::F32 => trakktor_core::punctuate::Precision::F32,
        }
    }

    pub(crate) fn to_stress(self) -> trakktor_core::stress::Precision {
        match self {
            PrecisionArg::F16 => trakktor_core::stress::Precision::F16,
            PrecisionArg::F32 => trakktor_core::stress::Precision::F32,
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

/// The `--output-format` value of `asr whisper`: which transcript files to
/// write.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum OutputFormatArg {
    /// Plain text: one line per segment.
    Txt,
    /// WebVTT subtitles.
    Vtt,
    /// SubRip (SRT) subtitles.
    Srt,
    /// Tab-separated values: start and end in milliseconds, then text.
    Tsv,
    /// The full JSON result (the same object printed to stdout).
    Json,
    /// Every format above, in one run.
    All,
}

/// The `--audio-decoder` value of `asr whisper`.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum AudioDecoderArg {
    /// Built-in pure-Rust decoder (the default); no external tools.
    Builtin,
    /// An external `ffmpeg` process; supports more input formats.
    Ffmpeg,
}

/// A clip boundary for `--start`/`--end`: a plain number of seconds (`90`,
/// `92.5`) or a `[[HH:]MM:]SS[.mmm]` clock (`5:30`, `1:02:03.250`). Held as
/// seconds; millisecond precision is preserved.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Timecode(pub(crate) f64);

impl std::str::FromStr for Timecode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, String> {
        let value = value.trim();
        let fields: Vec<&str> = value.split(':').collect();
        if fields.len() > 3 {
            return Err(format!(
                "`{value}`: expected seconds or [[HH:]MM:]SS[.mmm]"
            ));
        }
        let mut seconds = 0.0f64;
        for (i, field) in fields.iter().enumerate() {
            let parsed: f64 = field
                .trim()
                .parse()
                .map_err(|_| format!("`{value}`: `{field}` is not a number"))?;
            if parsed < 0.0 {
                return Err(format!("`{value}` must be non-negative"));
            }
            // Only the last (seconds) field may be fractional; earlier fields
            // are whole hours/minutes.
            let is_last = i + 1 == fields.len();
            if !is_last && parsed.fract() != 0.0 {
                return Err(format!(
                    "`{value}`: only the seconds field may be fractional"
                ));
            }
            let unit = 60f64.powi((fields.len() - 1 - i) as i32);
            seconds += parsed * unit;
        }
        Ok(Timecode(seconds))
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
pub(crate) enum TextCommand {
    /// Split text into paragraphs.
    ///
    /// Reads a UTF-8 text file, collapses its existing line breaks — a
    /// transcript's segment breaks are not paragraph breaks — and re-groups it
    /// into logical paragraphs with a local SaT (Segment any Text) model: an
    /// XLM-RoBERTa network that scores each position for a paragraph boundary.
    /// The first use of a model downloads it into the model directory
    /// (~/.trakktor by default; see --model-dir), and later runs reuse it,
    /// fully offline. A higher --threshold yields fewer, larger paragraphs;
    /// pass --model sat-3l-sm for finer, sentence-level splitting instead.
    /// Output is JSON by default — the model and a list of paragraphs, each
    /// with its character range and text; pass --text for the paragraphs
    /// separated by blank lines. Multilingual, including Russian and English.
    Structify(StructifyArgs),

    /// Restore punctuation and capitalization in raw text.
    ///
    /// Reads a UTF-8 text file of lowercase, unpunctuated text — the typical
    /// output of the Vosk and GigaAM speech engines — and restores punctuation,
    /// capitalization (including acronyms like NATO and U.S.), and sentence
    /// boundaries with a local multilingual model, fully offline. The first use
    /// of a model downloads it into the model directory (~/.trakktor by
    /// default; see --model-dir), and later runs reuse it. Output is JSON by
    /// default — the model, the restored text, and the list of sentences; pass
    /// --text for the restored text alone. Handles 47 languages, including
    /// Russian and English. A natural pipeline is `asr vosk` → `text punctuate`
    /// → `text structify`.
    Punctuate(PunctuateArgs),

    /// Mark the stressed vowel in Russian text (and restore the letter ё).
    ///
    /// Reads a UTF-8 text file of Russian and gives back the same text with the
    /// stress marked — `+` before the stressed vowel by default, which is what
    /// `tts espeech` reads — and with the letter ё written where it belongs,
    /// fully offline. Russian writes neither, and a speech synthesizer needs
    /// both: an unmarked word is read by guesswork, and a pair like все/всё
    /// cannot be told apart by a stress mark at all. Words you have already
    /// marked yourself are never re-marked, and --dict lets you fix the rest
    /// once and for all. The first use of a model downloads it into the model
    /// directory (~/.trakktor by default; see --model-dir), and later runs
    /// reuse it. Output is JSON by default — the marked text, counts, and the
    /// words left unmarked; pass --text for the marked text alone.
    Stress(StressArgs),
}

/// Flags of `text stress`.
#[derive(Args)]
pub(crate) struct StressArgs {
    /// Path to the UTF-8 text file to mark.
    #[arg(value_name = "input")]
    pub(crate) input: PathBuf,

    /// Model: a published name, downloaded on first use, or a path to a model
    /// directory. Names: silero-ru (Russian; an n-gram accentor with a
    /// stress and a ё head, plus a BERT solver for words whose spelling does
    /// not say how they are read).
    #[arg(
        long,
        default_value = trakktor_core::stress::DEFAULT_MODEL,
        value_name = "name|dir"
    )]
    pub(crate) model: String,

    /// Form of the stress mark in the output. `plus` writes `+` before the
    /// stressed vowel — the form the speech engines read; `acute` writes the
    /// combining acute accent after it, the form dictionaries and corpora use.
    /// Either form is also accepted on input and is never overwritten.
    #[arg(
        long,
        value_enum,
        default_value_t = MarkerArg::Plus,
        value_name = "form"
    )]
    pub(crate) marker: MarkerArg,

    /// Whether to restore the letter ё. `auto` writes it where it belongs —
    /// which is the only way to tell все from всё; `off` leaves every letter
    /// of the input exactly as it was and only adds marks.
    #[arg(
        long,
        value_enum,
        default_value_t = YoArg::Auto,
        value_name = "mode"
    )]
    pub(crate) yo: YoArg,

    /// Your own dictionary of stressed spellings, applied before the model —
    /// one marked word per line (`ф+орзац`, `Корол+ёв`), `#` starts a comment.
    /// Names, terms, and rare words belong here; repeat the flag for several
    /// files. A word listed here is spelled the way you wrote it, whatever the
    /// model would have said.
    #[arg(long = "dict", value_name = "path")]
    pub(crate) dictionaries: Vec<PathBuf>,

    /// Inference runtime executing the model. Both produce the same text;
    /// `burn` needs a build with the `burn` feature enabled, and on the CPU
    /// computes in f32 only.
    #[arg(
        long,
        value_enum,
        default_value_t = RuntimeArg::Candle,
        value_name = "runtime"
    )]
    pub(crate) runtime: RuntimeArg,

    /// Compute device. `metal` needs a build with the `metal` feature (for the
    /// candle runtime) or the `burn` feature (for the burn runtime) enabled,
    /// and is only available on macOS.
    #[arg(
        long,
        value_enum,
        default_value_t = DeviceArg::Cpu,
        value_name = "device"
    )]
    pub(crate) device: DeviceArg,

    /// Compute precision. `f32` (the default here) computes in full precision;
    /// `f16` uses about half the memory and is faster, but every decision this
    /// model makes is a threshold that half precision can flip.
    #[arg(
        long,
        value_enum,
        default_value_t = PrecisionArg::F32,
        value_name = "precision"
    )]
    pub(crate) precision: PrecisionArg,

    /// Words per forward batch.
    #[arg(long, default_value_t = 256, value_name = "int")]
    pub(crate) batch_size: usize,
}

/// The `--marker` value of `text stress`.
#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum MarkerArg {
    /// `+` before the stressed vowel.
    Plus,
    /// Combining acute accent after the stressed vowel.
    Acute,
}

impl MarkerArg {
    pub(crate) fn to_core(self) -> trakktor_core::stress::Marker {
        match self {
            MarkerArg::Plus => trakktor_core::stress::Marker::Plus,
            MarkerArg::Acute => trakktor_core::stress::Marker::Acute,
        }
    }
}

/// The `--yo` value of `text stress`.
#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum YoArg {
    /// Write ё where it belongs.
    Auto,
    /// Leave the letters of the input alone.
    Off,
}

/// Flags of `text punctuate`.
#[derive(Args)]
pub(crate) struct PunctuateArgs {
    /// Path to the UTF-8 text file to punctuate.
    #[arg(value_name = "input")]
    pub(crate) input: PathBuf,

    /// Model: a published name, downloaded on first use, or a path to a model
    /// directory. Names: xlmr-47lang (multilingual XLM-RoBERTa covering 47
    /// languages, including Russian and English; punctuation, capitalization,
    /// and sentence boundaries).
    #[arg(long, default_value = "xlmr-47lang", value_name = "name|dir")]
    pub(crate) model: String,

    /// Inference runtime executing the model. Both produce the same result;
    /// `burn` needs a build with the `burn` feature enabled, and on the CPU
    /// computes in f32 only.
    #[arg(
        long,
        value_enum,
        default_value_t = RuntimeArg::Candle,
        value_name = "runtime"
    )]
    pub(crate) runtime: RuntimeArg,

    /// Compute device. `metal` needs a build with the `metal` feature (for the
    /// candle runtime) or the `burn` feature (for the burn runtime) enabled,
    /// and is only available on macOS.
    #[arg(
        long,
        value_enum,
        default_value_t = DeviceArg::Cpu,
        value_name = "device"
    )]
    pub(crate) device: DeviceArg,

    /// Compute precision. `f16` (the default) uses about half the memory and
    /// is faster; `f32` runs in full precision for reproducible results,
    /// at twice the memory.
    #[arg(
        long,
        value_enum,
        default_value_t = PrecisionArg::F16,
        value_name = "precision"
    )]
    pub(crate) precision: PrecisionArg,

    /// Overlap between consecutive windows, in tokens, when the text is longer
    /// than one window. The seam is split evenly between the two windows; an
    /// odd value is rounded down to even.
    #[arg(long, default_value_t = 16, value_name = "int")]
    pub(crate) overlap: usize,

    /// Windows per forward batch — the main lever on GPU utilization.
    #[arg(long, default_value_t = 16, value_name = "int")]
    pub(crate) batch_size: usize,
}

/// Flags of `text structify`.
#[derive(Args)]
pub(crate) struct StructifyArgs {
    /// Path to the UTF-8 text file to structure.
    #[arg(value_name = "input")]
    pub(crate) input: PathBuf,

    /// Model: a published name, downloaded on first use, or a path to a
    /// checkpoint directory. The `-no-limited-lookahead` models (sat-1l,
    /// sat-3l, sat-6l, sat-9l, sat-12l [default]) score paragraph breaks —
    /// reader-style paragraphs. Depth matters: shallow models give a weakly
    /// calibrated paragraph signal, so sat-12l is the default; smaller ones
    /// are faster but need a lower --threshold and segment less cleanly.
    /// The `-sm` models (sat-1l-sm … sat-12l-sm) score finer sentence
    /// breaks instead.
    #[arg(
        long,
        default_value = trakktor_core::structify::DEFAULT_MODEL,
        value_name = "name|dir"
    )]
    pub(crate) model: String,

    /// Inference runtime executing the model. Both produce the same
    /// paragraphs; `burn` needs a build with the `burn` feature enabled, and
    /// on the CPU computes in f32 only.
    #[arg(
        long,
        value_enum,
        default_value_t = RuntimeArg::Candle,
        value_name = "runtime"
    )]
    pub(crate) runtime: RuntimeArg,

    /// Compute device. `metal` needs a build with the `metal` feature (for
    /// the candle runtime) or the `burn` feature (for the burn runtime)
    /// enabled, and is only available on macOS.
    #[arg(
        long,
        value_enum,
        default_value_t = DeviceArg::Cpu,
        value_name = "device"
    )]
    pub(crate) device: DeviceArg,

    /// Compute precision. `f16` (the default) uses about half the memory and
    /// is faster; `f32` runs in full precision for reproducible results,
    /// at twice the memory.
    #[arg(
        long,
        value_enum,
        default_value_t = PrecisionArg::F16,
        value_name = "precision"
    )]
    pub(crate) precision: PrecisionArg,

    /// Paragraph-boundary probability threshold in 0..=1; higher yields fewer,
    /// longer paragraphs.
    #[arg(long, default_value_t = 0.5, value_name = "float")]
    pub(crate) threshold: f32,

    /// Window step in tokens. A smaller stride overlaps windows more —
    /// steadier boundaries, more compute.
    #[arg(long, default_value_t = 256, value_name = "int")]
    pub(crate) stride: usize,

    /// Windows per forward batch — the main lever on GPU utilization.
    #[arg(long, default_value_t = 32, value_name = "int")]
    pub(crate) batch_size: usize,
}

#[derive(Subcommand)]
pub(crate) enum VadCommand {
    /// Report the detected speech spans as JSON, without writing any audio.
    ///
    /// Prints the speech intervals (seconds, on the original timeline) and
    /// summary statistics — how much of the file is speech versus silence, and
    /// the longest pause. A safe, side-effect-free way to inspect a file.
    Timeline(VadTimelineArgs),

    /// Write a new audio file with non-speech removed (or kept).
    ///
    /// Concatenates the detected speech into one file, dropping silence, music,
    /// and noise; `--keep non-speech` inverts it (keep the non-speech, drop the
    /// speech). The output copies the decoded original at full quality — its
    /// own sample rate, channels, and bit depth — with a short fade at each
    /// join so edits do not click. With `--max-silence-ms`, long pauses are
    /// shortened to that length instead of being removed.
    Cut(VadCutArgs),

    /// Write one file per detected speech span (one clip per utterance).
    ///
    /// Splits the recording on silence into many files, each holding a single
    /// span, dropping any shorter than `--min-duration-ms`. `--keep non-speech`
    /// splits out the non-speech spans instead. Like `cut`, every clip is the
    /// full-quality original.
    Split(VadSplitArgs),
}

/// The audio input and detection options shared by every `vad` subcommand.
#[derive(Args)]
pub(crate) struct VadDetectArgs {
    /// Path to the audio file.
    #[arg(value_name = "audio")]
    pub(crate) audio: PathBuf,

    /// Base defaults that the individual detection and shaping flags below
    /// override: `tight` removes silence aggressively (canonical Silero);
    /// `asr` bridges long pauses to keep speech in large chunks, like a
    /// transcription front-end; `natural` keeps more breathing room.
    #[arg(
        long,
        value_enum,
        default_value_t = PresetArg::Tight,
        value_name = "preset"
    )]
    pub(crate) preset: PresetArg,

    /// Speech-probability threshold in 0..=1; higher detects less speech.
    /// Defaults to the value from `--preset`.
    #[arg(long, value_name = "float")]
    pub(crate) threshold: Option<f32>,

    /// Drop detected speech shorter than this many milliseconds. Defaults to
    /// the value from `--preset`.
    #[arg(long, value_name = "ms")]
    pub(crate) min_speech_duration_ms: Option<u32>,

    /// A silence shorter than this many milliseconds does not split a speech
    /// span — brief pauses are bridged. Defaults to the value from `--preset`.
    #[arg(long, value_name = "ms")]
    pub(crate) min_silence_duration_ms: Option<u32>,

    /// Padding added to each side of a detected speech span, in milliseconds.
    /// Defaults to the value from `--preset`.
    #[arg(long, value_name = "ms")]
    pub(crate) speech_pad_ms: Option<u32>,

    /// Force-split speech longer than this many seconds; omit to never split.
    #[arg(long, value_name = "float")]
    pub(crate) max_speech_duration_s: Option<f64>,

    /// Work only within this window — a start offset in seconds or a
    /// `[[HH:]MM:]SS[.mmm]` clock. Cutting, splitting, and inversion all stay
    /// inside `[start, end]`. Used alone, the window runs to the end of the
    /// audio.
    #[arg(long, value_name = "time")]
    pub(crate) start: Option<Timecode>,

    /// End of the working window, in the same format as `--start`. Used alone,
    /// the window starts at the beginning of the audio.
    #[arg(long, value_name = "time")]
    pub(crate) end: Option<Timecode>,
}

/// The shaping and output options shared by `vad cut` and `vad split`.
#[derive(Args)]
pub(crate) struct VadShapeArgs {
    /// Which side of the speech/non-speech split to keep in the output.
    #[arg(
        long,
        value_enum,
        default_value_t = KeepArg::Speech,
        value_name = "what"
    )]
    pub(crate) keep: KeepArg,

    /// Output audio format. With the built-in encoder: `wav` (reproduces any
    /// source exactly) or `flac` (smaller, lossless, integer sources up to
    /// 24-bit). With `--audio-encoder ffmpeg`, any format ffmpeg writes by
    /// extension — for example mp3, aac, m4a, opus, or ogg.
    #[arg(long, default_value = "wav", value_name = "format")]
    pub(crate) format: String,

    /// Encoder for the output. `builtin` (the default) is pure Rust and writes
    /// wav/flac with no external tools; `ffmpeg` shells out to an installed
    /// `ffmpeg` and writes the many formats it supports (mp3, aac, opus, m4a,
    /// …); `auto` picks between them by `--format`. The samples are
    /// re-encoded, so cuts stay sample-accurate.
    #[arg(
        long,
        value_enum,
        default_value_t = AudioEncoderArg::Builtin,
        value_name = "encoder"
    )]
    pub(crate) audio_encoder: AudioEncoderArg,

    /// Target bitrate for lossy ffmpeg formats, such as `192k` or `320k` (sets
    /// ffmpeg's `-b:a`). Only affects `--audio-encoder ffmpeg`; omit for
    /// ffmpeg's own default.
    #[arg(long, value_name = "rate")]
    pub(crate) bitrate: Option<String>,

    /// Directory for the written file(s), created if missing.
    #[arg(long, default_value = ".", value_name = "dir")]
    pub(crate) output_dir: PathBuf,

    /// Linear fade at each cut boundary, in milliseconds, to avoid clicks.
    /// Defaults to the value from `--preset`.
    #[arg(long, value_name = "ms")]
    pub(crate) fade_ms: Option<u32>,

    /// Merge kept spans separated by a gap smaller than this many
    /// milliseconds. Defaults to the value from `--preset`.
    #[arg(long, value_name = "ms")]
    pub(crate) merge_gap_ms: Option<u32>,

    /// Extra audio kept on each side of a span, in milliseconds — added to the
    /// detector's own padding. Defaults to the value from `--preset`.
    #[arg(long, value_name = "ms")]
    pub(crate) margin_ms: Option<u32>,
}

/// Flags of `vad timeline`.
#[derive(Args)]
pub(crate) struct VadTimelineArgs {
    #[command(flatten)]
    pub(crate) detect: VadDetectArgs,
}

/// Flags of `vad cut`.
#[derive(Args)]
pub(crate) struct VadCutArgs {
    #[command(flatten)]
    pub(crate) detect: VadDetectArgs,

    #[command(flatten)]
    pub(crate) shape: VadShapeArgs,

    /// Keep at most this much of each pause instead of removing it — collapse
    /// long silences to a fixed gap rather than cutting them out. Omit to
    /// remove non-speech entirely. Applies only with `--keep speech`.
    #[arg(long, value_name = "ms")]
    pub(crate) max_silence_ms: Option<u32>,
}

/// Flags of `vad split`.
#[derive(Args)]
pub(crate) struct VadSplitArgs {
    #[command(flatten)]
    pub(crate) detect: VadDetectArgs,

    #[command(flatten)]
    pub(crate) shape: VadShapeArgs,

    /// Drop clips shorter than this many milliseconds.
    #[arg(long, default_value_t = 0, value_name = "ms")]
    pub(crate) min_duration_ms: u32,
}

/// The `--keep` value of `vad cut`/`vad split`.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum KeepArg {
    /// Keep speech, drop non-speech (the default).
    Speech,
    /// Keep non-speech, drop speech (the inversion).
    NonSpeech,
}

/// The `--audio-encoder` value of the commands that write audio.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum AudioEncoderArg {
    /// Pick by the format: built-in for wav and flac, ffmpeg for the rest.
    Auto,
    /// Built-in pure-Rust encoder: writes wav or flac.
    Builtin,
    /// An external `ffmpeg` process; writes many more formats (mp3, aac, …).
    Ffmpeg,
}

/// The `--preset` value of `vad`.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum PresetArg {
    /// Aggressive silence removal, canonical Silero (the default).
    Tight,
    /// Bridge long pauses, keeping speech in large chunks (an ASR front-end).
    Asr,
    /// Keep more breathing room around speech.
    Natural,
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

/// Reports a usage error (exit code 2) in clap's own style — used for value
/// conflicts clap cannot express declaratively, like `--end` before `--start`.
pub(crate) fn usage_error(message: &str) -> ! {
    Cli::command()
        .error(ErrorKind::ValueValidation, message)
        .exit()
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
                &global.model_dir()?,
                global.json(),
                global.pretty,
            ),
            AsrCommand::Gigaam(args) => crate::asr::run_gigaam(
                args,
                &global.model_dir()?,
                global.json(),
                global.pretty,
            ),
            AsrCommand::Vosk(args) => crate::asr::run_vosk(
                args,
                &global.model_dir()?,
                global.json(),
                global.pretty,
            ),
        },
        Command::Tts { command } => match command {
            TtsCommand::Qwen3Tts(args) => crate::tts::run_qwen3_tts(
                args,
                &global.model_dir()?,
                global.json(),
                global.pretty,
            ),
            TtsCommand::Espeech(args) => crate::tts::run_espeech(
                args,
                &global.model_dir()?,
                global.json(),
                global.pretty,
            ),
            TtsCommand::Silero(args) => {
                if args.text.is_none() &&
                    args.text_file.is_none() &&
                    !args.voice.eq_ignore_ascii_case("list")
                {
                    Cli::command()
                        .error(
                            ErrorKind::MissingRequiredArgument,
                            "the following required arguments were not \
                             provided:\n  <text|--text-file <path|->>",
                        )
                        .exit();
                }
                crate::tts::run_silero(
                    args,
                    &global.model_dir()?,
                    global.json(),
                    global.pretty,
                )
            },
        },
        Command::Vad { command } => match command {
            VadCommand::Timeline(args) => {
                crate::vad::run_timeline(args, global.json(), global.pretty)
            },
            VadCommand::Cut(args) => {
                crate::vad::run_cut(args, global.json(), global.pretty)
            },
            VadCommand::Split(args) => {
                crate::vad::run_split(args, global.json(), global.pretty)
            },
        },
        Command::Text { command } => match command {
            TextCommand::Structify(args) => crate::structify::run_structify(
                args,
                &global.model_dir()?,
                global.json(),
                global.pretty,
            ),
            TextCommand::Punctuate(args) => crate::punctuate::run_punctuate(
                args,
                &global.model_dir()?,
                global.json(),
                global.pretty,
            ),
            TextCommand::Stress(args) => crate::stress::run_stress(
                args,
                &global.model_dir()?,
                global.json(),
                global.pretty,
            ),
        },
        Command::Ocr { command } => match command {
            OcrCommand::Paddle(args) => crate::ocr::run_paddle(
                args,
                &global.model_dir()?,
                global.json(),
                global.pretty,
            ),
            OcrCommand::Vl(args) => crate::ocr::run_vl(
                args,
                &global.model_dir()?,
                global.json(),
                global.pretty,
            ),
            OcrCommand::Layout(args) => crate::ocr::run_layout(
                args,
                &global.model_dir()?,
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
