use std::sync::Arc;

use clap::{Args, Parser, Subcommand};
use trakktor::{
    delete::DeleteArgs, download::DownloadArgs, transcribe::TranscribeJobArgs,
};

#[derive(Parser, Debug)]
#[command(about, long_about = None)]
pub struct Cli {
    /// The AWS profile to use.
    #[arg(short('p'), long)]
    pub aws_profile: Arc<str>,
    /// The prefix to use for the CloudFormation stack names.
    #[arg(short, long, default_value = "trakktor")]
    pub stack_prefix: Arc<str>,
    /// Whether to run in development mode.
    #[arg(long)]
    pub dev: bool,
    /// The verbosity level (0-3).
    #[arg(long, default_value_t = 1)]
    pub verbosity: u8,
    #[clap(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Initialize the Trakktor stack.
    Initialize(Initialize),
    /// List all jobs.
    List,
    /// Download the result of a job.
    Download(DownloadArgs),
    /// Delete a job.
    Delete(DeleteArgs),
    /// Run a transcription job.
    Transcribe(TranscribeJobArgs),
}

#[derive(Args, Debug)]
pub struct Initialize {
    /// Silently agree to disclaimer.
    #[arg(long)]
    pub agree: bool,
}
