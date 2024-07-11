use clap::{Parser, Subcommand};

pub mod aws_batch;

#[derive(Parser, Debug)]
#[command(about, long_about = None)]
pub struct Cli {
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
    /// Handle and manage jobs within AWS Batch.
    AwsBatch(self::aws_batch::AwsBatch),
}
