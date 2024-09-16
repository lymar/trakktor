use std::sync::Arc;

use clap::{Parser, Subcommand, ValueHint};
use trakktor::{
    ai_chat::AIChat, embedding::EmbeddingsPlatform,
    llm::ChatCompletionPlatform, structify_text::StructifyText,
};

pub mod aws_batch;
pub mod structify_text;

#[derive(Parser, Debug)]
#[command(about, long_about = None, arg_required_else_help = true)]
pub struct Cli {
    /// Whether to run in development mode.
    #[arg(long)]
    pub dev: bool,
    /// The verbosity level (0-3).
    #[arg(long, default_value_t = 1)]
    pub verbosity: u8,
    /// The API key to use for OpenAI.
    #[arg(long, env = "OPENAI_API_KEY")]
    pub openai_api_key: Option<Arc<str>>,
    /// The server URL to use for OpenAI.
    #[arg(long, value_hint = ValueHint::Url, value_parser = url::Url::parse)]
    pub openai_server_url: Option<url::Url>,
    /// The chat platform to use for chat tasks.
    #[arg(long)]
    pub chat_platform: Option<ChatCompletionPlatform>,
    /// The model to use for chat tasks.
    #[arg(long)]
    pub chat_model: Option<Arc<str>>,
    /// The embeddings platform to use for embeddings tasks.
    #[arg(long)]
    pub embeddings_platform: Option<EmbeddingsPlatform>,
    /// The model to use for embeddings tasks.
    #[arg(long)]
    pub embeddings_model: Option<Arc<str>>,

    #[clap(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Handle and manage jobs within AWS Batch.
    AwsBatch(self::aws_batch::AwsBatch),
    /// Run AI to process chat messages from a file.
    AIChat(AIChat),
    /// Automatically structure and summarize unstructured text into sections
    /// and paragraphs.
    StructifyText(StructifyText),
}
