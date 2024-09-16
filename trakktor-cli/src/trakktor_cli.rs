mod cli;

use std::sync::Arc;

pub use cli::Cli;
use cli::Commands;
use trakktor::{
    ai_chat::{run_ai_chat, AllChatProviders},
    embedding::{EmbeddingsAPI, EmbeddingsPlatform},
    llm::{ChatCompletionAPI, ChatCompletionPlatform},
    open_ai::OpenAiAPI,
};

impl Cli {
    pub async fn run(self) -> anyhow::Result<()> {
        match &self.command {
            Commands::AwsBatch(aws_batch) => {
                self.run_aws_batch(aws_batch).await?;
            },
            Commands::AIChat(ai_chat) => {
                let all_providers = AllChatProviders {
                    open_ai: self.mk_open_ai_api(),
                };
                run_ai_chat(
                    ai_chat,
                    &self.chat_platform,
                    &self.chat_model,
                    &all_providers,
                )
                .await?;
            },
            Commands::StructifyText(structify_text) => {
                self.structify_text(structify_text).await?;
            },
        }

        Ok(())
    }

    fn mk_open_ai_api(&self) -> OpenAiAPI {
        OpenAiAPI {
            api_key: self.openai_api_key.clone(),
            server_url: self.openai_server_url.clone().map(|url| Arc::new(url)),
            chat_model: self.chat_model.clone(),
            embeddings_model: self.embeddings_model.clone(),
        }
    }

    fn mk_chat_api(&self) -> anyhow::Result<Box<dyn ChatCompletionAPI>> {
        match &self.chat_platform {
            Some(ChatCompletionPlatform::OpenAI) => {
                Ok(Box::new(self.mk_open_ai_api()))
            },
            None => anyhow::bail!("No chat provider specified!"),
        }
    }

    fn mk_embeddings_api(&self) -> anyhow::Result<Box<dyn EmbeddingsAPI>> {
        let platform = match (self.embeddings_platform, self.chat_platform) {
            (Some(platform), _) => platform,
            (None, Some(ChatCompletionPlatform::OpenAI)) => {
                EmbeddingsPlatform::OpenAI
            },
            (None, None) => {
                anyhow::bail!("No embeddings or chat platform specified!");
            },
        };

        match platform {
            EmbeddingsPlatform::OpenAI => Ok(Box::new(self.mk_open_ai_api())),
        }
    }
}
