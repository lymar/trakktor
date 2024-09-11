mod cli;

use std::sync::Arc;

pub use cli::Cli;
use cli::Commands;
use trakktor::{
    ai_chat::{run_ai_chat, AllChatProviders},
    llm::{open_ai::OpenAIChatProvider, ChatCompletionsProvider, Provider},
};

impl Cli {
    pub async fn run(self) -> anyhow::Result<()> {
        match &self.command {
            Commands::AwsBatch(aws_batch) => {
                self.run_aws_batch(aws_batch).await?;
            },
            Commands::AIChat(ai_chat) => {
                let all_providers = AllChatProviders {
                    open_ai: self.mk_open_ai_chat_provider(),
                };
                run_ai_chat(
                    ai_chat,
                    &self.chat_provider,
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

    fn mk_open_ai_chat_provider(&self) -> OpenAIChatProvider {
        OpenAIChatProvider {
            api_key: self.openai_api_key.clone(),
            server_url: self.openai_server_url.clone().map(|url| Arc::new(url)),
            model: self.chat_model.clone(),
        }
    }

    fn mk_chat_provider(
        &self,
    ) -> anyhow::Result<Box<dyn ChatCompletionsProvider>> {
        match &self.chat_provider {
            Some(Provider::OpenAI) => {
                Ok(Box::new(self.mk_open_ai_chat_provider()))
            },
            None => anyhow::bail!("No chat provider specified!"),
        }
    }
}
