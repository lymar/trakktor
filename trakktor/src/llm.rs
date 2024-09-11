use std::borrow::Cow;

use bon::builder;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

pub mod open_ai;

#[derive(ValueEnum, Clone, Copy, Debug, Deserialize)]
pub enum Provider {
    #[serde(rename = "open-ai")]
    OpenAI,
    // #[serde(rename = "aws-bedrock")]
    // AWSBedrock,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Message<'a> {
    pub role: Role,
    pub content: Cow<'a, str>,
}

#[builder]
#[derive(Debug)]
pub struct ChatCompletions<'a> {
    pub model_overwrite: Option<&'a str>,
    pub messages: &'a [Message<'a>],
    pub response_format: Option<&'a serde_json::Value>,
}

impl<'a> ChatCompletions<'a> {
    pub async fn run_with(
        self,
        provider: &impl ChatCompletionsProvider,
    ) -> anyhow::Result<Message<'static>> {
        provider.run_chat(self).await
    }
}

#[async_trait::async_trait]
pub trait ChatCompletionsProvider {
    async fn run_chat(
        &self,
        args: ChatCompletions<'_>,
    ) -> anyhow::Result<Message<'static>>;

    fn config_hash(&self) -> String;
}
