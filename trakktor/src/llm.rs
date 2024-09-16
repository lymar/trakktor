use std::borrow::Cow;

use bon::builder;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::config_hash::ConfigHash;

#[derive(ValueEnum, Clone, Copy, Debug, Deserialize)]
pub enum ChatCompletionPlatform {
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
pub struct ChatCompletionsArgs<'a> {
    pub model_overwrite: Option<&'a str>,
    pub messages: &'a [Message<'a>],
    pub response_format: Option<&'a serde_json::Value>,
}

impl<'a> ChatCompletionsArgs<'a> {
    pub async fn run_with(
        self,
        api: &impl ChatCompletionChatAPI,
    ) -> anyhow::Result<Message<'static>> {
        api.run_chat(self).await
    }
}

#[async_trait::async_trait]
pub trait ChatCompletionChatAPI {
    async fn run_chat(
        &self,
        args: ChatCompletionsArgs<'_>,
    ) -> anyhow::Result<Message<'static>>;
}

pub trait ChatCompletionAPI: ChatCompletionChatAPI + ConfigHash {}
