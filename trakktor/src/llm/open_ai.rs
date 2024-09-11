use std::sync::Arc;

use anyhow::{bail, Context};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::{Deserialize, Serialize};
use url::Url;

use super::{ChatCompletionsProvider, Message};
use crate::llm::Role;

pub const OPENAI_CHAT_DEFAULT_MODEL: &str = "gpt-4o";
pub const OPENAI_DEFAULT_SERVER_URL: &str = "https://api.openai.com";
const CHAT_ENDPOINT: &str = "v1/chat/completions";

#[derive(Debug, Clone)]
pub struct OpenAIChatProvider {
    pub api_key: Option<Arc<str>>,
    pub server_url: Option<Arc<Url>>,
    pub model: Option<Arc<str>>,
}

#[async_trait::async_trait]
impl ChatCompletionsProvider for OpenAIChatProvider {
    #[tracing::instrument(level = "debug", skip_all)]
    async fn run_chat(
        &self,
        args: super::ChatCompletions<'_>,
    ) -> anyhow::Result<Message<'static>> {
        let req = OpenAiChatCompletions {
            model: args
                .model_overwrite
                .or(self.model.as_deref())
                .unwrap_or_else(|| OPENAI_CHAT_DEFAULT_MODEL),
            messages: args.messages,
            response_format: args.response_format,
        };

        tracing::debug!("Calling OpenAI API...");
        let client = reqwest::Client::new();
        let endpoint = if let Some(server_url) = &self.server_url {
            server_url.join(CHAT_ENDPOINT)?
        } else {
            Url::parse(OPENAI_DEFAULT_SERVER_URL)?.join(CHAT_ENDPOINT)?
        };

        let mut req_builder = client.post(endpoint).json(&req);
        if let Some(api_key) = &self.api_key {
            req_builder =
                req_builder.header("Authorization", format!("Bearer {api_key}"))
        }
        let res = req_builder.send().await?;

        let code = res.status();
        tracing::debug!(status = ?code, "OpenAI API call completed");
        let res = res.text().await?;
        tracing::debug!(response = ?res, "OpenAI API response received");

        if !code.is_success() {
            bail!("Failed to call OpenAI API!\nCode: {code}\nResponse: {res}");
        }

        let res: OpenAiChatCompletionsResponse = serde_json::from_str(&res)
            .with_context(|| {
                format!("Failed to parse response from OpenAI API:\n{res}")
            })?;

        let choice =
            res.choices.into_iter().next().ok_or_else(|| {
                anyhow::anyhow!("Empty response from OpenAI API")
            })?;
        if !matches!(&choice.message.role, Role::Assistant) {
            bail!(
                "Unexpected role in response from OpenAI API: {:?}",
                choice.message.role
            );
        }

        tracing::info!(usage = ?res.usage, model = res.model,
            finish_reason = choice.finish_reason,
            "OpenAI API call completed successfully");

        Ok(Message {
            role: choice.message.role,
            content: choice.message.content,
        })
    }

    fn config_hash(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        if let Some(api_key) = &self.api_key {
            hasher.update(api_key.as_bytes());
        }
        hasher.update(b":");
        if let Some(server_url) = &self.server_url {
            hasher.update(server_url.as_str().as_bytes());
        }
        hasher.update(b":");
        if let Some(model) = &self.model {
            hasher.update(model.as_bytes());
        }
        URL_SAFE_NO_PAD.encode(&hasher.finalize().as_bytes())
    }
}

#[derive(Debug, Serialize)]
pub struct OpenAiChatCompletions<'a> {
    pub model: &'a str,
    pub messages: &'a [Message<'a>],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<&'a serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiChatCompletionsResponse {
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: Message<'static>,
    pub finish_reason: String,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}
