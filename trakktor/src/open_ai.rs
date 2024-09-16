use std::sync::Arc;

use anyhow::{bail, Context};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use url::Url;

use crate::{
    config_hash::ConfigHash,
    embedding::{EmbeddingsAPI, EmbeddingsArgs, EmbeddingsGetAPI},
    llm::{
        ChatCompletionAPI, ChatCompletionChatAPI, ChatCompletionsArgs, Message,
        Role,
    },
};

pub const OPENAI_DEFAULT_SERVER_URL: &str = "https://api.openai.com";

pub const OPENAI_CHAT_DEFAULT_MODEL: &str = "gpt-4o";
const CHAT_ENDPOINT: &str = "v1/chat/completions";

pub const OPENAI_EMBEDDING_DEFAULT_MODEL: &str = "text-embedding-3-large";
const EMBEDDING_ENDPOINT: &str = "v1/embeddings";

#[derive(Debug, Clone)]
pub struct OpenAiAPI {
    pub api_key: Option<Arc<str>>,
    pub server_url: Option<Arc<Url>>,
    pub chat_model: Option<Arc<str>>,
    pub embeddings_model: Option<Arc<str>>,
}

impl OpenAiAPI {
    #[tracing::instrument(level = "debug", skip(self, req))]
    async fn make_request<I, O>(
        &self,
        req: &I,
        endpoint: &str,
    ) -> anyhow::Result<O>
    where
        I: Serialize + ?Sized + std::fmt::Debug,
        O: DeserializeOwned + std::fmt::Debug,
    {
        let client = reqwest::Client::new();
        let endpoint = if let Some(server_url) = &self.server_url {
            server_url.join(endpoint)?
        } else {
            Url::parse(OPENAI_DEFAULT_SERVER_URL)?.join(endpoint)?
        };

        tracing::debug!(
            endpoint = endpoint.to_string(),
            ?req,
            "Sending request to API"
        );
        let mut req_builder = client.post(endpoint).json(&req);
        if let Some(api_key) = &self.api_key {
            req_builder =
                req_builder.header("Authorization", format!("Bearer {api_key}"))
        }
        let res = req_builder.send().await?;

        let code = res.status();
        tracing::debug!(status = ?code, "API call completed");
        let res = res.text().await?;
        tracing::debug!(response = ?res, "API response received");

        if !code.is_success() {
            bail!("Failed to call API!\nCode: {code}\nResponse: {res}");
        }

        Ok(serde_json::from_str(&res).with_context(|| {
            format!("Failed to parse response from API:\n{res}")
        })?)
    }
}

#[async_trait::async_trait]
impl ChatCompletionChatAPI for OpenAiAPI {
    #[tracing::instrument(level = "debug", skip_all)]
    async fn run_chat(
        &self,
        args: ChatCompletionsArgs<'_>,
    ) -> anyhow::Result<Message<'static>> {
        let res: OpenAiChatCompletionsResponse = self
            .make_request(
                &OpenAiChatCompletions {
                    model: args
                        .model_overwrite
                        .or(self.chat_model.as_deref())
                        .unwrap_or_else(|| OPENAI_CHAT_DEFAULT_MODEL),
                    messages: args.messages,
                    response_format: args.response_format,
                },
                CHAT_ENDPOINT,
            )
            .await?;

        let choice =
            res.choices.into_iter().next().ok_or_else(|| {
                anyhow::anyhow!("Empty response from Chat API")
            })?;
        if !matches!(&choice.message.role, Role::Assistant) {
            bail!(
                "Unexpected role in response from API: {:?}",
                choice.message.role
            );
        }

        tracing::info!(usage = ?res.usage, model = res.model,
            finish_reason = choice.finish_reason,
            "API call completed successfully");

        Ok(Message {
            role: choice.message.role,
            content: choice.message.content,
        })
    }
}

impl ConfigHash for OpenAiAPI {
    fn config_hash(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(format!("{:?}", self).as_bytes());
        URL_SAFE_NO_PAD.encode(&hasher.finalize().as_bytes())
    }
}

impl ChatCompletionAPI for OpenAiAPI {}

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
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    pub total_tokens: u64,
}

#[async_trait::async_trait]
impl EmbeddingsGetAPI for OpenAiAPI {
    #[tracing::instrument(level = "debug", skip_all)]
    async fn get_embedding(
        &self,
        args: EmbeddingsArgs<'_>,
    ) -> anyhow::Result<Vec<f64>> {
        let res: OpenAiEmbeddingsResponse = self
            .make_request(
                &OpenAiEmbeddings {
                    model: args
                        .model_overwrite
                        .or(self.embeddings_model.as_deref())
                        .unwrap_or_else(|| OPENAI_EMBEDDING_DEFAULT_MODEL),
                    input: args.input,
                },
                EMBEDDING_ENDPOINT,
            )
            .await?;

        Ok(res
            .data
            .into_iter()
            .next()
            .ok_or_else(|| {
                anyhow::anyhow!("Empty response from Embeddings API")
            })?
            .embedding)
    }
}

impl EmbeddingsAPI for OpenAiAPI {}

#[derive(Debug, Serialize)]
pub struct OpenAiEmbeddings<'a> {
    pub input: &'a str,
    pub model: &'a str,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiEmbeddingsResponse {
    pub data: Vec<OpenAiEmbeddingObject>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiEmbeddingObject {
    pub embedding: Vec<f64>,
}
