use bon::builder;
use clap::ValueEnum;
use serde::Deserialize;

use crate::config_hash::ConfigHash;

#[derive(ValueEnum, Clone, Copy, Debug, Deserialize)]
pub enum EmbeddingsPlatform {
    #[serde(rename = "open-ai")]
    OpenAI,
}

#[builder]
#[derive(Debug)]
pub struct EmbeddingsArgs<'a> {
    pub model_overwrite: Option<&'a str>,
    pub input: &'a str,
}

impl<'a> EmbeddingsArgs<'a> {
    pub async fn run_with(
        self,
        api: &impl EmbeddingsGetAPI,
    ) -> anyhow::Result<Vec<f64>> {
        api.get_embedding(self).await
    }
}

#[async_trait::async_trait]
pub trait EmbeddingsGetAPI {
    async fn get_embedding(
        &self,
        args: EmbeddingsArgs<'_>,
    ) -> anyhow::Result<Vec<f64>>;
}

pub trait EmbeddingsAPI: EmbeddingsGetAPI + ConfigHash {}
