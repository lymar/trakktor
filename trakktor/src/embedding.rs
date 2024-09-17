use bon::builder;
use clap::ValueEnum;
use serde::Deserialize;

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
        api: &impl EmbeddingsAPI,
    ) -> anyhow::Result<Vec<f64>> {
        api.get_embedding(self).await
    }
}

#[async_trait::async_trait]
pub trait EmbeddingsAPI {
    async fn get_embedding(
        &self,
        args: EmbeddingsArgs<'_>,
    ) -> anyhow::Result<Vec<f64>>;

    fn config_hash(&self) -> String;
}
