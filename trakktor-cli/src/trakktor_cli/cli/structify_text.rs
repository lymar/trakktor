use trakktor::structify_text::{run_structify_text, StructifyText};

use super::Cli;

impl Cli {
    pub async fn structify_text(
        &self,
        structify_text: &StructifyText,
    ) -> anyhow::Result<()> {
        let chat_provider = self.mk_chat_provider()?;
        run_structify_text(structify_text, &chat_provider).await?;

        Ok(())
    }
}
