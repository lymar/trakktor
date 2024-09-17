use trakktor::structify_text::{run_structify_text, StructifyText};

use super::Cli;

impl Cli {
    pub async fn structify_text(
        &self,
        structify_text: &StructifyText,
    ) -> anyhow::Result<()> {
        let chat_api = self.mk_chat_api()?;
        run_structify_text(structify_text, &chat_api).await?;

        Ok(())
    }
}
