mod cli;

pub use cli::Cli;
use cli::Commands;

impl Cli {
    pub async fn run(self) -> anyhow::Result<()> {
        match &self.command {
            Commands::AwsBatch(aws_batch) => {
                self.run_aws_batch(aws_batch).await?;
            },
        }

        Ok(())
    }
}
