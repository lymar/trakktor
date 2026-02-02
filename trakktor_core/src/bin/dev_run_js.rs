use std::env::args;

use trakktor_core::trk::{self, config::TrkConfig};

// export OPENAI_API_KEY=

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path = args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("No input file path provided"))?;
    std::env::var("OPENAI_API_KEY").map_err(|_| {
        anyhow::anyhow!("OPENAI_API_KEY environment variable not set")
    })?;

    let ecfg = TrkConfig::builder().js_file(&path).build();
    trk::run_trk(ecfg).await?;

    Ok(())
}
