use clap::Parser;
use tracing_subscriber::{
    self,
    filter::{filter_fn, LevelFilter},
    prelude::*,
    Layer,
};
use trakktor_cli::Cli;

mod trakktor_cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let log_level = if cli.dev {
        LevelFilter::TRACE
    } else {
        match cli.verbosity {
            0 => LevelFilter::WARN,
            1 => LevelFilter::INFO,
            2 => LevelFilter::DEBUG,
            _ => LevelFilter::TRACE,
        }
    };

    let layer = tracing_subscriber::fmt::layer()
        .with_level(true)
        .with_target(false)
        .without_time()
        .with_filter(filter_fn(move |metadata| {
            if metadata.target().starts_with("trakktor") {
                metadata.level() <= &log_level
            } else {
                metadata.level() <= &LevelFilter::WARN
            }
        }));
    tracing_subscriber::registry().with(layer).init();

    cli.run().await?;

    Ok(())
}
