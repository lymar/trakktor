pub mod config;
pub mod trk;

pub(crate) use trk::Trk;

pub async fn run_trk(cfg: config::TrkConfig) -> anyhow::Result<()> {
    trk::Trk::run(cfg).await
}
