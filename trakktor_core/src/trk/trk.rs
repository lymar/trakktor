use std::sync::Arc;

use tokio::select;

use crate::{logger::init_logger, trk::config};

pub(crate) struct Trk {
    pub cfg: config::TrkConfig,
    pub preview_lock: tokio::sync::Mutex<()>,
}

impl Trk {
    pub async fn run(cfg: config::TrkConfig) -> anyhow::Result<()> {
        init_logger()?;

        let inst = Arc::new(Self {
            cfg,
            preview_lock: tokio::sync::Mutex::new(()),
        });

        let exit_signal = tokio::signal::ctrl_c();

        let js_engine = crate::engine::run(inst.clone());

        select! {
            _ = exit_signal => {
                log::debug!("Received exit signal, shutting down Trk...");
            }
            res = js_engine => {
                log::debug!(res:?; "JS engine finished execution.");
                res?;
            }
        }

        Ok(())
    }

    pub fn js_log(&self, level: log::Level, msg: &str, indent: usize) {
        log::log!(target: "js", level, "{msg:>indent$}");
    }
}
