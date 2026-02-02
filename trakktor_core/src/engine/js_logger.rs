use std::sync::{Arc, Weak};

use boa_engine::{Context, Finalize, JsResult, Trace};
use boa_runtime::{ConsoleState, Logger};

use crate::trk::Trk;

#[derive(Debug, Trace, Finalize)]
pub(crate) struct JsLogger {
    #[unsafe_ignore_trace]
    trk: Weak<Trk>,
}

impl JsLogger {
    pub fn new(trk: &Arc<Trk>) -> Self {
        Self {
            trk: Arc::downgrade(&trk),
        }
    }

    fn log(&self, level: log::Level, msg: &str, indent: usize) -> JsResult<()> {
        if let Some(trk) = self.trk.upgrade() {
            trk.js_log(level, msg, indent);
        }
        Ok(())
    }
}

impl Logger for JsLogger {
    fn log(
        &self,
        msg: String,
        state: &ConsoleState,
        _context: &mut Context,
    ) -> JsResult<()> {
        self.log(log::Level::Info, &msg, state.indent())
    }

    fn info(
        &self,
        msg: String,
        state: &ConsoleState,
        _context: &mut Context,
    ) -> JsResult<()> {
        self.log(log::Level::Info, &msg, state.indent())
    }

    fn warn(
        &self,
        msg: String,
        state: &ConsoleState,
        _context: &mut Context,
    ) -> JsResult<()> {
        self.log(log::Level::Warn, &msg, state.indent())
    }

    fn error(
        &self,
        msg: String,
        state: &ConsoleState,
        _context: &mut Context,
    ) -> JsResult<()> {
        self.log(log::Level::Error, &msg, state.indent())
    }
}
