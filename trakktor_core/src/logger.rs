use std::{
    fmt::Write,
    sync::atomic::{AtomicBool, Ordering},
};

use log::{Level, LevelFilter, Metadata, Record};

struct TrkLogger {
    default_level: LevelFilter,
    module_levels: Vec<(String, LevelFilter)>,
}

impl log::Log for TrkLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        !LOG_MUTED.load(Ordering::Relaxed) &&
            &metadata.level().to_level_filter() <=
                self.module_levels
                    .iter()
                    .find(|(name, _level)| metadata.target().starts_with(name))
                    .map(|(_name, level)| level)
                    .unwrap_or(&self.default_level)
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let mut buff = String::with_capacity(64);
            match record.level() {
                Level::Error => buff.push_str("[ERR "),
                Level::Warn => buff.push_str("[WAR "),
                Level::Info => buff.push_str("[INF "),
                Level::Debug => buff.push_str("[DEB "),
                Level::Trace => buff.push_str("[TRA "),
            }

            let target = if !record.target().is_empty() {
                record.target()
            } else {
                record.module_path().unwrap_or_default()
            };

            write!(&mut buff, "{target}] {args}", args = record.args())
                .expect("Writing to string failed");

            let kvs = record.key_values();
            if kvs.count() > 0 {
                buff.push_str(" {");
                kvs.visit(&mut KVPrinter {
                    is_first: true,
                    buff: &mut buff,
                })
                .expect("Visiting key-values and writing to string failed");
                buff.push_str("}");
            }
            eprintln!("{}", buff);
        }
    }

    fn flush(&self) {}
}

struct KVPrinter<'a> {
    is_first: bool,
    buff: &'a mut String,
}

impl<'kvs, 'bs> log::kv::VisitSource<'kvs> for KVPrinter<'bs> {
    fn visit_pair(
        &mut self,
        key: log::kv::Key<'kvs>,
        value: log::kv::Value<'kvs>,
    ) -> Result<(), log::kv::Error> {
        if !self.is_first {
            write!(self.buff, ", ").expect("Writing to string failed");
        } else {
            self.is_first = false;
        }
        write!(self.buff, "{key}: {value}").expect("Writing to string failed");
        Ok(())
    }
}

impl TrkLogger {
    pub fn max_level(&self) -> LevelFilter {
        let max_level = self
            .module_levels
            .iter()
            .map(|(_name, level)| level)
            .copied()
            .max();
        max_level
            .map(|lvl| lvl.max(self.default_level))
            .unwrap_or(self.default_level)
    }
}

pub fn init_logger() -> anyhow::Result<()> {
    let mut module_levels: Vec<(String, LevelFilter)> = Vec::new();
    module_levels.sort_by_key(|(name, _level)| name.len().wrapping_neg());
    let logger = TrkLogger {
        default_level: LevelFilter::Trace,
        module_levels,
    };
    log::set_max_level(logger.max_level());
    log::set_boxed_logger(Box::new(logger))?;
    Ok(())
}

static LOG_MUTED: AtomicBool = AtomicBool::new(false);

pub struct LogMuteGuard {
    prev: bool,
}

impl Drop for LogMuteGuard {
    fn drop(&mut self) { LOG_MUTED.store(self.prev, Ordering::Relaxed); }
}

pub fn mute_log() -> LogMuteGuard {
    let prev = LOG_MUTED.swap(true, Ordering::Relaxed);
    LogMuteGuard { prev }
}
