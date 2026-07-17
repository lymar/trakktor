//! Optional stderr tracing for diagnosing transcription timing.
//!
//! Off by default. Set the `TRAKKTOR_ASR_TRACE` environment variable to a
//! non-empty value to emit per-window, per-attempt, and per-step timing to
//! stderr while a transcription runs. Purely diagnostic: it never changes the
//! transcription result, and with the variable unset it costs nothing.

use std::{sync::OnceLock, time::Instant};

/// Whether tracing is enabled (read once from the environment).
pub(crate) fn on() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("TRAKKTOR_ASR_TRACE").is_some_and(|v| !v.is_empty())
    })
}

/// A process-wide monotonic origin, so every trace line carries a wall-clock
/// offset from the first traced event.
fn origin() -> Instant {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

/// Emits one trace line to stderr when tracing is enabled; a no-op otherwise.
pub(crate) fn emit(args: std::fmt::Arguments<'_>) {
    if !on() {
        return;
    }
    let elapsed = origin().elapsed().as_secs_f64();
    eprintln!("[asr-trace {elapsed:8.2}s] {args}");
}
