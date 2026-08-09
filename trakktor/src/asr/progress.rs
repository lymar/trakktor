//! Shared progress reporting for work measured in seconds of audio.
//!
//! Every ASR engine drives the same live stderr line while it transcribes —
//! the audio position, percentage, elapsed wall time, and a rough estimate of
//! the **time remaining** — so a long run gives the same predictable heads-up
//! regardless of engine. This is the one implementation; each engine's run
//! adapts its own progress callback into [`live_reporter`] (mapping its
//! progress type to `processed_seconds` and an optional `total_seconds`) and
//! closes the line with [`finish_line`].
//!
//! The time-remaining estimate needs a known total, so it shows whenever the
//! audio length is known (the common file case) and is omitted only when the
//! source has no length (for example an `ffmpeg` pipe). Reporting is
//! throttled and, off a terminal, deduplicated so it never floods stderr.
//!
//! Speech enhancement measures its work the same way — seconds of audio out of
//! seconds of audio — so it draws the same line through
//! [`live_reporter_for`], with its own verb.

use std::{
    io::{IsTerminal, Write},
    time::{Duration, Instant},
};

/// The spinner shown at the head of the live line on a terminal.
const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Builds the shared progress reporter. The returned closure takes the audio
/// seconds processed and the total when known; call it from each engine's
/// progress callback.
///
/// On a terminal it rewrites one spinner line in place; off a terminal it
/// prints a new line only when the shown state advances. Either way updates
/// are throttled to at most one every 250 ms. When the total is known the line
/// carries the percentage and a `~mm:ss left` estimate; when it is not, just
/// the position and elapsed time.
pub(crate) fn live_reporter(started: Instant) -> impl FnMut(f64, Option<f64>) {
    live_reporter_for("transcribing", started)
}

/// The same line under another verb, for work that is not transcription but is
/// measured the same way.
pub(crate) fn live_reporter_for(
    verb: &'static str,
    started: Instant,
) -> impl FnMut(f64, Option<f64>) {
    let interactive = std::io::stderr().is_terminal();
    let mut frame: usize = 0;
    let mut last_render: Option<Instant> = None;
    let mut last_shown: f64 = 0.0;
    // Off-terminal dedup key: the percentage when the total is known, else the
    // whole audio-second, so a new line is printed only on real advance.
    let mut last_key: u64 = u64::MAX;
    // The time-remaining estimate is recomputed only when the position
    // advances (holding the last value between advances keeps it from creeping
    // upward while progress stalls); a placeholder shows until the first one.
    let mut eta_pos: f64 = 0.0;
    let mut eta: Option<String> = None;

    move |processed: f64, total: Option<f64>| {
        let now = Instant::now();
        let too_soon = last_render.is_some_and(|last| {
            now.duration_since(last) < Duration::from_millis(250)
        });
        if too_soon {
            return;
        }
        // Clamp to the furthest position shown so a mid-window reshuffle can't
        // tick the position backward.
        let position = processed.max(last_shown);
        let elapsed_s = started.elapsed().as_secs_f64();

        let (line, key) = match total {
            Some(total) if total > 0.0 => {
                let percent =
                    (position / total * 100.0).round().min(100.0) as u64;
                if position > eta_pos && position < total {
                    eta_pos = position;
                    eta =
                        Some(clock(elapsed_s * (total - position) / position));
                }
                let left = eta.as_deref().unwrap_or("--:--");
                (
                    format!(
                        "{verb} {} / {} ({percent}%) · {} elapsed · ~{left} \
                         left",
                        clock(position),
                        clock(total),
                        clock(elapsed_s),
                    ),
                    percent,
                )
            },
            _ => (
                format!(
                    "{verb} {} · {} elapsed",
                    clock(position),
                    clock(elapsed_s),
                ),
                position as u64,
            ),
        };

        // Off a terminal, only an advancing line is worth printing.
        if !interactive && key == last_key {
            return;
        }
        last_render = Some(now);
        last_shown = position;
        last_key = key;

        if interactive {
            let spin = SPINNER[frame % SPINNER.len()];
            frame = frame.wrapping_add(1);
            // `\x1b[K` clears the previous, possibly longer, line.
            eprint!("\r{spin} {line}\x1b[K");
            let _ = std::io::stderr().flush();
        } else {
            eprintln!("{line}");
        }
    }
}

/// Closes the live line with a final 100% and the total wall time. Call once
/// the run has finished, on a terminal (`\x1b[K` clears the leftover of the
/// longer live line). A no-op off a terminal, where the live line was already
/// newline-terminated.
pub(crate) fn finish_line(started: Instant, duration: f64) {
    finish_line_for("transcribing", started, duration);
}

/// The same close, under the verb [`live_reporter_for`] was given.
pub(crate) fn finish_line_for(verb: &str, started: Instant, duration: f64) {
    if !std::io::stderr().is_terminal() {
        return;
    }
    let total = clock(duration);
    let elapsed = clock(started.elapsed().as_secs_f64());
    eprintln!("\r✓ {verb} {total} / {total} (100%) · {elapsed} elapsed\x1b[K");
}

/// Formats a number of seconds as `mm:ss`, or `h:mm:ss` past an hour.
pub(crate) fn clock(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    let (hours, minutes, secs) =
        (total / 3600, (total % 3600) / 60, total % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes:02}:{secs:02}")
    }
}

/// A model-download progress reporter: percentages on stderr when it is a
/// terminal, one line per file otherwise. Shared by every engine that
/// downloads weights on first use.
pub(crate) fn download_progress() -> impl FnMut(&str, u64, Option<u64>) {
    let interactive = std::io::stderr().is_terminal();
    let mut announced: Option<String> = None;
    let mut last_percent: u64 = u64::MAX;
    move |file: &str, done: u64, total: Option<u64>| {
        if announced.as_deref() != Some(file) {
            announced = Some(file.to_string());
            last_percent = u64::MAX;
            if !interactive {
                eprintln!("downloading {file}...");
            }
        }
        if !interactive {
            return;
        }
        match total {
            Some(total) if total > 0 => {
                let percent = done * 100 / total;
                if percent != last_percent {
                    last_percent = percent;
                    eprint!("\rdownloading {file}: {percent}%");
                    if percent == 100 {
                        eprintln!();
                    }
                    let _ = std::io::stderr().flush();
                }
            },
            _ => {
                eprint!("\rdownloading {file}: {} MiB", done >> 20);
                let _ = std::io::stderr().flush();
            },
        }
    }
}
