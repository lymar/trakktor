//! Live progress reporting for a synthesis run.
//!
//! Synthesis runs at a few times slower than real time, so a page of text is
//! minutes of work and silence is not an option. The line shows which piece is
//! being spoken, how much audio exists so far, and how much longer it will
//! take.
//!
//! **One line, two engines.** The ticker, the stall notice and the off-terminal
//! announcements are shared; what differs is the measure of progress each
//! engine can honestly report, so only the text of the line is per-engine.
//! Qwen3-TTS knows how much audio it has produced but not how much is left, and
//! has to infer the total from the rate so far. ESpeech knows the length of a
//! piece before it starts and takes a fixed number of steps, so its fraction
//! done is a fact rather than an extrapolation.
//!
//! Speed is measured from the first advance rather than from the start of the
//! run: loading the weights takes seconds, and charging that to the first step
//! would put a wildly pessimistic number on the line just as the user starts
//! reading it.
//!
//! **The line is drawn by a ticker thread, not by the progress callback.**
//! Progress does not arrive at a steady pace: a vocoder or codec runs as one
//! long call, and on a GPU backend that compiles and autotunes its kernels the
//! first pass over a new shape can take several seconds. A line drawn only on
//! callbacks would freeze during those, which reads as a hang — so the ticker
//! keeps drawing and says outright how long it has been since anything moved.

use std::{
    io::{IsTerminal, Write},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use trakktor_core::tts::{espeech, qwen3_tts, silero};

use crate::asr::progress::clock;

/// The spinner shown at the head of the live line on a terminal.
const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// How often the line is redrawn on a terminal.
const TICK: Duration = Duration::from_millis(250);

/// How long a run has to be quiet before the line says so. Below this a gap is
/// just the pace of the model; above it, the user deserves to know that nothing
/// has arrived.
const STALL_NOTICE: Duration = Duration::from_secs(3);

/// Seconds of speech one unit of text is worth, used only until a piece has
/// actually finished. Measured on Russian prose at 0.17–0.20 seconds per engine
/// token; the estimate corrects itself after the first paragraph.
const SECONDS_PER_UNIT: f64 = 0.19;

/// Progress that has to be made before the speed is worth extrapolating from.
/// Below it the ratio is dominated by whatever the first work paid for.
const WARMUP: f64 = 1.0;

/// What one engine reported, in its own terms.
#[derive(Clone, Copy)]
pub(crate) enum Progress {
    /// A frame-by-frame run: audio accumulates, the total is unknown.
    Qwen3(qwen3_tts::SpeechProgress),
    /// A fixed-step run: the length of each piece is known before it starts.
    Espeech(espeech::SpeechProgress),
    /// A one-pass run: a piece is a single call, so only the pieces move.
    Silero(silero::SpeechProgress),
}

impl From<qwen3_tts::SpeechProgress> for Progress {
    fn from(progress: qwen3_tts::SpeechProgress) -> Self {
        Progress::Qwen3(progress)
    }
}

impl From<espeech::SpeechProgress> for Progress {
    fn from(progress: espeech::SpeechProgress) -> Self {
        Progress::Espeech(progress)
    }
}

impl From<silero::SpeechProgress> for Progress {
    fn from(progress: silero::SpeechProgress) -> Self {
        Progress::Silero(progress)
    }
}

impl Progress {
    /// The measure whose advance means the run is alive, and which the speed is
    /// extrapolated from: seconds of audio for one engine, fraction of the work
    /// for the other.
    fn measure(self) -> f64 {
        match self {
            Progress::Qwen3(progress) => progress.audio,
            Progress::Espeech(progress) => Self::fraction(progress) * 100.0,
            Progress::Silero(progress) => progress.audio,
        }
    }

    /// Which piece and stage this is — the granularity an off-terminal log
    /// announces at.
    fn step(self) -> (usize, u8) {
        match self {
            Progress::Qwen3(progress) => (
                progress.chunk,
                match progress.stage {
                    qwen3_tts::Stage::Generating => 0,
                    qwen3_tts::Stage::Decoding => 1,
                },
            ),
            Progress::Espeech(progress) => (
                progress.chunk,
                match progress.stage {
                    espeech::Stage::Solving => 0,
                    espeech::Stage::Vocoding => 1,
                },
            ),
            Progress::Silero(progress) => (
                progress.chunk,
                match progress.stage {
                    silero::Stage::Synthesizing => 0,
                    silero::Stage::Vocoding => 1,
                },
            ),
        }
    }

    /// How much of the whole text is done, for a fixed-step engine.
    ///
    /// The steps of one piece are equal work, and the pieces are weighted by
    /// the text they hold; the piece in flight is charged its share of what is
    /// left, since its own cost is not reported.
    fn fraction(progress: espeech::SpeechProgress) -> f64 {
        if progress.total_cost == 0 {
            return 0.0;
        }
        let left = progress.total_cost.saturating_sub(progress.done_cost);
        let pieces_left = progress.chunks.saturating_sub(progress.chunk) + 1;
        let current = left as f64 / pieces_left.max(1) as f64;
        let within = if progress.steps == 0 {
            0.0
        } else {
            progress.step as f64 / progress.steps as f64
        };
        ((progress.done_cost as f64 + current * within) /
            progress.total_cost as f64)
            .clamp(0.0, 1.0)
    }

    /// The line itself: position, elapsed time, estimate — and, when nothing
    /// has moved for a while, how long ago it last did.
    fn render(self, state: &Shared, started: Instant) -> String {
        let quiet = state.moved_at.elapsed();
        let elapsed = started.elapsed().as_secs_f64();
        let spent =
            state.anchor.map(|(since, _)| since.elapsed().as_secs_f64());
        match self {
            Progress::Qwen3(progress) => {
                let advanced =
                    state.anchor.map(|(_, from)| progress.audio - from);
                let (verb, tail) = match progress.stage {
                    // The codec runs as one call per piece, so the only honest
                    // thing to show is that it is running and for how long.
                    qwen3_tts::Stage::Decoding => (
                        "decoding",
                        format!(" · {:.0}s so far", quiet.as_secs_f64()),
                    ),
                    qwen3_tts::Stage::Generating if quiet >= STALL_NOTICE => (
                        "synthesizing",
                        format!(
                            " · no new audio for {:.0}s",
                            quiet.as_secs_f64()
                        ),
                    ),
                    qwen3_tts::Stage::Generating => (
                        "synthesizing",
                        format!(
                            " · ~{} left",
                            spent
                                .zip(advanced)
                                .and_then(|(spent, advanced)| {
                                    remaining_by_rate(
                                        &progress, spent, advanced,
                                    )
                                })
                                .map_or_else(|| "--:--".to_owned(), clock)
                        ),
                    ),
                };
                format!(
                    "{verb} {}/{} · {} audio · {} elapsed{tail}",
                    progress.chunk,
                    progress.chunks,
                    clock(progress.audio),
                    clock(elapsed),
                )
            },
            Progress::Espeech(progress) => {
                let done = Self::fraction(progress);
                let (verb, tail) = match progress.stage {
                    espeech::Stage::Vocoding => (
                        "vocoding",
                        format!(" · {:.0}s so far", quiet.as_secs_f64()),
                    ),
                    espeech::Stage::Solving if quiet >= STALL_NOTICE => (
                        "solving",
                        format!(" · quiet for {:.0}s", quiet.as_secs_f64()),
                    ),
                    espeech::Stage::Solving => (
                        "solving",
                        format!(
                            " · ~{} left",
                            spent
                                .zip(state.anchor.map(|(_, from)| from))
                                .and_then(
                                    |(spent, from)| remaining_by_fraction(
                                        done,
                                        from / 100.0,
                                        spent
                                    )
                                )
                                .map_or_else(|| "--:--".to_owned(), clock)
                        ),
                    ),
                };
                format!(
                    "{verb} {}/{} · step {}/{} · {:.0}% · {} audio · {} \
                     elapsed{tail}",
                    progress.chunk,
                    progress.chunks,
                    progress.step,
                    progress.steps,
                    done * 100.0,
                    clock(progress.audio),
                    clock(elapsed),
                )
            },
            // One call per piece, so there is nothing to report from inside
            // one: what moves is the count of pieces and the audio they came
            // to. That is honest, and this engine is fast enough that the line
            // rarely gets a second tick anyway.
            Progress::Silero(progress) => {
                let done = if progress.total_cost == 0 {
                    0.0
                } else {
                    progress.done_cost as f64 / progress.total_cost as f64
                };
                let tail = if quiet >= STALL_NOTICE {
                    format!(" · quiet for {:.0}s", quiet.as_secs_f64())
                } else {
                    String::new()
                };
                format!(
                    "speaking {}/{} · {:.0}% · {} audio · {} elapsed{tail}",
                    progress.chunk,
                    progress.chunks,
                    done * 100.0,
                    clock(progress.audio),
                    clock(elapsed),
                )
            },
        }
    }
}

/// What the ticker draws from: the newest progress and when it last moved.
struct Shared {
    latest: Option<Progress>,
    /// When the measure last advanced — the clock a stall is measured against.
    moved_at: Instant,
    /// Time and measure at the first advance, the baseline for speed.
    anchor: Option<(Instant, f64)>,
    /// The last piece and stage announced off a terminal, where the line is
    /// printed on change rather than redrawn.
    announced: Option<(usize, u8)>,
}

/// A running progress line.
pub(crate) struct Reporter {
    shared: Arc<Mutex<Shared>>,
    running: Arc<AtomicBool>,
    ticker: Option<JoinHandle<()>>,
    interactive: bool,
    started: Instant,
}

impl Reporter {
    /// Starts reporting. On a terminal a ticker thread redraws one line in
    /// place; off a terminal a line is printed per piece, so logs stay
    /// readable.
    pub(crate) fn start(started: Instant) -> Self {
        let interactive = std::io::stderr().is_terminal();
        let shared = Arc::new(Mutex::new(Shared {
            latest: None,
            moved_at: started,
            anchor: None,
            announced: None,
        }));
        let running = Arc::new(AtomicBool::new(true));

        let ticker = interactive.then(|| {
            let shared = Arc::clone(&shared);
            let running = Arc::clone(&running);
            thread::spawn(move || {
                let mut frame = 0usize;
                while running.load(Ordering::Relaxed) {
                    thread::sleep(TICK);
                    let state = shared.lock().expect("progress state");
                    let Some(progress) = state.latest else {
                        continue;
                    };
                    let line = progress.render(&state, started);
                    drop(state);
                    let spin = SPINNER[frame % SPINNER.len()];
                    frame = frame.wrapping_add(1);
                    // `\x1b[K` clears the previous, possibly longer, line.
                    eprint!("\r{spin} {line}\x1b[K");
                    let _ = std::io::stderr().flush();
                }
            })
        });

        Self {
            shared,
            running,
            ticker,
            interactive,
            started,
        }
    }

    /// Records the newest progress. Called on every step, so it only touches
    /// shared state — the drawing happens on the ticker.
    pub(crate) fn update(&self, progress: impl Into<Progress>) {
        let progress = progress.into();
        let mut state = self.shared.lock().expect("progress state");
        let moved = state
            .latest
            .is_none_or(|last| progress.measure() > last.measure());
        if moved {
            state.moved_at = Instant::now();
            state
                .anchor
                .get_or_insert((Instant::now(), progress.measure()));
        }
        state.latest = Some(progress);

        // Off a terminal there is no ticker; announce each new piece and each
        // change of stage, so a log still shows what the run is doing.
        let step = progress.step();
        if !self.interactive && state.announced != Some(step) {
            state.announced = Some(step);
            let line = progress.render(&state, self.started);
            eprintln!("{line}");
        }
    }

    /// Stops the ticker and closes the line with what was produced.
    pub(crate) fn finish(mut self, chunks: usize, duration: f64) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(ticker) = self.ticker.take() {
            let _ = ticker.join();
        }
        if !self.interactive {
            return;
        }
        eprintln!(
            "\r✓ synthesized {chunks} piece{} · {} audio · {} elapsed\x1b[K",
            if chunks == 1 { "" } else { "s" },
            clock(duration),
            clock(self.started.elapsed().as_secs_f64()),
        );
    }
}

impl Drop for Reporter {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(ticker) = self.ticker.take() {
            let _ = ticker.join();
        }
    }
}

/// Wall seconds still to come for an engine that only knows its rate: the audio
/// it is likely to still produce, at the speed it has been producing audio.
fn remaining_by_rate(
    progress: &qwen3_tts::SpeechProgress,
    spent: f64,
    advanced: f64,
) -> Option<f64> {
    if advanced < WARMUP || progress.total_cost == 0 {
        return None;
    }
    // Seconds of speech per unit of text: measured on what is already done,
    // assumed from prose until then.
    let rate = if progress.done_cost > 0 && progress.finished_audio > 0.0 {
        progress.finished_audio / progress.done_cost as f64
    } else {
        SECONDS_PER_UNIT
    };
    let expected_audio =
        (rate * progress.total_cost as f64).max(progress.audio);
    let audio_left = expected_audio - progress.audio;
    Some(audio_left * spent / advanced)
}

/// Wall seconds still to come for an engine that knows its fraction done.
fn remaining_by_fraction(done: f64, from: f64, spent: f64) -> Option<f64> {
    let covered = done - from;
    if covered <= 0.0 || done >= 1.0 {
        return None;
    }
    Some(spent * (1.0 - done) / covered)
}
