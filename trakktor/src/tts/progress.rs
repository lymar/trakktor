//! Live progress reporting for a synthesis run.
//!
//! Generation runs at a few times slower than real time, so a page of text is
//! minutes of work and silence is not an option. The line shows which
//! paragraph is being spoken, how much audio exists so far, and how much
//! longer it will take.
//!
//! The estimate needs an expected total, which no one knows in advance — the
//! model decides how long a sentence takes. It is inferred instead: the
//! finished pieces give seconds of audio per unit of text, that rate carried
//! over the text still to come gives the audio left, and the observed
//! generation speed turns it into wall time. Until the first piece is done, a
//! measured constant stands in.
//!
//! Speed is measured from the first frame rather than from the start of the
//! run: loading the weights takes seconds, and charging that to the first
//! frame would put a wildly pessimistic number on the line just as the user
//! starts reading it.
//!
//! **The line is drawn by a ticker thread, not by the progress callback.**
//! Frames do not arrive at a steady pace: the codec decodes a finished piece
//! in one go, and on a GPU backend that compiles and autotunes its kernels the
//! first pass over a new shape can take several seconds. A line drawn only on
//! frames would freeze during those, which reads as a hang — so the ticker
//! keeps drawing and says outright how long it has been since the last frame.

use std::{
    io::{IsTerminal, Write},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use trakktor_core::tts::qwen3_tts::{SpeechProgress, Stage, Synthesis};

use crate::asr::progress::clock;

/// The spinner shown at the head of the live line on a terminal.
const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// How often the line is redrawn on a terminal.
const TICK: Duration = Duration::from_millis(250);

/// How long generation has to be quiet before the line says so. Below this a
/// gap is just the pace of the model; above it, the user deserves to know that
/// nothing has arrived.
const STALL_NOTICE: Duration = Duration::from_secs(3);

/// Seconds of speech one unit of text (an engine token) is worth, used only
/// until a piece has actually finished. Measured on Russian prose at 0.17–0.20;
/// the estimate corrects itself after the first paragraph.
const SECONDS_PER_UNIT: f64 = 0.19;

/// Audio that has to be generated before the speed is worth extrapolating
/// from. Below it the ratio is dominated by whatever the first frames paid
/// for.
const SPEED_WARMUP_SECONDS: f64 = 1.0;

/// What the ticker draws from: the newest progress and when it last moved.
struct Shared {
    latest: Option<SpeechProgress>,
    /// When the audio position last advanced — the clock a stall is measured
    /// against.
    moved_at: Instant,
    /// Time and audio at the first frame, the baseline for generation speed.
    anchor: Option<(Instant, f64)>,
    /// The last piece and stage announced off a terminal, where the line is
    /// printed on change rather than redrawn.
    announced: Option<(usize, Stage)>,
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
    /// place; off a terminal a line is printed per paragraph, so logs stay
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
                    let line = render(&progress, &state, started);
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

    /// Records the newest progress. Called once per generated frame, so it
    /// only touches shared state — the drawing happens on the ticker.
    pub(crate) fn update(&self, progress: SpeechProgress) {
        let mut state = self.shared.lock().expect("progress state");
        let moved = state.latest.is_none_or(|last| progress.audio > last.audio);
        if moved {
            state.moved_at = Instant::now();
            state.anchor.get_or_insert((Instant::now(), progress.audio));
        }
        state.latest = Some(progress);

        // Off a terminal there is no ticker; announce each new paragraph and
        // each change of stage, so a log still shows what the run is doing.
        let step = (progress.chunk, progress.stage);
        if !self.interactive && state.announced != Some(step) {
            state.announced = Some(step);
            let line = render(&progress, &state, self.started);
            eprintln!("{line}");
        }
    }

    /// Stops the ticker and closes the line with what was produced.
    pub(crate) fn finish(mut self, synthesis: &Synthesis) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(ticker) = self.ticker.take() {
            let _ = ticker.join();
        }
        if !self.interactive {
            return;
        }
        eprintln!(
            "\r✓ synthesized {} paragraph{} · {} audio · {} elapsed\x1b[K",
            synthesis.chunks,
            if synthesis.chunks == 1 { "" } else { "s" },
            clock(synthesis.speech.duration()),
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

/// The line itself: position, elapsed time, estimate — and, when frames have
/// stopped arriving, how long ago the last one did.
fn render(
    progress: &SpeechProgress,
    state: &Shared,
    started: Instant,
) -> String {
    let speed = state.anchor.map(|(since, audio_then)| Speed {
        spent: since.elapsed().as_secs_f64(),
        generated: progress.audio - audio_then,
    });
    let quiet = state.moved_at.elapsed();
    let (verb, tail) = match progress.stage {
        // The codec runs as one call per piece, so the only honest thing to
        // show is that it is running and for how long.
        Stage::Decoding => {
            ("decoding", format!(" · {:.0}s so far", quiet.as_secs_f64()))
        },
        // Generation reports every frame; a gap between them is worth naming
        // rather than leaving the line frozen (a GPU backend compiling a
        // kernel on its first pass can take seconds).
        Stage::Generating if quiet >= STALL_NOTICE => (
            "synthesizing",
            format!(" · no new audio for {:.0}s", quiet.as_secs_f64()),
        ),
        Stage::Generating => (
            "synthesizing",
            format!(
                " · ~{} left",
                speed
                    .and_then(|speed| remaining(progress, speed))
                    .map_or_else(|| "--:--".to_owned(), clock)
            ),
        ),
    };
    format!(
        "{verb} {}/{} · {} audio · {} elapsed{tail}",
        progress.chunk,
        progress.chunks,
        clock(progress.audio),
        clock(started.elapsed().as_secs_f64()),
    )
}

/// How fast generation is running: wall seconds spent producing this much
/// audio, both measured from the first frame.
#[derive(Clone, Copy)]
struct Speed {
    spent: f64,
    generated: f64,
}

/// Wall seconds still to come, or `None` before there is anything to
/// extrapolate from.
fn remaining(progress: &SpeechProgress, speed: Speed) -> Option<f64> {
    if speed.generated < SPEED_WARMUP_SECONDS || progress.total_cost == 0 {
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
    Some(audio_left * speed.spent / speed.generated)
}
