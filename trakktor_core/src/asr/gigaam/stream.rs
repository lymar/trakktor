//! Streamed, bounded-memory transcription.
//!
//! [`StreamTranscriber`] consumes 16 kHz mono PCM in blocks of any size and
//! produces the same transcription as a whole-file run, holding only a few
//! minutes of audio at a time:
//!
//! - blocks feed the (causal) voice-activity model incrementally;
//! - finalized speech intervals go to a chunk planner that runs the DP cut
//!   optimization over a rolling lookahead window: once the pending speech
//!   spans [`COMMIT_HORIZON_S`], every chunk ending at least
//!   [`COMMIT_MARGIN_S`] before the frontier is *committed* — cut choices that
//!   far behind no longer change when more speech arrives;
//! - committed chunks are transcribed immediately and their PCM is dropped, so
//!   the retained buffer stays around `horizon + VAD latency` seconds
//!   regardless of the file length.
//!
//! Commits are triggered by the speech/window timeline, never by block
//! boundaries, so feeding the same audio in different block sizes yields an
//! identical transcription.
//!
//! Audio no longer than [`LONGFORM_THRESHOLD_S`] is transcribed as a single
//! chunk at [`finish`](StreamTranscriber::finish) (the PCM of a possibly-short
//! file is never shed, so the single-chunk path always has it whole).

#[cfg(test)]
mod tests;

use super::{
    constants::{LONGFORM_THRESHOLD_S, SAMPLE_RATE},
    decode::Word,
    error::GigaamError,
    runtime::AsrModel,
    segment::{Interval, chunk_boundaries},
    tokenizer::Tokenizer,
    transcribe::{
        Segment, TranscribeOptions, TranscribeProgress, Transcription,
        transcribe_chunk,
    },
};
use crate::vad::{
    SpeechDetector, SpeechSegment, Vad, VadOptions, VadStreamState,
};

/// Once the pending (uncommitted) speech spans this many seconds, the planner
/// runs the cut optimization and commits the stable prefix.
pub const COMMIT_HORIZON_S: f64 = 180.0;

/// Only chunks ending at least this many seconds before the pending frontier
/// are committed; the tail is re-optimized when more speech arrives. Far
/// larger than the DP's transition window, so committed cuts match what a
/// whole-file optimization would have chosen.
pub const COMMIT_MARGIN_S: f64 = 120.0;

/// Extra samples kept behind the earliest needed position when shedding PCM,
/// absorbing the window quantization of the VAD grid.
const SHED_GUARD_SAMPLES: usize = 512;

/// Plans chunk boundaries over a rolling window of finalized speech
/// intervals, committing cuts once they are far enough behind the frontier to
/// be stable.
struct ChunkPlanner {
    /// Finalized speech not yet covered by a committed chunk, ordered.
    pending: Vec<Interval>,
}

impl ChunkPlanner {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Adds a finalized speech interval; returns chunks that became final.
    fn push(&mut self, interval: Interval) -> Vec<(f64, f64)> {
        self.pending.push(interval);
        let start = self.pending[0].start;
        let frontier = interval.end;
        if frontier - start < COMMIT_HORIZON_S {
            return Vec::new();
        }
        self.commit(frontier)
    }

    /// Re-evaluates without new speech, `frontier` being the fully processed
    /// position — long silences also push pending speech far enough behind to
    /// commit (and release its PCM).
    fn poll(&mut self, frontier: f64) -> Vec<(f64, f64)> {
        match self.pending.first() {
            Some(first) if frontier - first.start >= COMMIT_HORIZON_S => {
                self.commit(frontier)
            },
            _ => Vec::new(),
        }
    }

    /// Runs the DP over the pending intervals and commits every chunk ending
    /// at least [`COMMIT_MARGIN_S`] before `frontier`.
    fn commit(&mut self, frontier: f64) -> Vec<(f64, f64)> {
        let content_end = self.pending.last().map_or(0.0, |i| i.end);
        let cuts = chunk_boundaries(&self.pending, content_end);
        let ready: Vec<(f64, f64)> = cuts
            .into_iter()
            .filter(|&(_, end)| end <= frontier - COMMIT_MARGIN_S)
            .collect();
        if let Some(&(_, boundary)) = ready.last() {
            // Keep only speech past the committed boundary. A cut can fall
            // inside an interval (the DP presplits over-long continuous
            // speech), in which case its tail half stays pending.
            self.pending.retain_mut(|unit| {
                if unit.end <= boundary {
                    false
                } else {
                    if unit.start < boundary {
                        unit.start = boundary;
                    }
                    true
                }
            });
        }
        ready
    }

    /// The start of the earliest pending speech — PCM below it may still be
    /// needed for a future chunk.
    fn earliest_pending_start(&self) -> Option<f64> {
        self.pending.first().map(|i| i.start)
    }

    /// Final cuts over whatever is still pending; `total` clamps the ends.
    fn finish(&mut self, total: f64) -> Vec<(f64, f64)> {
        let cuts = chunk_boundaries(&self.pending, total);
        self.pending.clear();
        cuts
    }
}

/// A streamed transcription session. Feed PCM with
/// [`push`](Self::push), then call [`finish`](Self::finish) once.
pub struct StreamTranscriber<'m> {
    model: &'m dyn AsrModel,
    tokenizer: &'m Tokenizer,
    options: TranscribeOptions,
    /// The declared total duration, when known — progress reporting only.
    total_hint: Option<f64>,
    vad: Vad,
    vad_state: VadStreamState,
    /// `None` only transiently inside `finish`.
    detector: Option<SpeechDetector>,
    planner: ChunkPlanner,
    /// Retained PCM; `buf[0]` is absolute sample `base`.
    buf: Vec<f32>,
    base: usize,
    /// Total samples fed.
    fed: usize,
    /// VAD windows whose probabilities have been consumed.
    windows: u64,
    segments: Vec<Segment>,
    texts: Vec<String>,
}

impl<'m> StreamTranscriber<'m> {
    /// Creates a session. `total_hint` (seconds), when known, only feeds the
    /// progress callback.
    ///
    /// # Errors
    ///
    /// Returns [`GigaamError::Vad`] when the VAD model cannot be loaded.
    pub fn new(
        model: &'m dyn AsrModel,
        tokenizer: &'m Tokenizer,
        options: TranscribeOptions,
        total_hint: Option<f64>,
    ) -> Result<Self, GigaamError> {
        let vad = Vad::load().map_err(|e| GigaamError::Vad(e.to_string()))?;
        Ok(Self {
            model,
            tokenizer,
            options,
            total_hint,
            vad,
            vad_state: VadStreamState::new(),
            detector: Some(SpeechDetector::new(&VadOptions::default())),
            planner: ChunkPlanner::new(),
            buf: Vec::new(),
            base: 0,
            fed: 0,
            segments: Vec::new(),
            texts: Vec::new(),
            windows: 0,
        })
    }

    /// Feeds the next block of 16 kHz mono samples. Chunks that became final
    /// are transcribed before returning (invoking `progress` per chunk) and
    /// their PCM is released.
    ///
    /// # Errors
    ///
    /// Returns [`GigaamError::Vad`] on a VAD failure and the model errors of
    /// the transcription itself.
    pub fn push(
        &mut self,
        samples: &[f32],
        progress: &mut dyn FnMut(TranscribeProgress),
    ) -> Result<(), GigaamError> {
        self.buf.extend_from_slice(samples);
        self.fed += samples.len();
        let probs = self
            .vad
            .stream_push(&mut self.vad_state, samples)
            .map_err(|e| GigaamError::Vad(e.to_string()))?;
        self.consume_probs(&probs, progress)?;
        self.shed_pcm();
        Ok(())
    }

    /// Ends the stream: flushes the VAD, plans the remaining chunks,
    /// transcribes them, and returns the full transcription.
    ///
    /// # Errors
    ///
    /// See [`push`](Self::push).
    pub fn finish(
        mut self,
        progress: &mut dyn FnMut(TranscribeProgress),
    ) -> Result<Transcription, GigaamError> {
        let probs = self
            .vad
            .stream_finish(&mut self.vad_state)
            .map_err(|e| GigaamError::Vad(e.to_string()))?;
        self.consume_probs(&probs, progress)?;

        let total = self.fed as f64 / SAMPLE_RATE as f64;

        // Short audio: one whole-file chunk, exactly like the reference
        // `.transcribe`; the speech layout is not used. The PCM is intact —
        // shedding never runs below the long-form threshold.
        if total <= LONGFORM_THRESHOLD_S {
            debug_assert_eq!(self.base, 0);
            if self.fed > 0 {
                self.transcribe_committed((0.0, total), progress)?;
            }
        } else {
            let mut tail = Vec::new();
            if let Some(detector) = self.detector.take() {
                detector.finish(self.fed, &mut tail);
            }
            let mut chunks = Vec::new();
            for seg in tail {
                chunks.extend(self.planner.push(to_interval(seg)));
            }
            chunks.extend(self.planner.finish(total));
            for chunk in chunks {
                self.transcribe_committed(chunk, progress)?;
            }
        }

        Ok(Transcription {
            text: self.texts.join(" "),
            segments: self.segments,
            duration: total,
        })
    }

    /// Samples currently retained — the streaming memory bound (diagnostics
    /// and tests).
    #[must_use]
    pub fn buffered_samples(&self) -> usize { self.buf.len() }

    /// Routes freshly computed window probabilities through the detector and
    /// the planner, transcribing any chunks that became final.
    fn consume_probs(
        &mut self,
        probs: &[f32],
        progress: &mut dyn FnMut(TranscribeProgress),
    ) -> Result<(), GigaamError> {
        let Some(mut detector) = self.detector.take() else {
            return Ok(());
        };
        let mut emitted = Vec::new();
        for &prob in probs {
            self.windows += 1;
            emitted.clear();
            detector.push(prob, &mut emitted);
            let mut chunks = Vec::new();
            for &seg in &emitted {
                chunks.extend(self.planner.push(to_interval(seg)));
            }
            // The fully processed position: long silences commit (and later
            // release) pending speech even with no new intervals arriving.
            let frontier = self.windows as f64 * 512.0 / SAMPLE_RATE as f64;
            chunks.extend(self.planner.poll(frontier));
            for chunk in chunks {
                self.transcribe_committed(chunk, progress)?;
            }
        }
        self.detector = Some(detector);
        Ok(())
    }

    /// Transcribes one committed chunk `[start, end)` (seconds) from the
    /// retained buffer and records the segment.
    fn transcribe_committed(
        &mut self,
        (start, end): (f64, f64),
        progress: &mut dyn FnMut(TranscribeProgress),
    ) -> Result<(), GigaamError> {
        let sr = SAMPLE_RATE as f64;
        let a = (start * sr) as usize;
        let b = ((end * sr) as usize).min(self.fed);
        if b <= a {
            return Ok(());
        }
        debug_assert!(
            a >= self.base,
            "chunk at {start}s reaches below the retained buffer"
        );
        let chunk = &self.buf[a - self.base..b - self.base];
        let result =
            transcribe_chunk(self.model, self.tokenizer, chunk, &self.options)?;
        if !result.text.is_empty() {
            self.texts.push(result.text.clone());
        }
        self.segments.push(Segment {
            id: self.segments.len(),
            start,
            end,
            text: result.text,
            words: result
                .words
                .into_iter()
                .map(|w| Word {
                    text: w.text,
                    start: w.start + start,
                    end: w.end + start,
                })
                .collect(),
        });
        progress(TranscribeProgress {
            processed_seconds: end,
            total_seconds: self.total_hint,
        });
        Ok(())
    }

    /// Drops PCM that no present or future chunk can need. Nothing is shed
    /// until the audio is longer than the long-form threshold, so the
    /// single-chunk path always has the whole file.
    fn shed_pcm(&mut self) {
        if self.fed as f64 / SAMPLE_RATE as f64 <= LONGFORM_THRESHOLD_S {
            return;
        }
        let sr = SAMPLE_RATE as f64;
        // Anything before the earliest pending speech (planner), the earliest
        // possibly-still-open speech (detector), and the VAD-processed
        // frontier is settled.
        let mut needed = self.windows as f64 * 512.0 / sr;
        if let Some(start) = self.planner.earliest_pending_start() {
            needed = needed.min(start);
        }
        if let Some(start) = self
            .detector
            .as_ref()
            .and_then(SpeechDetector::earliest_pending_second)
        {
            needed = needed.min(start);
        }
        let sample =
            ((needed * sr) as usize).saturating_sub(SHED_GUARD_SAMPLES);
        if sample > self.base {
            self.buf.drain(..sample - self.base);
            self.base = sample;
        }
    }
}

/// A detector segment as a planner interval.
fn to_interval(seg: SpeechSegment) -> Interval {
    Interval {
        start: seg.start,
        end: seg.end,
    }
}

/// Transcribes a whole in-memory buffer via a streaming session — one code
/// path with the block-wise stream, which the session's commit rule makes
/// bit-identical.
///
/// # Errors
///
/// See [`StreamTranscriber::push`].
pub fn transcribe_with_progress(
    model: &dyn AsrModel,
    tokenizer: &Tokenizer,
    audio: &[f32],
    options: &TranscribeOptions,
    progress: &mut dyn FnMut(TranscribeProgress),
) -> Result<Transcription, GigaamError> {
    let total = audio.len() as f64 / SAMPLE_RATE as f64;
    let mut session =
        StreamTranscriber::new(model, tokenizer, options.clone(), Some(total))?;
    session.push(audio, progress)?;
    session.finish(progress)
}

/// [`transcribe_with_progress`] without a progress callback.
///
/// # Errors
///
/// See [`StreamTranscriber::push`].
pub fn transcribe(
    model: &dyn AsrModel,
    tokenizer: &Tokenizer,
    audio: &[f32],
    options: &TranscribeOptions,
) -> Result<Transcription, GigaamError> {
    transcribe_with_progress(model, tokenizer, audio, options, &mut |_| {})
}
