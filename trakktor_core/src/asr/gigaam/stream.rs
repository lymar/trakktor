//! Streamed, bounded-memory transcription.
//!
//! [`StreamTranscriber`] consumes 16 kHz mono PCM in blocks of any size and
//! produces the same transcription as a whole-file run, holding only a few
//! minutes of audio at a time:
//!
//! - blocks feed the (causal) voice-activity model incrementally;
//! - finalized speech intervals go to the shared chunk planner
//!   ([`ChunkPlanner`]), which runs the DP cut optimization over a rolling
//!   lookahead window: once the pending speech spans [`COMMIT_HORIZON_S`],
//!   every chunk ending at least [`COMMIT_MARGIN_S`] before the frontier is
//!   *committed* — cut choices that far behind no longer change when more
//!   speech arrives;
//! - committed chunks are transcribed immediately and their PCM is dropped, so
//!   the retained buffer stays around `horizon + VAD latency` seconds
//!   regardless of the file length. A chunk's audio window reaches a little
//!   past its speech into the boundary pauses, so the planner holds that much
//!   extra PCM behind the pending speech, and remembers the last committed
//!   window's edge to keep the next one from overlapping it.
//!
//! A committed chunk is transcribed in two overlapped halves: the device
//! half (features, encoder, the one read-back) runs inline, and the pure-CPU
//! token decode — where the RNN-T head spends its sequential loop — runs on a
//! worker thread while the device already encodes the next chunk. One chunk
//! is in flight at a time; its segment is recorded when the next chunk's
//! device half is done (or at the end of the stream), so segments stay in
//! order. Every chunk goes through the same computation as an inline decode —
//! only the moment of computation moves — so the transcription is
//! bit-identical to the unpipelined one.
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
    super::segment::{
        BOUNDARY_CONTEXT_S, Chunk, ChunkPlanner, DecodedWord, Interval,
        own_words,
    },
    constants::{LONGFORM_THRESHOLD_S, SAMPLE_RATE},
    decode::Word,
    error::GigaamError,
    runtime::{AsrModel, Emissions},
    tokenizer::Tokenizer,
    transcribe::{
        Segment, TranscribeOptions, TranscribeProgress, Transcription,
        chunk_result,
    },
};
pub use crate::asr::segment::{COMMIT_HORIZON_S, COMMIT_MARGIN_S};
use crate::vad::{
    SpeechDetector, SpeechSegment, Vad, VadOptions, VadStreamState,
};

/// Extra samples kept behind the earliest needed position when shedding PCM,
/// absorbing the window quantization of the VAD grid.
const SHED_GUARD_SAMPLES: usize = 512;

/// A committed chunk whose device half is done and whose pure-CPU token
/// decode is running on a worker thread — the pipelining that hides the
/// transducer head's cost behind the next chunk's encode.
struct InFlight {
    chunk: Chunk,
    /// Absolute sample the chunk's audio window started at — the origin that
    /// puts its word times onto the recording's timeline.
    origin_sample: usize,
    /// The window's PCM length in samples — the words' frame→time scale.
    samples: usize,
    decode: std::thread::JoinHandle<Emissions>,
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
    /// Time spans of the words the previous chunk kept — what the seam
    /// reconciliation of the next chunk tests against.
    prev_kept: Vec<(f64, f64)>,
    /// The chunk whose decode is riding behind the current encode, if any.
    in_flight: Option<InFlight>,
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
            prev_kept: Vec::new(),
            in_flight: None,
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
                self.transcribe_committed(
                    Chunk {
                        start: 0.0,
                        end: total,
                        keep_start: 0.0,
                        keep_end: total,
                        window_start: 0.0,
                        window_end: total,
                    },
                    progress,
                )?;
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

        // Drain the pipeline: the last chunk has no successor to ride behind.
        if let Some(last) = self.in_flight.take() {
            self.record_segment(last, progress);
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

    /// Transcribes one committed chunk from the retained buffer: runs the
    /// device half inline, hands the pure-CPU decode to a worker thread, and
    /// records the segment of the *previous* in-flight chunk, whose decode
    /// overlapped this chunk's encode.
    fn transcribe_committed(
        &mut self,
        chunk: Chunk,
        progress: &mut dyn FnMut(TranscribeProgress),
    ) -> Result<(), GigaamError> {
        let sr = SAMPLE_RATE as f64;
        let a = (chunk.window_start.max(0.0) * sr) as usize;
        let b = ((chunk.window_end * sr) as usize).min(self.fed);
        if b <= a {
            return Ok(());
        }
        debug_assert!(
            a >= self.base,
            "chunk window at {}s reaches below the retained buffer",
            chunk.window_start
        );
        let pcm = &self.buf[a - self.base..b - self.base];
        let samples = pcm.len();
        let mel = self.model.feature().log_mel(pcm);
        let encoded = self.model.encode_chunk(&mel)?;
        let previous = self.in_flight.replace(InFlight {
            chunk,
            origin_sample: a,
            samples,
            decode: std::thread::spawn(move || encoded.decode()),
        });
        if let Some(previous) = previous {
            self.record_segment(previous, progress);
        }
        Ok(())
    }

    /// Joins an in-flight chunk's decode and records its segment. The model
    /// saw the chunk's audio *window*; the segment reports the speech span,
    /// and word times — local to the window — are shifted onto the original
    /// timeline by the window's start.
    fn record_segment(
        &mut self,
        done: InFlight,
        progress: &mut dyn FnMut(TranscribeProgress),
    ) {
        let emitted = match done.decode.join() {
            Ok(emitted) => emitted,
            // A worker panic is a head bug; surface it as our own panic.
            Err(panic) => std::panic::resume_unwind(panic),
        };
        let result = chunk_result(self.tokenizer, &emitted, done.samples);
        let origin = done.origin_sample as f64 / SAMPLE_RATE as f64;
        let (text, words) = own_words(
            result.words,
            origin,
            &done.chunk,
            &self.prev_kept,
            result.text,
        );
        self.prev_kept = words.iter().map(DecodedWord::span).collect();
        if !text.is_empty() {
            self.texts.push(text.clone());
        }
        self.segments.push(Segment {
            id: self.segments.len(),
            start: done.chunk.start,
            end: done.chunk.end,
            text,
            words: if self.options.word_timestamps {
                words
            } else {
                Vec::new()
            },
        });
        progress(TranscribeProgress {
            processed_seconds: done.chunk.end,
            total_seconds: self.total_hint,
        });
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
        // frontier is settled. Both speech positions are pulled back by the
        // context a future chunk's window may reach into.
        let mut needed = self.windows as f64 * 512.0 / sr;
        if let Some(start) = self.planner.earliest_pending_start() {
            needed = needed.min(start);
        }
        if let Some(start) = self
            .detector
            .as_ref()
            .and_then(SpeechDetector::earliest_pending_second)
        {
            needed = needed.min(start - BOUNDARY_CONTEXT_S);
        }
        let sample =
            ((needed * sr) as usize).saturating_sub(SHED_GUARD_SAMPLES);
        if sample > self.base {
            self.buf.drain(..sample - self.base);
            self.base = sample;
        }
    }
}

impl DecodedWord for Word {
    fn span(&self) -> (f64, f64) { (self.start, self.end) }

    fn shift(&mut self, offset: f64) {
        self.start += offset;
        self.end += offset;
    }

    fn text(&self) -> &str { &self.text }
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
