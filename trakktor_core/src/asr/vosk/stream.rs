//! Streamed, bounded-memory transcription in both model modes.
//!
//! [`StreamTranscriber`] consumes 16 kHz mono PCM in blocks of any size and
//! produces the same transcription as a whole-file run, holding a bounded
//! buffer. The pipeline depends on the model kind:
//!
//! - **Offline (full-context) models** follow the long-form scheme shared with
//!   the sibling engines: blocks feed the causal voice-activity model
//!   incrementally; finalized speech intervals go to the shared chunk planner,
//!   which runs the DP cut optimization over a rolling window (see
//!   [`crate::asr::segment`]); committed chunks are transcribed immediately
//!   (one full-context encoder pass + the transducer search) and their PCM is
//!   dropped. Audio no longer than the long-form threshold is one whole chunk.
//!   Commits are triggered by the speech/window timeline, never block
//!   boundaries, so any block sizing yields an identical transcription.
//!
//! - **Streaming (causal) models** run the native chunked pipeline: fbank
//!   frames are computed incrementally, and every full feature window
//!   (`window_frames`, advancing by `shift_frames`) goes through the stateful
//!   encoder session; the transducer search runs continuously over the whole
//!   file (the reference resets only around live-microphone endpointing, which
//!   a file does not need). At the end the audio is padded with a short silence
//!   tail so the last window clears the readiness gate, exactly like the
//!   reference applications. Output segments are presentation-only: emissions
//!   split at pauses of at least [`SEGMENT_GAP_S`]. No VAD is used; memory is
//!   bounded by the model state plus one window of PCM.

#[cfg(test)]
mod tests;

use super::{
    super::segment::{
        BOUNDARY_CONTEXT_S, Chunk, ChunkPlanner, DecodedWord, Interval,
        own_words,
    },
    constants::{ENCODER_FRAME_S, LONGFORM_THRESHOLD_S, SAMPLE_RATE},
    decode::{DecodeState, TransducerHead},
    error::VoskError,
    feature::{FRAME_SHIFT, FbankExtractor, OFFSET},
    runtime::{EncoderSeam, StreamingSession},
    tokenizer::Tokenizer,
    transcribe::{
        Segment, TranscribeOptions, TranscribeProgress, Transcription, Word,
        tokens_to_words, transcribe_chunk,
    },
};
pub use crate::asr::segment::{COMMIT_HORIZON_S, COMMIT_MARGIN_S};
use crate::vad::{
    SpeechDetector, SpeechSegment, Vad, VadOptions, VadStreamState,
};

/// Extra samples kept behind the earliest needed position when shedding PCM.
const SHED_GUARD_SAMPLES: usize = 512;

/// Silence appended after the last sample so the final streaming window
/// clears the readiness gate (the reference applications' tail padding).
const TAIL_PAD_S: f64 = 0.8;

/// A pause of at least this long between emissions starts a new output
/// segment of a streaming-model run.
pub const SEGMENT_GAP_S: f64 = 1.0;

/// The offline-model pipeline state: VAD, planner, and transcribed segments.
struct OfflineRun {
    vad: Vad,
    vad_state: VadStreamState,
    /// `None` only transiently inside `finish`.
    detector: Option<SpeechDetector>,
    planner: ChunkPlanner,
    /// VAD windows whose probabilities have been consumed.
    windows: u64,
    /// Time spans of the words the previous chunk kept — what the seam
    /// reconciliation of the next chunk tests against.
    prev_kept: Vec<(f64, f64)>,
    segments: Vec<Segment>,
    texts: Vec<String>,
}

/// The streaming-model pipeline state: the encoder session, the feature
/// cursor, and the running transducer search.
struct StreamingRun<'m> {
    session: Box<dyn StreamingSession + 'm>,
    decode: DecodeState,
    /// Start frame (on the signal's frame grid) of the next window.
    next_window: usize,
    /// Feature frames per window and per advance.
    window_frames: usize,
    shift_frames: usize,
}

enum Mode<'m> {
    // The offline run owns a VAD model, the streaming run the transducer
    // search state; box both so the enum stays small.
    Offline(Box<OfflineRun>),
    Streaming(Box<StreamingRun<'m>>),
}

/// A streamed transcription session. Feed PCM with [`push`](Self::push),
/// then call [`finish`](Self::finish) once.
pub struct StreamTranscriber<'m> {
    model: &'m dyn EncoderSeam,
    head: &'m TransducerHead,
    tokenizer: &'m Tokenizer,
    fbank: FbankExtractor,
    options: TranscribeOptions,
    /// The declared total duration, when known — progress reporting only.
    total_hint: Option<f64>,
    mode: Mode<'m>,
    /// Retained PCM; `buf[0]` is absolute sample `base`.
    buf: Vec<f32>,
    base: usize,
    /// Total samples fed.
    fed: usize,
}

impl<'m> StreamTranscriber<'m> {
    /// Creates a session; the mode follows the model (streaming geometry ⇒
    /// the chunked session, otherwise the long-form offline pipeline).
    /// `total_hint` (seconds), when known, only feeds the progress callback.
    ///
    /// # Errors
    ///
    /// [`VoskError::Vad`] when the VAD model cannot be loaded (offline mode);
    /// model errors from opening the streaming session.
    pub fn new(
        model: &'m dyn EncoderSeam,
        head: &'m TransducerHead,
        tokenizer: &'m Tokenizer,
        options: TranscribeOptions,
        total_hint: Option<f64>,
    ) -> Result<Self, VoskError> {
        let mode = match &model.config().streaming {
            Some(streaming) => Mode::Streaming(Box::new(StreamingRun {
                session: model.start_stream()?,
                decode: DecodeState::new(head, options.decoding),
                next_window: 0,
                window_frames: streaming.window_frames,
                shift_frames: streaming.shift_frames,
            })),
            None => Mode::Offline(Box::new(OfflineRun {
                vad: Vad::load().map_err(|e| VoskError::Vad(e.to_string()))?,
                vad_state: VadStreamState::new(),
                detector: Some(SpeechDetector::new(&VadOptions::default())),
                planner: ChunkPlanner::new(),
                windows: 0,
                prev_kept: Vec::new(),
                segments: Vec::new(),
                texts: Vec::new(),
            })),
        };
        Ok(Self {
            model,
            head,
            tokenizer,
            fbank: FbankExtractor::new(),
            options,
            total_hint,
            mode,
            buf: Vec::new(),
            base: 0,
            fed: 0,
        })
    }

    /// Feeds the next block of 16 kHz mono samples. Work that became final —
    /// committed offline chunks, full streaming windows — is processed
    /// before returning (invoking `progress`), and drained PCM is released.
    ///
    /// # Errors
    ///
    /// [`VoskError::Vad`] on a VAD failure (offline) and the model errors of
    /// encoding itself.
    pub fn push(
        &mut self,
        samples: &[f32],
        progress: &mut dyn FnMut(TranscribeProgress),
    ) -> Result<(), VoskError> {
        self.buf.extend_from_slice(samples);
        self.fed += samples.len();
        if matches!(self.mode, Mode::Streaming(_)) {
            self.pump_windows(false, progress)?;
            self.shed_pcm_streaming();
        } else {
            let probs = {
                let Mode::Offline(run) = &mut self.mode else {
                    unreachable!()
                };
                run.vad
                    .stream_push(&mut run.vad_state, samples)
                    .map_err(|e| VoskError::Vad(e.to_string()))?
            };
            self.consume_probs(&probs, progress)?;
            self.shed_pcm_offline();
        }
        Ok(())
    }

    /// Ends the stream: flushes the pipeline, processes the remainder, and
    /// returns the full transcription.
    ///
    /// # Errors
    ///
    /// See [`push`](Self::push).
    pub fn finish(
        mut self,
        progress: &mut dyn FnMut(TranscribeProgress),
    ) -> Result<Transcription, VoskError> {
        let total = self.fed as f64 / SAMPLE_RATE as f64;
        if matches!(self.mode, Mode::Streaming(_)) {
            self.finish_streaming(total, progress)
        } else {
            self.finish_offline(total, progress)
        }
    }

    /// Samples currently retained — the streaming memory bound (diagnostics
    /// and tests).
    #[must_use]
    pub fn buffered_samples(&self) -> usize { self.buf.len() }

    // ---- offline-model pipeline ------------------------------------------

    fn finish_offline(
        &mut self,
        total: f64,
        progress: &mut dyn FnMut(TranscribeProgress),
    ) -> Result<Transcription, VoskError> {
        let probs = {
            let Mode::Offline(run) = &mut self.mode else {
                unreachable!()
            };
            run.vad
                .stream_finish(&mut run.vad_state)
                .map_err(|e| VoskError::Vad(e.to_string()))?
        };
        self.consume_probs(&probs, progress)?;

        // Short audio: one whole-file chunk, like the reference; the PCM is
        // intact — shedding never runs below the long-form threshold.
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
            let Mode::Offline(run) = &mut self.mode else {
                unreachable!()
            };
            let mut tail = Vec::new();
            if let Some(detector) = run.detector.take() {
                detector.finish(self.fed, &mut tail);
            }
            let mut chunks = Vec::new();
            for seg in tail {
                chunks.extend(run.planner.push(to_interval(seg)));
            }
            chunks.extend(run.planner.finish(total));
            for chunk in chunks {
                self.transcribe_committed(chunk, progress)?;
            }
        }

        let Mode::Offline(run) = &mut self.mode else {
            unreachable!()
        };
        Ok(Transcription {
            text: run.texts.join(" "),
            segments: std::mem::take(&mut run.segments),
            duration: total,
        })
    }

    /// Routes freshly computed window probabilities through the detector and
    /// the planner, transcribing any chunks that became final.
    fn consume_probs(
        &mut self,
        probs: &[f32],
        progress: &mut dyn FnMut(TranscribeProgress),
    ) -> Result<(), VoskError> {
        let mut committed: Vec<Chunk> = Vec::new();
        {
            let Mode::Offline(run) = &mut self.mode else {
                unreachable!()
            };
            let Some(mut detector) = run.detector.take() else {
                return Ok(());
            };
            let mut emitted = Vec::new();
            for &prob in probs {
                run.windows += 1;
                emitted.clear();
                detector.push(prob, &mut emitted);
                for &seg in &emitted {
                    committed.extend(run.planner.push(to_interval(seg)));
                }
                // The fully processed position: long silences commit (and
                // later release) pending speech even with no new intervals.
                let frontier = run.windows as f64 * 512.0 / SAMPLE_RATE as f64;
                committed.extend(run.planner.poll(frontier));
            }
            run.detector = Some(detector);
        }
        for chunk in committed {
            self.transcribe_committed(chunk, progress)?;
        }
        Ok(())
    }

    /// Transcribes one committed chunk from the retained buffer and records the
    /// segment. The model sees the chunk's audio *window*; the segment reports
    /// the speech span, and word times — local to the window — are shifted onto
    /// the original timeline by `window_start`.
    fn transcribe_committed(
        &mut self,
        chunk: Chunk,
        progress: &mut dyn FnMut(TranscribeProgress),
    ) -> Result<(), VoskError> {
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
        let pcm = self.buf[a - self.base..b - self.base].to_vec();
        let result = transcribe_chunk(
            self.model,
            self.head,
            self.tokenizer,
            &self.fbank,
            &pcm,
            &self.options,
        )?;
        let origin = a as f64 / sr;
        let want_words = self.options.word_timestamps;
        let Mode::Offline(run) = &mut self.mode else {
            unreachable!()
        };
        let (text, words) = own_words(
            result.words,
            origin,
            &chunk,
            &run.prev_kept,
            result.text,
        );
        run.prev_kept = words.iter().map(DecodedWord::span).collect();
        if !text.is_empty() {
            run.texts.push(text.clone());
        }
        run.segments.push(Segment {
            id: run.segments.len(),
            start: chunk.start,
            end: chunk.end,
            text,
            words: if want_words { words } else { Vec::new() },
        });
        progress(TranscribeProgress {
            processed_seconds: chunk.end,
            total_seconds: self.total_hint,
        });
        Ok(())
    }

    /// Drops PCM that no present or future chunk can need (offline mode).
    /// Nothing is shed until the audio is longer than the long-form
    /// threshold, so the single-chunk path always has the whole file.
    fn shed_pcm_offline(&mut self) {
        if self.fed as f64 / SAMPLE_RATE as f64 <= LONGFORM_THRESHOLD_S {
            return;
        }
        let Mode::Offline(run) = &mut self.mode else {
            unreachable!()
        };
        let sr = SAMPLE_RATE as f64;
        // Both speech positions are pulled back by the context a future
        // chunk's audio window may reach into.
        let mut needed = run.windows as f64 * 512.0 / sr;
        if let Some(start) = run.planner.earliest_pending_start() {
            needed = needed.min(start);
        }
        if let Some(start) = run
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

    // ---- streaming-model pipeline ----------------------------------------

    /// Encodes and decodes every feature window that is ready. Without
    /// `flushed` only windows strictly inside the finalized frame count are
    /// consumed (the reference readiness gate `processed + T < ready`).
    fn pump_windows(
        &mut self,
        flushed: bool,
        progress: &mut dyn FnMut(TranscribeProgress),
    ) -> Result<(), VoskError> {
        loop {
            let Mode::Streaming(run) = &mut self.mode else {
                unreachable!()
            };
            let ready = self.fbank.n_frames(self.fed, flushed);
            if run.next_window + run.window_frames >= ready {
                return Ok(());
            }
            let features = self.fbank.compute_range(
                &self.buf,
                self.base,
                run.next_window,
                run.window_frames,
            );
            let (encoded, frames) = run.session.accept(&features)?;
            run.decode.decode_block(self.head, &encoded, frames);
            run.next_window += run.shift_frames;
            progress(TranscribeProgress {
                processed_seconds: run.next_window as f64 * FRAME_SHIFT as f64 /
                    SAMPLE_RATE as f64,
                total_seconds: self.total_hint,
            });
        }
    }

    /// Drops PCM below the earliest sample the next window can read.
    fn shed_pcm_streaming(&mut self) {
        let Mode::Streaming(run) = &self.mode else {
            unreachable!()
        };
        let first_needed = (run.next_window * FRAME_SHIFT)
            .saturating_sub(OFFSET + SHED_GUARD_SAMPLES);
        if first_needed > self.base {
            self.buf.drain(..first_needed - self.base);
            self.base = first_needed;
        }
    }

    fn finish_streaming(
        &mut self,
        total: f64,
        progress: &mut dyn FnMut(TranscribeProgress),
    ) -> Result<Transcription, VoskError> {
        // Tail padding: silence so the last real window clears the gate.
        let pad = (TAIL_PAD_S * SAMPLE_RATE as f64) as usize;
        self.buf.extend(std::iter::repeat_n(0.0f32, pad));
        self.fed += pad;
        self.pump_windows(true, progress)?;

        let Mode::Streaming(run) = &mut self.mode else {
            unreachable!()
        };
        let decode = std::mem::replace(
            &mut run.decode,
            DecodeState::new(self.head, self.options.decoding),
        );
        let emitted = decode.finish(self.head);

        // Presentation segmentation: split at pauses ≥ SEGMENT_GAP_S.
        let gap_frames = (SEGMENT_GAP_S / ENCODER_FRAME_S).round() as usize;
        let mut segments: Vec<Segment> = Vec::new();
        let mut texts: Vec<String> = Vec::new();
        let mut range_start = 0usize;
        let frames = &emitted.token_frames;
        for i in 0..=emitted.token_ids.len() {
            let boundary = i == emitted.token_ids.len() ||
                (i > range_start && frames[i] - frames[i - 1] >= gap_frames);
            if !boundary {
                continue;
            }
            if i > range_start {
                let ids = &emitted.token_ids[range_start..i];
                let frs = &frames[range_start..i];
                let text = self.tokenizer.decode(ids);
                let words = if self.options.word_timestamps {
                    tokens_to_words(self.tokenizer, ids, frs, 0.0)
                } else {
                    Vec::new()
                };
                let start = frs[0] as f64 * ENCODER_FRAME_S;
                let end = (frs[frs.len() - 1] + 1) as f64 * ENCODER_FRAME_S;
                if !text.is_empty() {
                    texts.push(text.clone());
                }
                segments.push(Segment {
                    id: segments.len(),
                    start,
                    end: end.min(total),
                    text,
                    words,
                });
            }
            range_start = i;
        }

        Ok(Transcription {
            text: texts.join(" "),
            segments,
            duration: total,
        })
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
/// path with the block-wise stream, which the commit rules make
/// bit-identical.
///
/// # Errors
///
/// See [`StreamTranscriber::push`].
pub fn transcribe_with_progress(
    model: &dyn EncoderSeam,
    head: &TransducerHead,
    tokenizer: &Tokenizer,
    audio: &[f32],
    options: &TranscribeOptions,
    progress: &mut dyn FnMut(TranscribeProgress),
) -> Result<Transcription, VoskError> {
    let total = audio.len() as f64 / SAMPLE_RATE as f64;
    let mut session =
        StreamTranscriber::new(model, head, tokenizer, *options, Some(total))?;
    session.push(audio, progress)?;
    session.finish(progress)
}

/// [`transcribe_with_progress`] without a progress callback.
///
/// # Errors
///
/// See [`StreamTranscriber::push`].
pub fn transcribe(
    model: &dyn EncoderSeam,
    head: &TransducerHead,
    tokenizer: &Tokenizer,
    audio: &[f32],
    options: &TranscribeOptions,
) -> Result<Transcription, VoskError> {
    transcribe_with_progress(
        model,
        head,
        tokenizer,
        audio,
        options,
        &mut |_| {},
    )
}
