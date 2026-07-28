//! Driving one synthesis end to end.
//!
//! Ties the pieces together: prepare the reference, take its mel spectrogram
//! and speech rate, tokenize the transcript together with the text to speak,
//! decide how long the result will be, then hand the solver a starting point
//! and walk it to the end. A long text becomes several utterances — the window
//! a single one fits in is what the reference's own budget leaves — and they
//! are joined by the shared layer.
//!
//! Nothing here touches a tensor: the networks sit behind [`SpeechModel`], so
//! this reads the same on either runtime.

use std::path::Path;

use super::{
    SpeechProgress, Stage, SynthesisOptions,
    config::DitConfig,
    error::EspeechError,
    mel::MelBasis,
    model::SpeechModel,
    reference::Reference,
    schedule,
    tokenizer::{self, CharTokenizer},
};
use crate::tts::{self, CloneMode, Join, Speech, Voice};

/// How the schedule is bent. Negative crowds the steps toward the start of the
/// trajectory, where the detail is decided; the reference uses exactly this
/// value and offers no reason to expose it.
const SWAY: f32 = -1.0;

/// Share of the paragraph pause used between pieces of one paragraph — those
/// are mid-thought, not a new one.
const CONTINUATION_PAUSE: f64 = 0.5;

/// The odd multiplier that spreads consecutive indices across the seed space
/// (the golden-ratio constant used for exactly this).
const SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Languages the checkpoints speak, as `--language` may spell them.
const LANGUAGES: &[&str] = &["auto", "russian", "ru", "rus"];

/// The result of one synthesis: the audio and what it took to make it.
#[derive(Debug, Clone)]
pub struct Synthesis {
    /// The synthesized audio.
    pub speech: Speech,
    /// How the voice was chosen.
    pub voice: Voice,
    /// Mel frames generated — 93.75 to the second, summed over the pieces and
    /// counting only what was added to the reference.
    pub frames: usize,
    /// Pieces spoken and stitched together.
    pub chunks: usize,
    /// Seconds of reference the voice was taken from, after preparation.
    pub ref_seconds: f64,
    /// Whether the reference had to be shortened to the model's window.
    pub ref_clipped: bool,
}

/// How an over-budget paragraph is cut down: the text and the budget go in,
/// pieces within that budget come out.
///
/// The engine states the need and stays out of the how — sentence segmentation
/// is a model of its own, and the caller decides which one (and pays for it
/// only when a paragraph really is too long).
pub type SplitParagraph<'a> =
    dyn FnMut(&str, usize) -> Result<Vec<String>, EspeechError> + 'a;

/// What one piece came back as, before the pieces are joined.
struct SpokenPiece {
    speech: Speech,
    /// Mel frames generated for it, the reference's own frames excluded.
    frames: usize,
}

/// One piece of a planned run: what to speak and how it sits in the text.
struct Piece {
    text: String,
    /// Whether the piece opens a paragraph (rather than continuing one that
    /// had to be cut).
    opens_paragraph: bool,
}

/// A loaded engine with its reference voice, ready to synthesize.
pub struct Synthesizer {
    tokenizer: CharTokenizer,
    model: Box<dyn SpeechModel>,
    reference: Reference,
    /// The reference transcript, closed the way the model expects it.
    ref_text: String,
    /// The reference's log-mel, `[cond_frames, mel]`.
    ref_mel: Vec<f32>,
    cond_frames: usize,
    /// Where the reference came from, for the output contract.
    ref_audio: String,
}

impl Synthesizer {
    /// Wraps a loaded runtime with its character table and its reference voice.
    ///
    /// # Errors
    ///
    /// Returns [`EspeechError::RefTextRequired`] for a blank transcript,
    /// [`EspeechError::Checkpoint`] when the character table is missing, and
    /// [`EspeechError::RefAudioTooShort`] when the reference holds too little
    /// audio for a spectrogram.
    pub fn load(
        model_dir: &Path,
        model: Box<dyn SpeechModel>,
        reference: Reference,
        ref_text: &str,
        ref_audio: &str,
    ) -> Result<Self, EspeechError> {
        if ref_text.trim().is_empty() {
            return Err(EspeechError::RefTextRequired);
        }
        let tokenizer =
            CharTokenizer::load(&model_dir.join(super::download::VOCAB_FILE))?;
        if tokenizer.size() != model.config().vocab_size {
            return Err(EspeechError::Checkpoint(format!(
                "the character table holds {} entries but the model was \
                 trained for {}",
                tokenizer.size(),
                model.config().vocab_size
            )));
        }
        let ref_mel = model.mel_basis().log_mel(&reference.wave)?;
        let cond_frames = MelBasis::frames_for(reference.wave.len());
        Ok(Self {
            tokenizer,
            model,
            reference,
            ref_text: tokenizer::close_reference_text(ref_text.trim()),
            ref_mel,
            cond_frames,
            ref_audio: ref_audio.to_owned(),
        })
    }

    /// The geometry the model was loaded with.
    #[must_use]
    pub fn config(&self) -> &DitConfig { self.model.config() }

    /// How much text one piece may hold, in the unit
    /// [`text_cost`](Self::text_cost) measures.
    #[must_use]
    pub fn chunk_budget(&self, speed: f32) -> usize {
        schedule::chunk_budget(
            self.ref_text.len(),
            self.reference.seconds(),
            speed,
        )
    }

    /// What a piece of text costs against [`chunk_budget`](Self::chunk_budget):
    /// its bytes of UTF-8, which is the unit the reference's own estimate of
    /// speech rate is stated in.
    #[must_use]
    pub fn text_cost(text: &str) -> usize { text.len() }

    /// Rejects a language the checkpoints cannot speak.
    ///
    /// # Errors
    ///
    /// Returns [`EspeechError::UnsupportedLanguage`] for anything but Russian.
    pub fn check_language(language: Option<&str>) -> Result<(), EspeechError> {
        let Some(language) = language else {
            return Ok(());
        };
        if LANGUAGES
            .iter()
            .any(|known| language.eq_ignore_ascii_case(known))
        {
            return Ok(());
        }
        Err(EspeechError::UnsupportedLanguage(format!(
            "this engine's checkpoints speak Russian only, not `{language}`; \
             for other languages use `trakktor tts qwen3-tts`"
        )))
    }

    /// Synthesizes a whole text, paragraph by paragraph, and stitches the
    /// pieces into one waveform.
    ///
    /// A paragraph over [`chunk_budget`](Self::chunk_budget) is handed to
    /// `split(text, budget)`, which must return pieces within that budget; the
    /// caller decides how. Pieces that still come back too long are cut at
    /// punctuation as a last resort — on the short budget a reference recording
    /// leaves, a single long sentence is ordinary rather than exceptional.
    ///
    /// Each piece starts from its own noise, derived from the run's seed, so a
    /// rerun reproduces the file while neighbouring pieces do not share a
    /// starting point.
    ///
    /// # Errors
    ///
    /// Returns [`EspeechError::TextEmpty`] when nothing is left to speak, and
    /// otherwise a backend failure, plus whatever `split` reports.
    pub fn speak_paragraphs(
        &mut self,
        paragraphs: &[String],
        options: &SynthesisOptions,
        split: &mut SplitParagraph<'_>,
        progress: &mut dyn FnMut(SpeechProgress),
    ) -> Result<Synthesis, EspeechError> {
        let budget = self.chunk_budget(options.speed);
        let plan = self.plan(paragraphs, budget, split)?;
        if plan.is_empty() {
            return Err(EspeechError::TextEmpty);
        }

        let total_cost: usize =
            plan.iter().map(|piece| Self::text_cost(&piece.text)).sum();
        let mut spoken: Vec<Speech> = Vec::new();
        let mut pauses: Vec<f64> = Vec::new();
        let mut frames = 0usize;
        let mut done_cost = 0usize;
        let mut finished_audio = 0.0f64;

        for (index, piece) in plan.iter().enumerate() {
            let seed = options
                .seed
                .wrapping_add((index as u64).wrapping_mul(SEED_STRIDE));
            let chunks = plan.len();
            let spoken_piece = self.speak_piece(
                &piece.text,
                options,
                seed,
                &mut |stage, step, steps, audio| {
                    progress(SpeechProgress {
                        stage,
                        chunk: index + 1,
                        chunks,
                        step,
                        steps,
                        finished_audio,
                        audio: finished_audio + audio,
                        done_cost,
                        total_cost,
                    });
                },
            )?;

            if index > 0 {
                pauses.push(if piece.opens_paragraph {
                    options.pause
                } else {
                    options.pause * CONTINUATION_PAUSE
                });
            }
            frames += spoken_piece.frames;
            done_cost += Self::text_cost(&piece.text);
            finished_audio += spoken_piece.speech.duration();
            spoken.push(spoken_piece.speech);
        }

        Ok(Synthesis {
            speech: tts::stitch(
                &spoken,
                &pauses,
                Join {
                    match_levels: options.match_levels,
                    ..Join::default()
                },
            ),
            voice: Voice::Clone {
                mode: CloneMode::InContext,
                ref_audio: self.ref_audio.clone(),
            },
            frames,
            chunks: spoken.len(),
            ref_seconds: self.reference.seconds(),
            ref_clipped: self.reference.clipped,
        })
    }

    /// Speaks one piece, reporting progress as the solver walks.
    ///
    /// Returns the audio and the mel frames it was generated from.
    fn speak_piece(
        &mut self,
        text: &str,
        options: &SynthesisOptions,
        seed: u64,
        progress: &mut dyn FnMut(Stage, usize, usize, f64),
    ) -> Result<SpokenPiece, EspeechError> {
        if text.trim().is_empty() {
            return Err(EspeechError::TextEmpty);
        }
        let mel_channels = self.model.config().mel_channels;
        let cut_frames = self.reference.cut_frames();

        // The model sees one string: the reference transcript and then the text
        // to speak. The reference tokenizes exactly this concatenation.
        let ids = self.tokenizer.encode(&format!("{}{text}", self.ref_text));
        let frames = schedule::duration_frames(
            cut_frames,
            self.ref_text.len(),
            text.len(),
            self.cond_frames,
            ids.len(),
            options.speed,
        );
        let generated = frames.saturating_sub(cut_frames);
        let seconds = generated as f64 * f64::from(super::config::HOP as u32) /
            f64::from(self.model.sample_rate());

        let noise = schedule::noise(seed, frames, mel_channels);
        self.model.prepare(
            &self.ref_mel,
            self.cond_frames,
            &ids,
            frames,
            &noise,
        )?;

        let times = schedule::timesteps(options.nfe_step, SWAY);
        progress(Stage::Solving, 0, options.nfe_step, seconds);
        for step in 0..options.nfe_step {
            let (from, to) = (times[step], times[step + 1]);
            self.model.step(from, to - from, options.cfg_strength)?;
            progress(Stage::Solving, step + 1, options.nfe_step, seconds);
        }

        // Vocoding is one long call with nothing to report from inside it;
        // announcing the stage is what keeps a caller's progress line from
        // looking stuck.
        progress(Stage::Vocoding, options.nfe_step, options.nfe_step, seconds);
        let mut samples = self.model.finish(cut_frames)?;
        let gain = self.reference.output_gain();
        if gain != 1.0 {
            for sample in &mut samples {
                *sample *= gain;
            }
        }
        Ok(SpokenPiece {
            speech: Speech {
                samples,
                sample_rate: self.model.sample_rate(),
            },
            frames: generated,
        })
    }

    /// Lays out the pieces to speak: one per paragraph, or several when a
    /// paragraph is over budget.
    fn plan(
        &self,
        paragraphs: &[String],
        budget: usize,
        split: &mut SplitParagraph<'_>,
    ) -> Result<Vec<Piece>, EspeechError> {
        let mut plan = Vec::new();
        for paragraph in paragraphs {
            let text = paragraph.trim();
            if text.is_empty() {
                continue;
            }
            if Self::text_cost(text) <= budget {
                plan.push(Piece {
                    text: text.to_owned(),
                    opens_paragraph: true,
                });
                continue;
            }
            let mut parts = split(text, budget)?;
            // A splitter that returns nothing must not swallow the paragraph.
            if parts.is_empty() {
                parts.push(text.to_owned());
            }
            let mut first = true;
            for part in parts {
                for piece in fit(&part, budget) {
                    plan.push(Piece {
                        text: piece,
                        opens_paragraph: first,
                    });
                    first = false;
                }
            }
        }
        Ok(plan)
    }
}

/// Cuts `text` down to pieces of at most `budget` bytes, at punctuation where
/// there is any and at a word boundary otherwise.
///
/// This is the floor under the caller's splitter, not a replacement for it: a
/// sentence longer than the budget cannot be split by a sentence model, and the
/// budget a reference recording leaves is short enough that such sentences are
/// common.
fn fit(text: &str, budget: usize) -> Vec<String> {
    let text = text.trim();
    if text.len() <= budget || budget == 0 {
        return vec![text.to_owned()];
    }
    let mut pieces = Vec::new();
    let mut rest = text;
    while rest.len() > budget {
        // The last break that fits: a clause boundary if there is one, else a
        // space, else the budget itself.
        let window = &rest[..char_boundary(rest, budget)];
        let cut = window
            .rfind([',', ';', ':', '.', '!', '?', '—', '–'])
            .map(|index| {
                index + window[index..].chars().next().map_or(1, char::len_utf8)
            })
            .or_else(|| window.rfind(' '))
            .unwrap_or(window.len());
        // A cut at zero would hand the whole remainder back and loop forever.
        // Only a budget narrower than one character gets there, which the
        // budget rules do not produce — but a loop is not something to
        // leave resting on a caller's arithmetic.
        let cut =
            cut.max(rest.chars().next().map_or(rest.len(), char::len_utf8));
        let (piece, tail) = rest.split_at(cut.min(rest.len()));
        let piece = piece.trim();
        if !piece.is_empty() {
            pieces.push(piece.to_owned());
        }
        rest = tail.trim_start();
        if rest.is_empty() {
            break;
        }
    }
    if !rest.is_empty() {
        pieces.push(rest.to_owned());
    }
    pieces
}

/// The largest character boundary at or below `at`.
fn char_boundary(text: &str, at: usize) -> usize {
    let mut at = at.min(text.len());
    while at > 0 && !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

#[cfg(test)]
mod tests;
