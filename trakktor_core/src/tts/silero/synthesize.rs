//! Driving one synthesis end to end.
//!
//! Nothing here touches a tensor: the networks sit behind
//! [`SpeechModel`](super::model::SpeechModel), so this reads the same on either
//! runtime. What it owns is the order of things — clean the text, run the
//! pipeline, invert the spectrum, and (when a lower rate was asked for) hand
//! the waveform to the model's own filterbank rather than to a resampler.
//!
//! A long text becomes several utterances. The ceiling is not a matter of taste
//! here but of the position tables the model carries: past five thousand mel
//! frames there is no position to give the next one, so a piece that would
//! reach there is refused rather than quietly truncated.

use super::{
    SpeechProgress, Stage, SynthesisOptions,
    config::{CHUNK_BUDGET, FRAME_SECONDS, SAMPLE_RATE},
    error::SileroError,
    frontend,
    istft::Pqmf,
    model::{SpeechModel, Utterance},
    tables::Tables,
};
use crate::tts::{self, Join, Speech, Voice, text};

/// Share of the paragraph pause used between pieces of one paragraph — those
/// are mid-thought, not a new one.
const CONTINUATION_PAUSE: f64 = 0.5;

/// The sample rates the model itself can produce.
pub const SAMPLE_RATES: [u32; 3] = [48_000, 24_000, 8_000];

/// The result of one synthesis: the audio and what it took to make it.
#[derive(Debug, Clone)]
pub struct Synthesis {
    /// The synthesized audio.
    pub speech: Speech,
    /// How the voice was chosen.
    pub voice: Voice,
    /// Pieces spoken and stitched together.
    pub chunks: usize,
    /// Mel frames generated, summed over the pieces — 80 to the second.
    pub frames: usize,
    /// Symbols fed to the model, summed over the pieces.
    pub symbols: usize,
    /// The utterance types the intonation head was given, when the model has
    /// one.
    pub utterances: Vec<&'static str>,
    /// Characters the frontend could not spell and removed.
    pub dropped: usize,
    /// Pieces skipped whole because nothing in them was readable — a code
    /// block, a line of Latin, a bare number.
    pub skipped: usize,
}

/// How an over-budget paragraph is cut down: the text and the budget go in,
/// pieces within that budget come out.
///
/// The engine states the need and stays out of the how — sentence segmentation
/// is a model of its own, and the caller decides which one (and pays for it
/// only when a paragraph really is too long).
pub type SplitParagraph<'a> =
    dyn FnMut(&str, usize) -> Result<Vec<String>, SileroError> + 'a;

/// One piece of a planned run: what to speak and how it sits in the text.
struct Piece {
    text: String,
    /// Whether the piece opens a paragraph (rather than continuing one that
    /// had to be cut).
    opens_paragraph: bool,
}

/// A loaded engine, ready to synthesize.
pub struct Synthesizer {
    model: Box<dyn SpeechModel>,
    tables: Tables,
}

impl Synthesizer {
    /// Pairs a loaded runtime with the tables of the same model.
    ///
    /// # Errors
    ///
    /// Returns [`SileroError::Checkpoint`] when the tables are missing or do
    /// not match the weights.
    pub fn load(
        model_dir: &std::path::Path,
        model: Box<dyn SpeechModel>,
    ) -> Result<Self, SileroError> {
        let tables = Tables::load(model_dir)?;
        if tables.symbols.len() != model.config().symbols {
            return Err(SileroError::Checkpoint(format!(
                "the alphabet holds {} symbols but the weights were trained \
                 for {}",
                tables.symbols.len(),
                model.config().symbols
            )));
        }
        // Fewer names than rows is fine (a model may keep an unnamed slot);
        // more would let a voice resolve to a row the weights do not have.
        if tables.speakers.len() > model.config().speaker_slots {
            return Err(SileroError::Checkpoint(format!(
                "the tables name {} speakers but the weights hold rows for \
                 only {}",
                tables.speakers.len(),
                model.config().speaker_slots
            )));
        }
        Ok(Self { model, tables })
    }

    /// The voices this model speaks with.
    #[must_use]
    pub fn voices(&self) -> Vec<&str> { self.tables.voices() }

    /// Rejects a sample rate the model cannot produce.
    ///
    /// # Errors
    ///
    /// Returns [`SileroError::InvalidOptions`] for anything but the three the
    /// vocoder's own filterbanks give.
    pub fn check_sample_rate(rate: u32) -> Result<(), SileroError> {
        if SAMPLE_RATES.contains(&rate) {
            return Ok(());
        }
        Err(SileroError::InvalidOptions(format!(
            "this model produces {} Hz and derives {} Hz and {} Hz from it \
             with its own filterbank; `{rate}` is not one of them",
            SAMPLE_RATES[0], SAMPLE_RATES[1], SAMPLE_RATES[2]
        )))
    }

    /// Rejects a reading the durations cannot be computed from.
    ///
    /// The rate divides the predicted durations, so zero or below (or
    /// anything non-finite) has no meaning — division by it would inflate the
    /// frame counts without bound. Zero pitch is allowed: it flattens the
    /// contour into the reference's own `robot` voice.
    ///
    /// # Errors
    ///
    /// Returns [`SileroError::InvalidOptions`] naming the flag.
    pub fn check_reading(rate: f32, pitch: f32) -> Result<(), SileroError> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err(SileroError::InvalidOptions(format!(
                "--rate must be a positive number, not {rate}"
            )));
        }
        if !pitch.is_finite() || pitch < 0.0 {
            return Err(SileroError::InvalidOptions(format!(
                "--pitch must be zero (a flat, robotic contour) or a positive \
                 number, not {pitch}"
            )));
        }
        Ok(())
    }

    /// Resolves a voice name to its row of the speaker table.
    ///
    /// # Errors
    ///
    /// Returns [`SileroError::UnknownVoice`] naming the voices this model has.
    pub fn voice(&self, name: &str) -> Result<usize, SileroError> {
        self.tables
            .speaker(name)
            .ok_or_else(|| SileroError::UnknownVoice {
                voice: name.to_owned(),
                known: self.tables.speaker_list(),
            })
    }

    /// Synthesizes a whole text, paragraph by paragraph, and stitches the
    /// pieces into one waveform.
    ///
    /// A paragraph over the budget is handed to `split(text, budget)`, which
    /// must return pieces within it; the caller decides how. Pieces that still
    /// come back too long are cut at punctuation as a last resort.
    ///
    /// # Errors
    ///
    /// Returns [`SileroError::TextEmpty`] when nothing is left to speak,
    /// [`SileroError::TextTooLong`] when a piece does not fit the model's
    /// window, and otherwise a backend failure plus whatever `split` reports.
    pub fn speak_paragraphs(
        &self,
        paragraphs: &[String],
        options: &SynthesisOptions,
        split: &mut SplitParagraph<'_>,
        progress: &mut dyn FnMut(SpeechProgress),
    ) -> Result<Synthesis, SileroError> {
        Self::check_sample_rate(options.sample_rate)?;
        Self::check_reading(options.rate, options.pitch)?;
        let speaker = self.voice(&options.voice)?;
        let plan = plan(paragraphs, piece_budget(options.rate), split)?;
        if plan.is_empty() {
            return Err(SileroError::TextEmpty);
        }

        // Every piece goes through the frontend first. A piece it can spell
        // nothing of — a code block, a line of Latin, a bare number — is
        // skipped and counted rather than failing the run: those are ordinary
        // residents of a real document, and a reading with a hole beats no
        // reading at all. Only a text with no readable piece left is an error.
        let language = self.language(speaker);
        let has_intonation = self.model.config().utterance_types > 0;
        let mut ready: Vec<(Piece, frontend::Utterances)> = Vec::new();
        let mut dropped = 0usize;
        let mut skipped = 0usize;
        for piece in plan {
            match frontend::prepare(
                &piece.text,
                &self.tables,
                language.as_deref(),
                options.rate,
                options.pitch,
                has_intonation,
            ) {
                Ok(prepared) => ready.push((piece, prepared)),
                Err(SileroError::TextEmpty) => {
                    dropped += piece
                        .text
                        .chars()
                        .filter(|c| !c.is_whitespace())
                        .count();
                    skipped += 1;
                },
                Err(other) => return Err(other),
            }
        }
        if ready.is_empty() {
            return Err(SileroError::TextEmpty);
        }

        let total_cost: usize =
            ready.iter().map(|(piece, _)| piece.text.len()).sum();
        let chunks = ready.len();
        let mut spoken: Vec<Speech> = Vec::new();
        let mut pauses: Vec<f64> = Vec::new();
        let mut frames = 0usize;
        let mut symbols = 0usize;
        let mut done_cost = 0usize;
        let mut finished_audio = 0.0f64;
        let mut utterances: Vec<&'static str> = Vec::new();

        for (index, (piece, prepared)) in ready.iter().enumerate() {
            let mut report = |stage: Stage, audio: f64| {
                progress(SpeechProgress {
                    stage,
                    chunk: index + 1,
                    chunks,
                    finished_audio,
                    audio: finished_audio + audio,
                    done_cost,
                    total_cost,
                });
            };
            report(Stage::Synthesizing, 0.0);

            let piece_speech =
                self.speak_piece(prepared, speaker, options, &mut |audio| {
                    report(Stage::Vocoding, audio);
                })?;
            if index > 0 {
                pauses.push(if piece.opens_paragraph {
                    options.pause
                } else {
                    options.pause * CONTINUATION_PAUSE
                });
            }
            frames += piece_speech.frames;
            symbols += prepared.ids.len();
            dropped += prepared.dropped;
            if has_intonation {
                utterances.extend(
                    frontend::utterance_types(&piece.text)
                        .into_iter()
                        .map(name_of),
                );
            }
            done_cost += piece.text.len();
            finished_audio += piece_speech.speech.duration();
            spoken.push(piece_speech.speech);
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
            voice: Voice::Preset {
                name: options.voice.clone(),
            },
            chunks: spoken.len(),
            frames,
            symbols,
            utterances,
            dropped,
            skipped,
        })
    }

    /// Speaks one already-prepared piece.
    fn speak_piece(
        &self,
        prepared: &frontend::Utterances,
        speaker: usize,
        options: &SynthesisOptions,
        report: &mut dyn FnMut(f64),
    ) -> Result<SpokenPiece, SileroError> {
        let spectrum = self.model.synthesize(&Utterance {
            ids: prepared.ids.clone(),
            speaker,
            rate: prepared.rate.clone(),
            pitch: prepared.pitch.clone(),
            types: prepared.types.clone(),
        })?;
        report(spectrum.frames as f64 * FRAME_SECONDS);

        let wave = self.model.window().inverse(
            &spectrum.magnitude,
            &spectrum.phase,
            spectrum.frames,
        );
        let samples = self.lower_rate(wave, options.sample_rate)?;
        Ok(SpokenPiece {
            speech: Speech {
                samples,
                sample_rate: options.sample_rate,
            },
            frames: spectrum.frames,
        })
    }

    /// Takes the native waveform down to a lower rate, the way the model does
    /// it: through its own analysis filterbank, not a resampler.
    fn lower_rate(
        &self,
        wave: Vec<f32>,
        sample_rate: u32,
    ) -> Result<Vec<f32>, SileroError> {
        if sample_rate == SAMPLE_RATE {
            return Ok(wave);
        }
        let bands = (SAMPLE_RATE / sample_rate) as usize;
        let bank: &Pqmf = self.model.pqmf(bands).ok_or_else(|| {
            SileroError::InvalidOptions(format!(
                "this model carries no filterbank for {sample_rate} Hz"
            ))
        })?;
        Ok(bank.low_band(&wave))
    }

    /// The language a voice is read as, which is the prefix of its name in
    /// every published model that has more than one.
    fn language(&self, speaker: usize) -> Option<String> {
        if self.tables.translit.is_empty() {
            return None;
        }
        self.tables
            .speakers
            .get(speaker)
            .and_then(|name| name.split('_').next())
            .map(str::to_owned)
    }
}

/// What one piece came back as, before the pieces are joined.
struct SpokenPiece {
    speech: Speech,
    frames: usize,
}

/// The piece budget for a run: [`CHUNK_BUDGET`] shrunk in step with a slow
/// rate. Slowing the speech stretches the same bytes over more frames, and the
/// model's ceiling is in frames — without this, `--rate 0.5` would push a
/// full-budget piece past the position table. A fast rate does not widen the
/// budget: it is also the length the model is known to read well.
fn piece_budget(rate: f32) -> usize {
    ((CHUNK_BUDGET as f64) * f64::from(rate.min(1.0)))
        .round()
        .max(1.0) as usize
}

/// Lays out the pieces to speak: one per paragraph, or several when a
/// paragraph is over budget.
fn plan(
    paragraphs: &[String],
    budget: usize,
    split: &mut SplitParagraph<'_>,
) -> Result<Vec<Piece>, SileroError> {
    let mut plan = Vec::new();
    for paragraph in paragraphs {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }
        if paragraph.len() <= budget {
            plan.push(Piece {
                text: paragraph.to_owned(),
                opens_paragraph: true,
            });
            continue;
        }
        let mut parts = split(paragraph, budget)?;
        // A splitter that returns nothing must not swallow the paragraph.
        if parts.is_empty() {
            parts.push(paragraph.to_owned());
        }
        let mut first = true;
        for part in parts {
            for piece in text::fit(&part, budget) {
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

/// The name an utterance type is reported under.
fn name_of(utterance: super::intonation::Utterance) -> &'static str {
    use super::intonation::Utterance as U;
    match utterance {
        U::Statement => "statement",
        U::WhQuestion => "wh_question",
        U::GeneralQuestion => "question",
        U::Alternative => "alternative",
        U::Tag => "tag_question",
        U::Exclamation => "exclamation",
    }
}

#[cfg(test)]
mod tests;
