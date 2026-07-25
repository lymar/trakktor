//! Driving one synthesis end to end.
//!
//! Ties the pieces together: tokenize the text, lay out the two-track prompt,
//! prime the talker with it, then generate frame by frame — the talker picks
//! codebook 0, the code predictor fills the rest, and the finished frame folds
//! back into the talker's next input. When the talker calls an end to the
//! speech, the collected frames go to the codec decoder and come back as a
//! waveform.
//!
//! Nothing here touches a tensor: the networks sit behind [`SpeechModel`], so
//! this loop reads the same on either runtime.

use std::path::Path;

use super::{
    Sampling, SpeechProgress, Stage, SynthesisOptions,
    config::{GenerationDefaults, ModelConfig, ModelType},
    error::Qwen3TtsError,
    model::SpeechModel,
    prompt::{self, PromptSpec},
    sampler::{Rule, Sampler},
    tokenizer::TextTokenizer,
};
use crate::tts::{self, Join, Speech, Voice};

/// How many frames must be produced before the model is allowed to stop.
///
/// The reference holds generation open for two frames; without it a model that
/// opens with the end code would return silence.
const MIN_FRAMES: usize = 2;

/// Frames one text token is estimated to cost — the upper bound the chunk
/// budget is derived from.
///
/// Measured on this engine (0.6b, `serena`): a Russian paragraph runs 2.15–2.46
/// frames per token, and English, whose BPE packs about twice the characters
/// into a token, works out near 4.2. Text tokens are the steadier unit here
/// than characters, whose seconds-per-character varies about fourfold across
/// languages. The estimate is deliberately on the high side: falling short of
/// the ceiling costs one extra paragraph break, overshooting it costs speech.
const FRAMES_PER_TOKEN: f64 = 4.5;

/// The longest a single utterance is allowed to run, in seconds of speech.
///
/// The checkpoints allow far more — their `generation_config` sets the frame
/// ceiling around eleven minutes — but a piece that long is a bad bet: one
/// derailed generation costs minutes of wall time, the key/value cache grows
/// with it, and the model was trained on utterances rather than chapters. Two
/// minutes leaves ordinary prose paragraphs whole and bounds what a single
/// failure costs.
const MAX_CHUNK_SECONDS: f64 = 120.0;

/// Share of the paragraph pause used between pieces of one paragraph — those
/// are mid-thought, not a new one.
const CONTINUATION_PAUSE: f64 = 0.5;

/// How many times a piece that hit the frame ceiling may be cut down and
/// spoken again before the truncation is accepted.
const MAX_RETRIES: u8 = 2;

/// The result of one synthesis: the audio and what it took to make it.
#[derive(Debug, Clone)]
pub struct Synthesis {
    /// The synthesized audio.
    pub speech: Speech,
    /// How the voice was chosen.
    pub voice: Voice,
    /// The target language, when one was requested.
    pub language: Option<String>,
    /// Frames generated — one per 1/12.5 s of audio, summed over the pieces.
    pub frames: usize,
    /// Pieces spoken and stitched together.
    pub chunks: usize,
    /// Whether generation stopped at the frame ceiling rather than because the
    /// model finished speaking — the tail of that piece is then missing.
    pub truncated: bool,
}

/// How an over-budget paragraph is cut down: the text and the budget go in,
/// pieces within that budget come out.
///
/// The engine states the need and stays out of the how — sentence segmentation
/// is a model of its own, and the caller decides which one (and pays for it
/// only when a paragraph really is too long).
pub type SplitParagraph<'a> =
    dyn FnMut(&str, usize) -> Result<Vec<String>, Qwen3TtsError> + 'a;

/// What one piece came back as, before the pieces are joined.
struct SpokenPiece {
    speech: Speech,
    frames: usize,
    truncated: bool,
}

/// One piece of a planned run: what to speak and how it sits in the text.
struct Piece {
    text: String,
    /// Whether the piece opens a paragraph (rather than continuing one that
    /// had to be cut).
    opens_paragraph: bool,
    /// How many times this piece has already been cut down after hitting the
    /// ceiling.
    retries: u8,
}

/// A loaded engine, ready to synthesize.
pub struct Synthesizer {
    config: ModelConfig,
    defaults: GenerationDefaults,
    tokenizer: TextTokenizer,
    model: Box<dyn SpeechModel>,
    /// The newline that closes the prompt's opening role.
    newline_id: u32,
    /// Codes the talker must never emit for codebook 0.
    forbidden: Vec<u32>,
}

impl Synthesizer {
    /// Wraps a loaded runtime with everything around it: the tokenizer, the
    /// checkpoint's sampling defaults, and the codes the talker may not emit.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::InvalidModel`] when the checkpoint is missing
    /// files, malformed, or of an unsupported kind, and
    /// [`Qwen3TtsError::Tokenizer`] when the tokenizer cannot be assembled.
    pub fn load(
        model_dir: &Path,
        model: Box<dyn SpeechModel>,
    ) -> Result<Self, Qwen3TtsError> {
        let config = model.config().clone();
        if config.model_type != ModelType::CustomVoice {
            return Err(Qwen3TtsError::InvalidModel(format!(
                "this checkpoint conditions the voice in a way the engine \
                 does not serve yet ({:?}); use a CustomVoice checkpoint",
                config.model_type
            )));
        }
        let defaults = GenerationDefaults::read(model_dir)?;

        let tokenizer = TextTokenizer::load(
            &model_dir.join("vocab.json"),
            &model_dir.join("merges.txt"),
        )?;
        // The prompt's opening role ends in a newline; ask the tokenizer for
        // it rather than assuming an id.
        let newline_id = *tokenizer.encode("\n")?.first().ok_or_else(|| {
            Qwen3TtsError::Tokenizer(
                "the tokenizer does not encode a newline".into(),
            )
        })?;

        let forbidden = forbidden_codes(&config);
        Ok(Self {
            config,
            defaults,
            tokenizer,
            model,
            newline_id,
            forbidden,
        })
    }

    /// The checkpoint's configuration — its voices, languages, and geometry.
    #[must_use]
    pub fn config(&self) -> &ModelConfig { &self.config }

    /// The sampling defaults the checkpoint ships.
    #[must_use]
    pub fn defaults(&self) -> GenerationDefaults { self.defaults }

    /// The frame ceiling the checkpoint states — the point past which
    /// generation gives up on ever seeing an end-of-speech code.
    #[must_use]
    pub fn max_frames(&self) -> usize { self.defaults.max_new_tokens }

    /// Frames one piece may generate: the checkpoint's ceiling, capped at
    /// [`MAX_CHUNK_SECONDS`] of speech.
    #[must_use]
    pub fn chunk_frames(&self) -> usize {
        let seconds_per_frame = self.seconds_per_frame();
        if seconds_per_frame <= 0.0 {
            return self.max_frames();
        }
        let by_time = (MAX_CHUNK_SECONDS / seconds_per_frame) as usize;
        self.max_frames().min(by_time.max(1))
    }

    /// How much text one piece may hold, in the unit
    /// [`text_cost`](Self::text_cost) measures.
    #[must_use]
    pub fn chunk_budget(&self) -> usize {
        ((self.chunk_frames() as f64) / FRAMES_PER_TOKEN) as usize
    }

    /// What a piece of text costs against [`chunk_budget`](Self::chunk_budget):
    /// the tokens it takes in this model's own text vocabulary.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::Tokenizer`] on a tokenizer failure.
    pub fn text_cost(&self, text: &str) -> Result<usize, Qwen3TtsError> {
        Ok(self.tokenizer.encode(text)?.len())
    }

    /// Synthesizes `text` as a single utterance.
    ///
    /// Text past the frame ceiling comes back `truncated`; to speak a long text
    /// whole, use [`speak_paragraphs`](Self::speak_paragraphs).
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::TextEmpty`] for empty text,
    /// [`Qwen3TtsError::UnsupportedVoice`] or
    /// [`Qwen3TtsError::UnsupportedLanguage`] when the checkpoint cannot serve
    /// the request, and [`Qwen3TtsError::InvalidModel`] on a backend failure.
    pub fn speak(
        &mut self,
        text: &str,
        options: &SynthesisOptions,
    ) -> Result<Synthesis, Qwen3TtsError> {
        let spoken =
            self.speak_piece(text, options, options.sampling, &mut |_, _| {})?;
        Ok(Synthesis {
            speech: spoken.speech,
            voice: Voice::Preset {
                name: options.voice.clone(),
            },
            language: options.language.clone(),
            frames: spoken.frames,
            chunks: 1,
            truncated: spoken.truncated,
        })
    }

    /// Synthesizes a whole text, paragraph by paragraph, and stitches the
    /// pieces into one waveform.
    ///
    /// A paragraph over [`chunk_budget`](Self::chunk_budget) is handed to
    /// `split(text, budget)`, which must return pieces within that budget; the
    /// caller decides how (this engine has no opinion on sentence
    /// segmentation, and the splitter is loaded lazily only when a paragraph
    /// actually needs one). Should a piece still hit the frame ceiling, it is
    /// split again against half its cost and spoken anew — up to twice, after
    /// which the truncation is reported rather than chased.
    ///
    /// Each piece is sampled from its own seed, derived from the run's, so a
    /// rerun reproduces the file while neighbouring pieces do not share a
    /// stream. `progress` is called after every generated frame.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3TtsError::TextEmpty`] when nothing is left to speak, and
    /// otherwise the errors of [`speak`](Self::speak), plus whatever `split`
    /// reports.
    pub fn speak_paragraphs(
        &mut self,
        paragraphs: &[String],
        options: &SynthesisOptions,
        split: &mut SplitParagraph<'_>,
        progress: &mut dyn FnMut(SpeechProgress),
    ) -> Result<Synthesis, Qwen3TtsError> {
        let budget = self.chunk_budget();
        let mut plan = self.plan(paragraphs, budget, split)?;
        if plan.is_empty() {
            return Err(Qwen3TtsError::TextEmpty);
        }

        let total_cost = self.plan_cost(&plan)?;
        let mut spoken: Vec<Speech> = Vec::new();
        let mut pauses: Vec<f64> = Vec::new();
        let mut frames = 0usize;
        let mut truncated = false;
        let mut done_cost = 0usize;
        let mut finished_audio = 0.0f64;

        let mut index = 0;
        while index < plan.len() {
            let piece = &plan[index];
            let sampling = piece_sampling(options.sampling, index);
            let chunks = plan.len();
            let result = self.speak_piece(
                &piece.text,
                options,
                sampling,
                &mut |stage, audio| {
                    progress(SpeechProgress {
                        stage,
                        chunk: index + 1,
                        chunks,
                        finished_audio,
                        audio: finished_audio + audio,
                        done_cost,
                        total_cost,
                    });
                },
            )?;

            // A piece that ran into the ceiling is cut down and spoken again:
            // the estimate that sized it was evidently too generous, or
            // generation fell into a loop.
            if result.truncated && plan[index].retries < MAX_RETRIES {
                let piece = &plan[index];
                let half = self.text_cost(&piece.text)?.div_ceil(2);
                let parts = split(&piece.text, half)?;
                if parts.len() > 1 {
                    let opens = piece.opens_paragraph;
                    let retries = piece.retries + 1;
                    plan.splice(
                        index..=index,
                        parts.into_iter().enumerate().map(|(offset, text)| {
                            Piece {
                                text,
                                opens_paragraph: opens && offset == 0,
                                retries,
                            }
                        }),
                    );
                    continue;
                }
            }

            if index > 0 {
                pauses.push(if plan[index].opens_paragraph {
                    options.pause
                } else {
                    options.pause * CONTINUATION_PAUSE
                });
            }
            frames += result.frames;
            truncated |= result.truncated;
            done_cost += self.text_cost(&plan[index].text)?;
            finished_audio += result.speech.duration();
            spoken.push(result.speech);
            index += 1;
        }

        Ok(Synthesis {
            speech: tts::stitch(&spoken, &pauses, Join::default()),
            voice: Voice::Preset {
                name: options.voice.clone(),
            },
            language: options.language.clone(),
            frames,
            chunks: spoken.len(),
            truncated,
        })
    }

    /// Lays out the pieces to speak: one per paragraph, or several when a
    /// paragraph is over budget.
    fn plan(
        &self,
        paragraphs: &[String],
        budget: usize,
        split: &mut SplitParagraph<'_>,
    ) -> Result<Vec<Piece>, Qwen3TtsError> {
        let mut plan = Vec::new();
        for paragraph in paragraphs {
            let text = paragraph.trim();
            if text.is_empty() {
                continue;
            }
            if self.text_cost(text)? <= budget {
                plan.push(Piece {
                    text: text.to_owned(),
                    opens_paragraph: true,
                    retries: 0,
                });
                continue;
            }
            let mut parts = split(text, budget)?;
            // A splitter that returns nothing must not swallow the paragraph:
            // speaking it whole (and possibly truncating) beats losing it.
            if parts.is_empty() {
                parts.push(text.to_owned());
            }
            for (offset, part) in parts.into_iter().enumerate() {
                plan.push(Piece {
                    text: part,
                    opens_paragraph: offset == 0,
                    retries: 0,
                });
            }
        }
        Ok(plan)
    }

    /// What the whole plan costs, for the progress fraction.
    fn plan_cost(&self, plan: &[Piece]) -> Result<usize, Qwen3TtsError> {
        plan.iter().map(|piece| self.text_cost(&piece.text)).sum()
    }

    /// Speaks one piece, reporting the seconds of audio generated so far.
    fn speak_piece(
        &mut self,
        text: &str,
        options: &SynthesisOptions,
        sampling: Sampling,
        progress: &mut dyn FnMut(Stage, f64),
    ) -> Result<SpokenPiece, Qwen3TtsError> {
        if text.trim().is_empty() {
            return Err(Qwen3TtsError::TextEmpty);
        }
        let speaker_id = self.config.talker.speaker_id(&options.voice)?;
        let language_id =
            self.config.language_id(options.language.as_deref())?;

        let text_ids = self.tokenizer.encode(text)?;
        let positions = prompt::build(
            &self.config,
            &PromptSpec {
                text_ids: &text_ids,
                newline_id: self.newline_id,
                speaker_id: Some(speaker_id),
                language_id,
            },
        );

        let (frames, truncated) =
            self.generate(&positions, sampling, progress)?;
        // Decoding a finished piece is one long call with nothing to report
        // from inside it; announcing the stage is what keeps the caller's line
        // from looking stuck.
        progress(
            Stage::Decoding,
            frames.len() as f64 * self.seconds_per_frame(),
        );
        let samples = self.model.decode(&frames)?;

        Ok(SpokenPiece {
            speech: Speech {
                samples,
                sample_rate: self.model.sample_rate(),
            },
            frames: frames.len(),
            truncated,
        })
    }

    /// Runs the generation loop, returning the finished frames.
    fn generate(
        &mut self,
        positions: &[prompt::Position],
        sampling: Sampling,
        progress: &mut dyn FnMut(Stage, f64),
    ) -> Result<(Vec<Vec<u32>>, bool), Qwen3TtsError> {
        let (talker_rule, predictor_rule, repetition_penalty, seed) =
            self.rules(sampling);
        let max_frames = self.chunk_frames();
        let seconds_per_frame = self.seconds_per_frame();
        let mut sampler = Sampler::new(seed);

        let mut logits = self.model.prime(positions)?;

        let eos = self.config.talker.codec_eos_token_id;
        let mut frames: Vec<Vec<u32>> = Vec::new();
        let mut history: Vec<u32> = Vec::new();
        let mut truncated = false;

        loop {
            // Holding the speech open for the first frames keeps a model that
            // opens with the end code from returning silence.
            let mut forbidden = self.forbidden.clone();
            if frames.len() < MIN_FRAMES {
                forbidden.push(eos);
            }
            let first = sampler.pick(
                &mut logits,
                talker_rule,
                &history,
                repetition_penalty,
                &forbidden,
            );
            if first == eos {
                break;
            }
            if frames.len() >= max_frames {
                truncated = true;
                break;
            }
            history.push(first);

            // The code predictor fills the rest of this frame, conditioned on
            // the talker's state and the code it just picked.
            let residuals =
                self.model.predict_residuals(first, &mut |scores, _step| {
                    let mut scores = scores.to_vec();
                    sampler.pick(&mut scores, predictor_rule, &[], 1.0, &[])
                })?;

            let mut frame = Vec::with_capacity(residuals.len() + 1);
            frame.push(first);
            frame.extend_from_slice(&residuals);

            logits = self.model.advance(&frame)?;
            frames.push(frame);
            progress(
                Stage::Generating,
                frames.len() as f64 * seconds_per_frame,
            );
        }

        Ok((frames, truncated))
    }

    /// Seconds of audio one frame carries — what turns a frame count into a
    /// position for the progress report.
    fn seconds_per_frame(&self) -> f64 {
        let rate = self.model.sample_rate();
        if rate == 0 {
            return 0.0;
        }
        f64::from(self.model.samples_per_frame() as u32) / f64::from(rate)
    }

    /// Resolves the sampling rules for both levels.
    ///
    /// The repetition penalty is not part of sampling: the reference applies
    /// it to the logits either way, and without it greedy decoding never
    /// reaches the end-of-speech code on anything but the shortest text.
    fn rules(&self, sampling: Sampling) -> (Rule, Rule, f32, u64) {
        match sampling {
            Sampling::Greedy => (
                Rule::Greedy,
                Rule::Greedy,
                self.defaults.repetition_penalty,
                0,
            ),
            Sampling::TopK {
                top_k,
                temperature,
                repetition_penalty,
                seed,
            } => (
                Rule::TopK { top_k, temperature },
                // The code predictor keeps the checkpoint's own settings: it
                // is a separate level with its own defaults.
                Rule::TopK {
                    top_k: self.defaults.predictor_top_k,
                    temperature: self.defaults.predictor_temperature,
                },
                repetition_penalty,
                seed,
            ),
        }
    }
}

/// The odd multiplier that spreads consecutive indices across the seed space
/// (the golden-ratio constant used for exactly this).
const SEED_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

/// How piece `index` samples: the run's rule with a seed derived from the
/// run's, so the whole file reproduces from one `--seed` while no two pieces
/// draw the same sequence. Greedy has nothing to derive.
fn piece_sampling(sampling: Sampling, index: usize) -> Sampling {
    match sampling {
        Sampling::Greedy => Sampling::Greedy,
        Sampling::TopK {
            top_k,
            temperature,
            repetition_penalty,
            seed,
        } => Sampling::TopK {
            top_k,
            temperature,
            repetition_penalty,
            seed: seed.wrapping_add((index as u64).wrapping_mul(SEED_STRIDE)),
        },
    }
}

/// The codes the talker must never emit for codebook 0.
///
/// The tail of the codec vocabulary holds control ids rather than audio codes;
/// only the end-of-speech code is left reachable, so the model can stop.
fn forbidden_codes(config: &ModelConfig) -> Vec<u32> {
    let vocab = config.talker.vocab_size as u32;
    let predictor_vocab = config.talker.code_predictor.vocab_size as u32;
    (predictor_vocab..vocab)
        .filter(|code| *code != config.talker.codec_eos_token_id)
        .collect()
}

#[cfg(test)]
mod tests;
