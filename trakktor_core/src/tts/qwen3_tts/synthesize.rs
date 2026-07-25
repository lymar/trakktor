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
    Sampling, SynthesisOptions,
    config::{GenerationDefaults, ModelConfig, ModelType},
    error::Qwen3TtsError,
    model::SpeechModel,
    prompt::{self, PromptSpec},
    sampler::{Rule, Sampler},
    tokenizer::TextTokenizer,
};
use crate::tts::{Speech, Voice};

/// How many frames must be produced before the model is allowed to stop.
///
/// The reference holds generation open for two frames; without it a model that
/// opens with the end code would return silence.
const MIN_FRAMES: usize = 2;

/// The result of one synthesis: the audio and what it took to make it.
#[derive(Debug, Clone)]
pub struct Synthesis {
    /// The synthesized audio.
    pub speech: Speech,
    /// How the voice was chosen.
    pub voice: Voice,
    /// The target language, when one was requested.
    pub language: Option<String>,
    /// Frames generated — one per 1/12.5 s of audio.
    pub frames: usize,
    /// Whether generation stopped at the frame ceiling rather than because the
    /// model finished speaking — the tail of the text is then missing.
    pub truncated: bool,
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

    /// Synthesizes `text`.
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

        let (frames, truncated) = self.generate(&positions, options)?;
        let samples = self.model.decode(&frames)?;

        Ok(Synthesis {
            speech: Speech {
                samples,
                sample_rate: self.model.sample_rate(),
            },
            voice: Voice::Preset {
                name: options.voice.clone(),
            },
            language: options.language.clone(),
            frames: frames.len(),
            truncated,
        })
    }

    /// Runs the generation loop, returning the finished frames.
    fn generate(
        &mut self,
        positions: &[prompt::Position],
        options: &SynthesisOptions,
    ) -> Result<(Vec<Vec<u32>>, bool), Qwen3TtsError> {
        let (talker_rule, predictor_rule, repetition_penalty, seed) =
            self.rules(options);
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
            if frames.len() >= options.max_frames {
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
        }

        Ok((frames, truncated))
    }

    /// Resolves the sampling rules for both levels.
    ///
    /// The repetition penalty is not part of sampling: the reference applies
    /// it to the logits either way, and without it greedy decoding never
    /// reaches the end-of-speech code on anything but the shortest text.
    fn rules(&self, options: &SynthesisOptions) -> (Rule, Rule, f32, u64) {
        match options.sampling {
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
