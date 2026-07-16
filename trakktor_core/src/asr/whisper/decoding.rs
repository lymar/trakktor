//! Decoding of one 30 s window.
//!
//! Runs the full selection policy over a [`ForwardProvider`]: option
//! validation, initial-token assembly (prompt and prefix), optional language
//! detection, the sampling loop with logit filters, greedy or beam-search
//! token selection, candidate ranking, and the result diagnostics
//! (average log-probability, no-speech probability, compression ratio) that
//! the transcription loop's fallback logic consumes.

mod decoder;
mod filters;
pub(crate) mod math;
#[cfg(test)]
mod tests;

use std::io::Write as _;

use decoder::{
    BeamSearchDecoder, GreedyDecoder, MaximumLikelihoodRanker, TokenDecoder,
};
use filters::{
    ApplyTimestampRules, LogitFilter, SuppressBlank, SuppressTokens,
};

use super::{
    constants::CHUNK_LENGTH,
    error::WhisperError,
    model::{ForwardProvider, Logits},
    tokenizer::{Task, TokenId, Tokenizer},
};

/// A prompt or prefix: raw text (encoded with a leading space after
/// trimming) or ready-made tokens.
#[derive(Debug, Clone)]
pub enum PromptInput {
    /// Text to encode.
    Text(String),
    /// Pre-encoded tokens.
    Tokens(Vec<TokenId>),
}

/// Options of one window decode.
#[derive(Debug, Clone)]
pub struct DecodingOptions {
    /// Transcribe in the source language or translate into English.
    pub task: Task,
    /// Language of the audio; `None` detects it from the window.
    pub language: Option<String>,
    /// Sampling temperature: `0.0` selects greedily, higher values sample.
    pub temperature: f32,
    /// Maximum tokens to sample; defaults to half the decoder context.
    pub sample_len: Option<usize>,
    /// Number of independent trajectories when sampling (temperature > 0).
    pub best_of: Option<usize>,
    /// Number of beams when selecting greedily (temperature == 0).
    pub beam_size: Option<usize>,
    /// Beam-search patience; effectively 1.0 when unset.
    pub patience: Option<f64>,
    /// Length penalty alpha in `[0, 1]`; unset means plain length
    /// normalization.
    pub length_penalty: Option<f64>,
    /// Previous-context prompt, placed before the start sequence.
    pub prompt: Option<PromptInput>,
    /// Prefix of the current window, placed after the start sequence.
    pub prefix: Option<PromptInput>,
    /// Token ids to suppress; `-1` expands to the non-speech set. `None` or
    /// an empty list disables the filter entirely.
    pub suppress_tokens: Option<Vec<i64>>,
    /// Suppress a blank-only start (a lone space or an immediate eot).
    pub suppress_blank: bool,
    /// Sample text only, marking the sequence with `<|notimestamps|>`.
    pub without_timestamps: bool,
    /// Largest allowed initial timestamp, in seconds.
    pub max_initial_timestamp: Option<f64>,
}

impl Default for DecodingOptions {
    fn default() -> Self {
        Self {
            task: Task::Transcribe,
            language: None,
            temperature: 0.0,
            sample_len: None,
            best_of: None,
            beam_size: None,
            patience: None,
            length_penalty: None,
            prompt: None,
            prefix: None,
            suppress_tokens: Some(vec![-1]),
            suppress_blank: true,
            without_timestamps: false,
            max_initial_timestamp: Some(1.0),
        }
    }
}

/// The outcome of one window decode.
#[derive(Debug, Clone)]
pub struct DecodeResult {
    /// The language used or detected.
    pub language: String,
    /// Sampled tokens: initial tokens and the final eot are excluded,
    /// timestamp tokens are included.
    pub tokens: Vec<TokenId>,
    /// The decoded text, trimmed.
    pub text: String,
    /// Cumulative log-probability over `tokens.len() + 1` positions (the
    /// eot counts).
    pub avg_logprob: f32,
    /// Probability of `<|nospeech|>` right after the start token.
    pub no_speech_prob: f32,
    /// The temperature this result was produced with.
    pub temperature: f32,
    /// zlib compression ratio of the text — high values mean repetitive
    /// output.
    pub compression_ratio: f32,
}

/// How much text shrinks under zlib: `utf-8 bytes / compressed bytes`.
/// Repetitive (looping) output compresses far better than normal speech.
pub fn compression_ratio(text: &str) -> f32 {
    let bytes = text.as_bytes();
    let mut encoder = flate2::write::ZlibEncoder::new(
        Vec::new(),
        flate2::Compression::default(),
    );
    encoder
        .write_all(bytes)
        .expect("writing to a Vec cannot fail");
    let compressed = encoder.finish().expect("zlib in-memory never fails");
    bytes.len() as f32 / compressed.len() as f32
}

/// Detects the spoken language of an encoded window: one forward pass of the
/// start token, with everything except language tokens suppressed.
///
/// Returns the most probable language code and the full distribution over
/// the model's languages.
///
/// # Errors
///
/// Fails when the backend fails.
pub fn detect_language<P: ForwardProvider>(
    provider: &mut P,
    tokenizer: &Tokenizer,
    features: &P::AudioFeatures,
) -> Result<(&'static str, Vec<(&'static str, f32)>), WhisperError> {
    provider.begin_decode(1, features)?;
    let logits = provider.decode_step(&[tokenizer.sot()], 1);
    provider.end_decode();
    let logits = logits?;

    let mut row = logits.row(0, 0).to_vec();
    let language_tokens = tokenizer.all_language_tokens();
    let keep: std::collections::HashSet<usize> =
        language_tokens.iter().map(|&t| t as usize).collect();
    for (index, value) in row.iter_mut().enumerate() {
        if !keep.contains(&index) {
            *value = f32::NEG_INFINITY;
        }
    }

    let best = math::argmax(&row) as TokenId;
    let probs = math::softmax(&row);
    let codes = tokenizer.all_language_codes();
    let distribution: Vec<(&'static str, f32)> = language_tokens
        .iter()
        .zip(&codes)
        .map(|(&token, &code)| (code, probs[token as usize]))
        .collect();
    let language = language_tokens
        .iter()
        .position(|&token| token == best)
        .map(|index| codes[index])
        .expect("argmax of a language-masked row is a language token");
    Ok((language, distribution))
}

/// Decodes one encoded window under `options`.
///
/// The tokenizer must match the model family and carry the language and task
/// the options assume (the transcription loop constructs it once). When
/// `options.language` is `None`, the language is detected from the window
/// and the language token in the start sequence is overwritten accordingly.
///
/// # Errors
///
/// Returns [`WhisperError::InvalidOptions`] for inconsistent options and
/// backend errors as they are.
pub fn decode<P: ForwardProvider>(
    provider: &mut P,
    tokenizer: &Tokenizer,
    features: &P::AudioFeatures,
    options: &DecodingOptions,
) -> Result<DecodeResult, WhisperError> {
    verify_options(options)?;

    let (n_ctx, n_audio_ctx) = {
        let dims = provider.dims();
        (dims.n_text_ctx, dims.n_audio_ctx)
    };
    let n_group = options.beam_size.or(options.best_of).unwrap_or(1);
    let sample_len = options.sample_len.unwrap_or(n_ctx / 2);

    let sot_sequence = if options.without_timestamps {
        tokenizer.sot_sequence_including_notimestamps()
    } else {
        tokenizer.sot_sequence().to_vec()
    };
    let mut initial_tokens = build_initial_tokens(
        tokenizer,
        options,
        &sot_sequence,
        n_ctx,
        sample_len,
    );
    let sample_begin = initial_tokens.len();
    let sot_index = initial_tokens
        .iter()
        .position(|&t| t == tokenizer.sot())
        .expect("the start sequence always contains sot");

    // Detect the language if requested, overwriting the language token.
    let language = match &options.language {
        Some(language) => language.clone(),
        None => {
            let language_token =
                tokenizer.language_token().ok_or_else(|| {
                    WhisperError::InvalidOptions(
                        "this model has no language tokens, so the language \
                         cannot be detected"
                            .into(),
                    )
                })?;
            let (code, _) = detect_language(provider, tokenizer, features)?;
            debug_assert_eq!(initial_tokens[sot_index + 1], language_token);
            initial_tokens[sot_index + 1] = tokenizer
                .to_language_token(code)
                .expect("detected languages are within the model's set");
            code.to_string()
        },
    };

    let mut decoder: Box<dyn TokenDecoder> = match options.beam_size {
        Some(beam_size) => Box::new(BeamSearchDecoder::new(
            beam_size,
            tokenizer.eot(),
            options.patience,
        )?),
        None => Box::new(GreedyDecoder::new(
            options.temperature,
            tokenizer.eot(),
            None,
        )),
    };
    decoder.reset();

    let mut logit_filters: Vec<Box<dyn LogitFilter>> = Vec::new();
    if options.suppress_blank {
        logit_filters
            .push(Box::new(SuppressBlank::new(tokenizer, sample_begin)));
    }
    if matches!(&options.suppress_tokens, Some(list) if !list.is_empty()) {
        logit_filters.push(Box::new(SuppressTokens::new(build_suppress_set(
            tokenizer, options,
        )?)));
    }
    if !options.without_timestamps {
        let precision = CHUNK_LENGTH as f64 / n_audio_ctx as f64;
        let max_initial_timestamp_index = options
            .max_initial_timestamp
            .map(|max| (max / precision).round() as usize);
        logit_filters.push(Box::new(ApplyTimestampRules::new(
            tokenizer,
            sample_begin,
            max_initial_timestamp_index,
        )));
    }

    // The main sampling loop, with the session torn down on every path.
    provider.begin_decode(n_group, features)?;
    let loop_result = main_loop(
        provider,
        tokenizer,
        &initial_tokens,
        n_group,
        n_ctx,
        sample_len,
        sot_index,
        &logit_filters,
        decoder.as_mut(),
    );
    provider.end_decode();
    let (tokens, sum_logprobs, no_speech_prob) = loop_result?;

    // Final candidates per group; slice out the sampled tokens.
    let (candidates, sum_logprobs) =
        decoder.finalize(vec![tokens], vec![sum_logprobs]);
    let candidates: Vec<Vec<TokenId>> = candidates
        .into_iter()
        .next()
        .expect("single audio")
        .into_iter()
        .map(|sequence| {
            let eot_at = sequence[sample_begin..]
                .iter()
                .position(|&t| t == tokenizer.eot())
                .expect("finalized sequences always contain eot");
            sequence[sample_begin..sample_begin + eot_at].to_vec()
        })
        .collect();
    let sum_logprobs = sum_logprobs.into_iter().next().expect("single audio");

    // Pick the best candidate.
    let ranker = MaximumLikelihoodRanker::new(options.length_penalty);
    let selected = ranker
        .rank(std::slice::from_ref(&candidates), &[sum_logprobs.clone()])[0];
    let tokens = candidates
        .into_iter()
        .nth(selected)
        .expect("ranker returns a valid index");
    let text = tokenizer.decode(&tokens).trim().to_string();
    let avg_logprob = sum_logprobs[selected] / (tokens.len() as f32 + 1.0);

    Ok(DecodeResult {
        language,
        tokens,
        text: text.clone(),
        avg_logprob,
        no_speech_prob,
        temperature: options.temperature,
        compression_ratio: compression_ratio(&text),
    })
}

/// The sampling loop: forward, probe no-speech at the first step, filter,
/// select, and stop on completion or context overflow.
#[allow(clippy::too_many_arguments)]
fn main_loop<P: ForwardProvider>(
    provider: &mut P,
    tokenizer: &Tokenizer,
    initial_tokens: &[TokenId],
    n_group: usize,
    n_ctx: usize,
    sample_len: usize,
    sot_index: usize,
    logit_filters: &[Box<dyn LogitFilter>],
    decoder: &mut dyn TokenDecoder,
) -> Result<(Vec<Vec<TokenId>>, Vec<f32>, f32), WhisperError> {
    let mut tokens: Vec<Vec<TokenId>> = vec![initial_tokens.to_vec(); n_group];
    let mut sum_logprobs = vec![0.0f32; n_group];
    let mut no_speech_prob = f32::NAN;

    for step in 0..sample_len {
        let logits: Logits = if step == 0 {
            let flat: Vec<TokenId> = tokens.concat();
            provider.decode_step(&flat, n_group)?
        } else {
            let last: Vec<TokenId> = tokens
                .iter()
                .map(|sequence| *sequence.last().unwrap())
                .collect();
            provider.decode_step(&last, n_group)?
        };

        if step == 0 {
            // The no-speech probability is read once, from the distribution
            // right after the start token (of the group's first row).
            let probs = math::softmax(logits.row(0, sot_index));
            no_speech_prob = probs[tokenizer.no_speech() as usize];
        }

        let mut last_logits: Vec<Vec<f32>> = (0..n_group)
            .map(|row| logits.last_position(row).to_vec())
            .collect();
        for filter in logit_filters {
            filter.apply(&mut last_logits, &tokens);
        }

        let (completed, rearrange) =
            decoder.update(&mut tokens, &last_logits, &mut sum_logprobs)?;
        if let Some(source_indices) = rearrange {
            provider.rearrange_kv_cache(&source_indices)?;
        }
        if completed || tokens[0].len() > n_ctx {
            break;
        }
    }

    Ok((tokens, sum_logprobs, no_speech_prob))
}

fn verify_options(options: &DecodingOptions) -> Result<(), WhisperError> {
    let fail =
        |message: &str| Err(WhisperError::InvalidOptions(message.to_string()));
    if options.beam_size.is_some() && options.best_of.is_some() {
        return fail("beam_size and best_of can't be given together");
    }
    if options.temperature == 0.0 && options.best_of.is_some() {
        return fail("best_of with greedy sampling (T=0) is not compatible");
    }
    if options.patience.is_some() && options.beam_size.is_none() {
        return fail("patience requires beam_size to be given");
    }
    if let Some(alpha) = options.length_penalty {
        if !(0.0..=1.0).contains(&alpha) {
            return fail(
                "length_penalty (alpha) should be a value between 0 and 1",
            );
        }
    }
    if let Some(list) = &options.suppress_tokens {
        if list.iter().any(|&t| t < -1) {
            return fail(
                "suppress_tokens accepts token ids and the -1 sentinel",
            );
        }
    }
    Ok(())
}

/// The start sequence extended with the prefix (after it) and the prompt
/// (before it, marked with `<|startofprev|>`).
fn build_initial_tokens(
    tokenizer: &Tokenizer,
    options: &DecodingOptions,
    sot_sequence: &[TokenId],
    n_ctx: usize,
    sample_len: usize,
) -> Vec<TokenId> {
    let mut tokens = sot_sequence.to_vec();

    if let Some(prefix) = &options.prefix {
        let prefix_tokens = match prefix {
            PromptInput::Text(text) => {
                tokenizer.encode(&format!(" {}", text.trim()))
            },
            PromptInput::Tokens(tokens) => tokens.clone(),
        };
        // The reference keeps the last `n_ctx/2 - sample_len` tokens via a
        // Python slice; with the default sample length that bound is zero,
        // and `[-0:]` keeps everything.
        let max_prefix_len = n_ctx as i64 / 2 - sample_len as i64;
        tokens.extend(python_tail(prefix_tokens, max_prefix_len));
    }

    if let Some(prompt) = &options.prompt {
        let prompt_tokens = match prompt {
            PromptInput::Text(text) => {
                tokenizer.encode(&format!(" {}", text.trim()))
            },
            PromptInput::Tokens(tokens) => tokens.clone(),
        };
        let keep = n_ctx / 2 - 1;
        let tail_start = prompt_tokens.len().saturating_sub(keep);
        let mut with_prompt = vec![tokenizer.sot_prev()];
        with_prompt.extend_from_slice(&prompt_tokens[tail_start..]);
        with_prompt.extend(tokens);
        tokens = with_prompt;
    }

    tokens
}

/// `values[-count:]` with Python semantics: a positive `count` keeps that
/// many trailing elements, zero keeps everything, and a negative `count`
/// drops that many leading elements.
pub(crate) fn python_tail(values: Vec<TokenId>, count: i64) -> Vec<TokenId> {
    use std::cmp::Ordering;
    match count.cmp(&0) {
        Ordering::Greater => {
            let start = values.len().saturating_sub(count as usize);
            values[start..].to_vec()
        },
        Ordering::Equal => values,
        Ordering::Less => {
            let skip = ((-count) as usize).min(values.len());
            values[skip..].to_vec()
        },
    }
}

/// Expands the suppress option into the concrete token set: `-1` pulls in
/// the non-speech tokens, and the special tokens that must never be sampled
/// are always included.
fn build_suppress_set(
    tokenizer: &Tokenizer,
    options: &DecodingOptions,
) -> Result<Vec<TokenId>, WhisperError> {
    let raw = options.suppress_tokens.clone().unwrap_or_default();

    let mut suppress: Vec<TokenId> = Vec::new();
    if raw.contains(&-1) {
        suppress.extend(raw.iter().filter(|&&t| t >= 0).map(|&t| t as TokenId));
        suppress.extend(tokenizer.non_speech_tokens());
    } else {
        suppress.extend(raw.iter().map(|&t| t as TokenId));
    }

    suppress.extend([
        tokenizer.transcribe(),
        tokenizer.translate(),
        tokenizer.sot(),
        tokenizer.sot_prev(),
        tokenizer.sot_lm(),
        tokenizer.no_speech(),
    ]);

    suppress.sort_unstable();
    suppress.dedup();
    Ok(suppress)
}
