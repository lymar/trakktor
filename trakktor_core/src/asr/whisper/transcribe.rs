//! The transcription loop — the anti-looping core of the engine.
//!
//! Walks the audio in 30 s windows and, per window: decodes with a
//! temperature-fallback schedule (a too-repetitive or too-improbable result
//! retries at a higher temperature, switching from beam search to sampling),
//! skips windows judged silent, splits the result into segments along
//! timestamp pairs, advances the seek strictly by the timestamps the model
//! produced, carries the accumulated text as the next window's prompt, and
//! resets that context whenever a window needed a high temperature — so a
//! failure can never propagate.
//!
//! Every window also yields a [`WindowTrace`] with the decisions taken,
//! which the differential tests compare against the reference
//! implementation.

#[cfg(test)]
mod tests;

use super::{
    constants::{
        FRAMES_PER_SECOND, HOP_LENGTH, N_FRAMES, N_SAMPLES, SAMPLE_RATE,
    },
    decoding::{self, DecodeResult, DecodingOptions, PromptInput, python_tail},
    error::WhisperError,
    model::ForwardProvider,
    tokenizer::{Task, TokenId, Tokenizer},
};

/// Options of a whole-file transcription.
///
/// The defaults mirror the reference signature; note that the reference
/// command line additionally defaults `beam_size` and `best_of` to 5.
#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// Fallback schedule: the temperatures tried in order until a window's
    /// result is accepted. Must not be empty.
    pub temperature: Vec<f32>,
    /// Treat a window as failed when the text compresses better than this.
    pub compression_ratio_threshold: Option<f32>,
    /// Treat a window as failed when the average log-probability is lower.
    pub logprob_threshold: Option<f32>,
    /// Consider a window silent when `<|nospeech|>` is more probable than
    /// this and the average log-probability is below `logprob_threshold`.
    pub no_speech_threshold: Option<f32>,
    /// Feed the accumulated text as the prompt of the next window. Disabling
    /// reduces the chance of failure loops at the cost of consistency.
    pub condition_on_previous_text: bool,
    /// Text prompt for the first window.
    pub initial_prompt: Option<String>,
    /// Prepend `initial_prompt` to every window's prompt, not just the
    /// first.
    pub carry_initial_prompt: bool,
    /// `start,end` second pairs to process; the last end defaults to the end
    /// of the audio. Empty means the whole file. Values must be
    /// non-negative.
    pub clip_timestamps: Vec<f32>,
    /// Language of the audio; `None` detects it on the first window.
    pub language: Option<String>,
    /// Transcribe or translate.
    pub task: Task,
    /// Beam width at zero temperature.
    pub beam_size: Option<usize>,
    /// Sampling trajectories at non-zero temperatures.
    pub best_of: Option<usize>,
    /// Beam-search patience.
    pub patience: Option<f64>,
    /// Length-penalty alpha in `[0, 1]`.
    pub length_penalty: Option<f64>,
    /// Token ids to suppress; `-1` expands to the non-speech set.
    pub suppress_tokens: Option<Vec<i64>>,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            temperature: vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold: Some(2.4),
            logprob_threshold: Some(-1.0),
            no_speech_threshold: Some(0.6),
            condition_on_previous_text: true,
            initial_prompt: None,
            carry_initial_prompt: false,
            clip_timestamps: Vec::new(),
            language: None,
            task: Task::Transcribe,
            beam_size: None,
            best_of: None,
            patience: None,
            length_penalty: None,
            suppress_tokens: Some(vec![-1]),
        }
    }
}

/// One transcribed segment.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Running index within the transcription.
    pub id: usize,
    /// Frame offset of the window this segment came from.
    pub seek: usize,
    /// Start time, seconds.
    pub start: f64,
    /// End time, seconds.
    pub end: f64,
    /// Segment text (empty when the segment was cleared as instantaneous or
    /// text-free).
    pub text: String,
    /// The segment's tokens, timestamps included.
    pub tokens: Vec<TokenId>,
    /// Diagnostics of the window this segment came from.
    pub temperature: f32,
    /// Average log-probability of the window.
    pub avg_logprob: f32,
    /// Compression ratio of the window's text.
    pub compression_ratio: f32,
    /// No-speech probability of the window.
    pub no_speech_prob: f32,
}

/// A whole-file transcription.
#[derive(Debug, Clone)]
pub struct Transcription {
    /// The full text (initial prompt excluded).
    pub text: String,
    /// The segments in order.
    pub segments: Vec<Segment>,
    /// The language used or detected.
    pub language: String,
    /// Duration of the audio content, seconds.
    pub duration: f64,
}

/// One decode attempt within a window's temperature fallback.
#[derive(Debug, Clone)]
pub struct FallbackAttempt {
    /// The temperature tried.
    pub temperature: f32,
    /// The attempt's sampled tokens.
    pub tokens: Vec<TokenId>,
    /// Average log-probability of the attempt.
    pub avg_logprob: f32,
    /// Compression ratio of the attempt's text.
    pub compression_ratio: f32,
    /// No-speech probability of the window.
    pub no_speech_prob: f32,
    /// Whether the thresholds rejected this attempt.
    pub needs_fallback: bool,
}

/// The decisions taken for one window — the unit of differential testing
/// against the reference implementation.
#[derive(Debug, Clone)]
pub struct WindowTrace {
    /// Frame offset before the window.
    pub seek: usize,
    /// Window start, seconds.
    pub time_offset: f64,
    /// Prompt length fed to the decoder.
    pub prompt_len: usize,
    /// Every decode attempt in schedule order; the last one was accepted.
    pub attempts: Vec<FallbackAttempt>,
    /// Whether the window was skipped as silent.
    pub should_skip: bool,
    /// Frame offset after the window.
    pub seek_after: usize,
    /// Whether the window ended in a lone trailing timestamp.
    pub single_timestamp_ending: bool,
    /// Number of consecutive-timestamp boundaries found.
    pub consecutive_count: usize,
    /// Whether the accumulated context was reset after this window.
    pub context_reset: bool,
    /// `(start, end)` of the segments the window produced.
    pub segments: Vec<(f64, f64)>,
}

/// Transcribes 16 kHz mono PCM.
///
/// # Errors
///
/// Fails on inconsistent options, an unsupported language, or a backend
/// failure.
pub fn transcribe<P: ForwardProvider>(
    provider: &mut P,
    audio: &[f32],
    options: &TranscribeOptions,
) -> Result<Transcription, WhisperError> {
    transcribe_with_trace(provider, audio, options)
        .map(|(transcription, _)| transcription)
}

/// [`transcribe`], additionally returning the per-window decision traces.
///
/// # Errors
///
/// See [`transcribe`].
pub fn transcribe_with_trace<P: ForwardProvider>(
    provider: &mut P,
    audio: &[f32],
    options: &TranscribeOptions,
) -> Result<(Transcription, Vec<WindowTrace>), WhisperError> {
    if options.temperature.is_empty() {
        return Err(WhisperError::InvalidOptions(
            "the temperature schedule must not be empty".into(),
        ));
    }
    if options.clip_timestamps.iter().any(|&ts| ts < 0.0) {
        return Err(WhisperError::InvalidOptions(
            "clip timestamps must be non-negative".into(),
        ));
    }

    let (multilingual, num_languages, n_text_ctx, n_audio_ctx, bands) = {
        let dims = provider.dims();
        (
            dims.is_multilingual(),
            dims.num_languages(),
            dims.n_text_ctx,
            dims.n_audio_ctx,
            dims.mel_bands()?,
        )
    };

    // The mel of the whole audio plus one window of silence, normalized
    // globally; windows are sliced out of it.
    let mel = decoding_mel(audio, bands);
    let content_frames = mel.n_frames() - N_FRAMES;
    let content_duration =
        content_frames as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;

    // Determine the language: given, forced for English-only models, or
    // detected on the first window.
    let language: String = match &options.language {
        Some(language) => language.clone(),
        None if !multilingual => "en".to_string(),
        None => {
            let detection_tokenizer =
                Tokenizer::new(true, num_languages, None, None)?;
            let features = provider.encode(&mel.window(0, mel.n_frames()))?;
            let (code, _) = decoding::detect_language(
                provider,
                &detection_tokenizer,
                &features,
            )?;
            code.to_string()
        },
    };
    let tokenizer = Tokenizer::new(
        multilingual,
        num_languages,
        Some(&language),
        Some(options.task),
    )?;

    // Clips: second pairs to frame ranges; the last end defaults to the end
    // of the content.
    // Ends are capped at the audio length so a clip can never point past
    // the content.
    let mut seek_points: Vec<usize> = options
        .clip_timestamps
        .iter()
        .map(|&ts| {
            let frame =
                (f64::from(ts) * FRAMES_PER_SECOND as f64).round() as usize;
            frame.min(content_frames)
        })
        .collect();
    if seek_points.is_empty() {
        seek_points.push(0);
    }
    if seek_points.len() % 2 == 1 {
        seek_points.push(content_frames);
    }
    let seek_clips: Vec<(usize, usize)> = seek_points
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect();

    // Mel frames per output token (2) and seconds per token (0.02).
    assert_eq!(
        N_FRAMES % n_audio_ctx,
        0,
        "audio context must divide frames"
    );
    let input_stride = N_FRAMES / n_audio_ctx;
    let time_precision =
        input_stride as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;

    let mut all_tokens: Vec<TokenId> = Vec::new();
    let mut all_segments: Vec<Segment> = Vec::new();
    let mut prompt_reset_since = 0usize;
    let mut traces: Vec<WindowTrace> = Vec::new();

    let mut remaining_prompt_length = n_text_ctx as i64 / 2 - 1;
    let initial_prompt_tokens: Vec<TokenId> = match &options.initial_prompt {
        Some(prompt) => {
            let tokens = tokenizer.encode(&format!(" {}", prompt.trim()));
            all_tokens.extend_from_slice(&tokens);
            remaining_prompt_length -= tokens.len() as i64;
            tokens
        },
        None => Vec::new(),
    };

    let mut clip_idx = 0usize;
    let mut seek = seek_clips[0].0;

    while clip_idx < seek_clips.len() {
        let (seek_clip_start, seek_clip_end) = seek_clips[clip_idx];
        if seek < seek_clip_start {
            seek = seek_clip_start;
        }
        if seek >= seek_clip_end {
            clip_idx += 1;
            if clip_idx < seek_clips.len() {
                seek = seek_clips[clip_idx].0;
            }
            continue;
        }

        let seek_before = seek;
        let time_offset = seek as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;
        let segment_size = N_FRAMES
            .min(content_frames - seek)
            .min(seek_clip_end - seek);
        let segment_duration =
            segment_size as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;
        let window = mel.window(seek, segment_size);

        let prompt: Vec<TokenId> = if options.carry_initial_prompt {
            let ignored = initial_prompt_tokens.len().max(prompt_reset_since);
            let remaining = python_tail(
                all_tokens[ignored..].to_vec(),
                remaining_prompt_length,
            );
            let mut prompt = initial_prompt_tokens.clone();
            prompt.extend(remaining);
            prompt
        } else {
            all_tokens[prompt_reset_since..].to_vec()
        };
        let prompt_len = prompt.len();

        let features = provider.encode(&window)?;
        let (result, attempts) = decode_with_fallback(
            provider, &tokenizer, &features, options, &language, prompt,
        )?;

        // Silence check: skip the whole window unless the text is confident
        // enough to keep despite the no-speech probability.
        if let Some(no_speech_threshold) = options.no_speech_threshold {
            let mut should_skip = result.no_speech_prob > no_speech_threshold;
            if let Some(logprob_threshold) = options.logprob_threshold {
                if result.avg_logprob > logprob_threshold {
                    should_skip = false;
                }
            }
            if should_skip {
                seek += segment_size;
                traces.push(WindowTrace {
                    seek: seek_before,
                    time_offset,
                    prompt_len,
                    attempts,
                    should_skip: true,
                    seek_after: seek,
                    single_timestamp_ending: false,
                    consecutive_count: 0,
                    context_reset: false,
                    segments: Vec::new(),
                });
                continue;
            }
        }

        // Split the window into segments along timestamp pairs and advance
        // the seek by what the model actually timestamped.
        let tokens = &result.tokens;
        let timestamp_begin = tokenizer.timestamp_begin();
        let is_timestamp: Vec<bool> =
            tokens.iter().map(|&t| t >= timestamp_begin).collect();
        let single_timestamp_ending = tokens.len() >= 2 &&
            !is_timestamp[tokens.len() - 2] &&
            is_timestamp[tokens.len() - 1];
        let consecutive: Vec<usize> = (0..tokens.len().saturating_sub(1))
            .filter(|&i| is_timestamp[i] && is_timestamp[i + 1])
            .map(|i| i + 1)
            .collect();
        let consecutive_count = consecutive.len();

        let mut current_segments: Vec<Segment> = Vec::new();
        if !consecutive.is_empty() {
            let mut slices = consecutive;
            if single_timestamp_ending {
                slices.push(tokens.len());
            }
            let mut last_slice = 0usize;
            for &current_slice in &slices {
                let sliced = &tokens[last_slice..current_slice];
                let start_pos = f64::from(sliced[0] - timestamp_begin);
                let end_pos =
                    f64::from(sliced[sliced.len() - 1] - timestamp_begin);
                current_segments.push(new_segment(
                    &tokenizer,
                    seek_before,
                    time_offset + start_pos * time_precision,
                    time_offset + end_pos * time_precision,
                    sliced.to_vec(),
                    &result,
                ));
                last_slice = current_slice;
            }

            if single_timestamp_ending {
                // A lone trailing timestamp: no speech after it.
                seek += segment_size;
            } else {
                // Ignore the unfinished tail and continue from the last
                // timestamp the model produced.
                let last_timestamp_pos =
                    (tokens[last_slice - 1] - timestamp_begin) as usize;
                seek += last_timestamp_pos * input_stride;
            }
        } else {
            let mut duration = segment_duration;
            let timestamps: Vec<TokenId> = tokens
                .iter()
                .copied()
                .filter(|&t| t >= timestamp_begin)
                .collect();
            if let Some(&last) = timestamps.last() {
                if last != timestamp_begin {
                    // No consecutive timestamps, but the window has one:
                    // trust it for the duration.
                    duration =
                        f64::from(last - timestamp_begin) * time_precision;
                }
            }
            current_segments.push(new_segment(
                &tokenizer,
                seek_before,
                time_offset,
                time_offset + duration,
                tokens.clone(),
                &result,
            ));
            seek += segment_size;
        }

        // Instantaneous or text-free segments are kept but emptied.
        for segment in &mut current_segments {
            if segment.start == segment.end || segment.text.trim().is_empty() {
                segment.text = String::new();
                segment.tokens = Vec::new();
            }
        }

        let segment_spans: Vec<(f64, f64)> = current_segments
            .iter()
            .map(|segment| (segment.start, segment.end))
            .collect();
        let first_id = all_segments.len();
        for (offset, mut segment) in current_segments.into_iter().enumerate() {
            segment.id = first_id + offset;
            all_tokens.extend_from_slice(&segment.tokens);
            all_segments.push(segment);
        }

        // The second pillar of loop prevention: a window that needed a high
        // temperature does not feed the next window's prompt.
        let context_reset =
            !options.condition_on_previous_text || result.temperature > 0.5;
        if context_reset {
            prompt_reset_since = all_tokens.len();
        }

        traces.push(WindowTrace {
            seek: seek_before,
            time_offset,
            prompt_len,
            attempts,
            should_skip: false,
            seek_after: seek,
            single_timestamp_ending,
            consecutive_count,
            context_reset,
            segments: segment_spans,
        });
    }

    let text = tokenizer.decode(&all_tokens[initial_prompt_tokens.len()..]);
    Ok((
        Transcription {
            text,
            segments: all_segments,
            language,
            duration: content_duration,
        },
        traces,
    ))
}

/// The mel of the whole signal plus one window of trailing silence.
fn decoding_mel(
    audio: &[f32],
    bands: super::feature::MelBands,
) -> super::feature::Mel {
    super::feature::log_mel_spectrogram(audio, bands, N_SAMPLES)
}

/// Decodes one window, escalating the temperature until the result passes
/// the repetition and log-probability thresholds (silence is accepted as
/// is). Returns the accepted (or final) result and every attempt.
fn decode_with_fallback<P: ForwardProvider>(
    provider: &mut P,
    tokenizer: &Tokenizer,
    features: &P::AudioFeatures,
    options: &TranscribeOptions,
    language: &str,
    prompt: Vec<TokenId>,
) -> Result<(DecodeResult, Vec<FallbackAttempt>), WhisperError> {
    let mut attempts: Vec<FallbackAttempt> = Vec::new();
    let mut decode_result: Option<DecodeResult> = None;

    for &temperature in &options.temperature {
        // At t == 0 beam search runs (best_of dropped); above, stochastic
        // sampling with best_of trajectories (beam and patience dropped).
        let decoding_options = DecodingOptions {
            task: options.task,
            language: Some(language.to_string()),
            temperature,
            sample_len: None,
            best_of: if temperature > 0.0 {
                options.best_of
            } else {
                None
            },
            beam_size: if temperature > 0.0 {
                None
            } else {
                options.beam_size
            },
            patience: if temperature > 0.0 {
                None
            } else {
                options.patience
            },
            length_penalty: options.length_penalty,
            prompt: if prompt.is_empty() {
                None
            } else {
                Some(PromptInput::Tokens(prompt.clone()))
            },
            prefix: None,
            suppress_tokens: options.suppress_tokens.clone(),
            suppress_blank: true,
            without_timestamps: false,
            max_initial_timestamp: Some(1.0),
        };

        let result =
            decoding::decode(provider, tokenizer, features, &decoding_options)?;

        let mut needs_fallback = false;
        if let Some(threshold) = options.compression_ratio_threshold {
            if result.compression_ratio > threshold {
                needs_fallback = true; // too repetitive
            }
        }
        if let Some(threshold) = options.logprob_threshold {
            if result.avg_logprob < threshold {
                needs_fallback = true; // average log probability is too low
            }
        }
        if let (Some(no_speech_threshold), Some(logprob_threshold)) =
            (options.no_speech_threshold, options.logprob_threshold)
        {
            if result.no_speech_prob > no_speech_threshold &&
                result.avg_logprob < logprob_threshold
            {
                needs_fallback = false; // silence
            }
        }

        attempts.push(FallbackAttempt {
            temperature,
            tokens: result.tokens.clone(),
            avg_logprob: result.avg_logprob,
            compression_ratio: result.compression_ratio,
            no_speech_prob: result.no_speech_prob,
            needs_fallback,
        });
        decode_result = Some(result);
        if !needs_fallback {
            break;
        }
    }

    Ok((
        decode_result.expect("the temperature schedule is not empty"),
        attempts,
    ))
}

/// Builds a segment: the text excludes special and timestamp tokens, the
/// token list keeps them.
fn new_segment(
    tokenizer: &Tokenizer,
    seek: usize,
    start: f64,
    end: f64,
    tokens: Vec<TokenId>,
    result: &DecodeResult,
) -> Segment {
    let text_tokens: Vec<TokenId> = tokens
        .iter()
        .copied()
        .filter(|&token| token < tokenizer.eot())
        .collect();
    Segment {
        id: 0, // assigned when appended to the transcription
        seek,
        start,
        end,
        text: tokenizer.decode(&text_tokens),
        tokens,
        temperature: result.temperature,
        avg_logprob: result.avg_logprob,
        compression_ratio: result.compression_ratio,
        no_speech_prob: result.no_speech_prob,
    }
}
