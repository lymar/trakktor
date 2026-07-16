//! Word-level timing: aligning text tokens to audio time.
//!
//! One full decoder forward yields the raw pre-softmax cross-attention
//! scores; the scores of the model's word-alignment heads are normalized,
//! median-filtered, averaged, and traced with dynamic time warping. Token
//! boundaries of words then map onto the DTW jump times. Punctuation merges
//! into its neighboring word, and segment boundaries get a set of empirical
//! corrections around pauses.

#[cfg(test)]
mod tests;

use super::{
    constants::{HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND},
    decoding::math,
    error::WhisperError,
    model::ForwardProvider,
    tokenizer::{TokenId, Tokenizer},
    transcribe::Segment,
};

/// One word with its time span and confidence.
#[derive(Debug, Clone)]
pub struct Word {
    /// The word text, usually with its leading space.
    pub word: String,
    /// Start time, seconds (rounded to centiseconds).
    pub start: f64,
    /// End time, seconds (rounded to centiseconds).
    pub end: f64,
    /// Mean probability of the word's tokens.
    pub probability: f32,
}

/// One aligned word before it is assigned to a segment.
#[derive(Debug, Clone)]
pub(crate) struct WordTiming {
    pub(crate) word: String,
    pub(crate) tokens: Vec<TokenId>,
    pub(crate) start: f64,
    pub(crate) end: f64,
    pub(crate) probability: f32,
}

/// Applies a median filter of odd width `filter_width` along each row,
/// reflect-padding the edges. Rows shorter than the padding are returned
/// unchanged.
pub(crate) fn median_filter(rows: &mut [Vec<f32>], filter_width: usize) {
    assert!(
        filter_width > 0 && filter_width % 2 == 1,
        "`filter_width` should be an odd number"
    );
    let pad = filter_width / 2;
    for row in rows.iter_mut() {
        if row.len() <= pad {
            continue;
        }
        // Reflect padding without repeating the edge sample.
        let n = row.len();
        let mut padded = Vec::with_capacity(n + 2 * pad);
        for k in 0..pad {
            padded.push(row[pad - k]);
        }
        padded.extend_from_slice(row);
        for k in 0..pad {
            padded.push(row[n - 2 - k]);
        }

        let mut window = vec![0.0f32; filter_width];
        for (i, slot) in row.iter_mut().enumerate() {
            window.copy_from_slice(&padded[i..i + filter_width]);
            window.sort_by(f32::total_cmp);
            *slot = window[pad];
        }
    }
}

/// Dynamic time warping over a cost matrix (rows = text, columns = time).
/// Returns the path as parallel row and column index sequences.
///
/// The indices are signed: the reference's backtrace can emit `-1` entries
/// at the matrix border, and downstream code tolerates them the same way.
pub(crate) fn dtw(cost_matrix: &[Vec<f32>]) -> (Vec<i64>, Vec<i64>) {
    let n = cost_matrix.len();
    let m = cost_matrix[0].len();
    let mut cost = vec![vec![f32::INFINITY; m + 1]; n + 1];
    let mut trace = vec![vec![-1i8; m + 1]; n + 1];
    cost[0][0] = 0.0;

    for j in 1..=m {
        for i in 1..=n {
            let c0 = cost[i - 1][j - 1];
            let c1 = cost[i - 1][j];
            let c2 = cost[i][j - 1];

            // The tie-breaking here mirrors the reference exactly: only a
            // strict minimum picks the diagonal or vertical move.
            let (c, t) = if c0 < c1 && c0 < c2 {
                (c0, 0)
            } else if c1 < c0 && c1 < c2 {
                (c1, 1)
            } else {
                (c2, 2)
            };

            // The reference accumulates in f64 but stores the cost in f32.
            cost[i][j] =
                (f64::from(cost_matrix[i - 1][j - 1]) + f64::from(c)) as f32;
            trace[i][j] = t;
        }
    }

    // Backtrace from the far corner.
    for slot in trace[0].iter_mut() {
        *slot = 2;
    }
    for row in trace.iter_mut() {
        row[0] = 1;
    }

    let mut i = n;
    let mut j = m;
    let mut path: Vec<(i64, i64)> = Vec::new();
    while i > 0 || j > 0 {
        path.push((i as i64 - 1, j as i64 - 1));
        match trace[i][j] {
            0 => {
                i -= 1;
                j -= 1;
            },
            1 => i -= 1,
            2 => j -= 1,
            other => unreachable!("unexpected trace value {other}"),
        }
    }
    path.reverse();
    let text_indices = path.iter().map(|&(i, _)| i).collect();
    let time_indices = path.iter().map(|&(_, j)| j).collect();
    (text_indices, time_indices)
}

/// Aligns `text_tokens` to time over the encoded window.
///
/// Returns one timing per word (the trailing end-of-text pseudo-word is
/// dropped); empty input or a split that yields nothing alignable returns an
/// empty list.
///
/// # Errors
///
/// Fails when the backend fails.
pub(crate) fn find_alignment<P: ForwardProvider>(
    provider: &mut P,
    tokenizer: &Tokenizer,
    text_tokens: &[TokenId],
    features: &P::AudioFeatures,
    num_frames: usize,
    alignment_heads: &[(usize, usize)],
) -> Result<Vec<WordTiming>, WhisperError> {
    const MEDFILT_WIDTH: usize = 7;
    const QK_SCALE: f32 = 1.0;

    if text_tokens.is_empty() {
        return Ok(Vec::new());
    }

    let sot_len = tokenizer.sot_sequence().len();
    let mut tokens: Vec<TokenId> =
        Vec::with_capacity(sot_len + 1 + text_tokens.len() + 1);
    tokens.extend_from_slice(tokenizer.sot_sequence());
    tokens.push(tokenizer.no_timestamps());
    tokens.extend_from_slice(text_tokens);
    tokens.push(tokenizer.eot());

    let (logits, cross_qk) =
        provider.forward_with_cross_qk(&tokens, features)?;

    // Probability of each text token, normalized over the text vocabulary.
    let eot = tokenizer.eot() as usize;
    let text_token_probs: Vec<f32> = text_tokens
        .iter()
        .enumerate()
        .map(|(i, &token)| {
            let row = &logits.row(0, sot_len + i)[..eot];
            math::softmax(row)[token as usize]
        })
        .collect();

    // Alignment-head scores over the window's real frames.
    let n_tokens = tokens.len();
    let n_frames = (num_frames / 2).min(cross_qk.n_frames());
    let n_heads = alignment_heads.len();
    for &(layer, head) in alignment_heads {
        if layer >= cross_qk.n_layers() || head >= cross_qk.n_heads() {
            return Err(WhisperError::InvalidModel(format!(
                "alignment head ({layer}, {head}) is outside the model's {} \
                 layers x {} heads",
                cross_qk.n_layers(),
                cross_qk.n_heads()
            )));
        }
    }

    // weights[h][t][f]: softmax over frames, then standardization over the
    // token axis (population std), then a median filter along frames.
    let mut weights: Vec<Vec<Vec<f32>>> = alignment_heads
        .iter()
        .map(|&(layer, head)| {
            let scores = cross_qk.head(layer, head);
            (0..n_tokens)
                .map(|t| {
                    let row: Vec<f32> = scores[t * cross_qk.n_frames()..
                        t * cross_qk.n_frames() + n_frames]
                        .iter()
                        .map(|&x| x * QK_SCALE)
                        .collect();
                    math::softmax(&row)
                })
                .collect()
        })
        .collect();

    for head in weights.iter_mut() {
        for f in 0..n_frames {
            let mut mean = 0.0f64;
            for t in 0..n_tokens {
                mean += f64::from(head[t][f]);
            }
            mean /= n_tokens as f64;
            let mut var = 0.0f64;
            for t in 0..n_tokens {
                let d = f64::from(head[t][f]) - mean;
                var += d * d;
            }
            let std = (var / n_tokens as f64).sqrt();
            for t in 0..n_tokens {
                head[t][f] = ((f64::from(head[t][f]) - mean) / std) as f32;
            }
        }
        median_filter(head, MEDFILT_WIDTH);
    }

    // Mean over heads, service positions sliced off; DTW over the negated
    // matrix.
    let matrix_rows = n_tokens - 1 - sot_len;
    let mut cost: Vec<Vec<f32>> = Vec::with_capacity(matrix_rows);
    for t in sot_len..n_tokens - 1 {
        let mut row = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let mut sum = 0.0f32;
            for head in &weights {
                sum += head[t][f];
            }
            row.push(-(sum / n_heads as f32));
        }
        cost.push(row);
    }
    let (text_indices, time_indices) = dtw(&cost);

    let mut with_eot = text_tokens.to_vec();
    with_eot.push(tokenizer.eot());
    let (words, word_tokens) = tokenizer.split_to_word_tokens(&with_eot);
    if word_tokens.len() <= 1 {
        return Ok(Vec::new());
    }
    let mut word_boundaries: Vec<usize> = vec![0];
    let mut acc = 0usize;
    for tokens in &word_tokens[..word_tokens.len() - 1] {
        acc += tokens.len();
        word_boundaries.push(acc);
    }

    // Jump times: the moments the DTW path advances to the next text row.
    let mut jump_times: Vec<f64> = Vec::new();
    let mut previous_text_index: Option<i64> = None;
    for (position, &text_index) in text_indices.iter().enumerate() {
        let is_jump = match previous_text_index {
            None => true,
            Some(previous) => text_index != previous,
        };
        if is_jump {
            jump_times
                .push(time_indices[position] as f64 / TOKENS_PER_SECOND as f64);
        }
        previous_text_index = Some(text_index);
    }

    let mut result = Vec::with_capacity(word_boundaries.len() - 1);
    for (index, window) in word_boundaries.windows(2).enumerate() {
        let (from, to) = (window[0], window[1]);
        let probability = {
            let span = &text_token_probs[from..to];
            let sum: f64 = span.iter().map(|&p| f64::from(p)).sum();
            (sum / span.len() as f64) as f32
        };
        result.push(WordTiming {
            word: words[index].clone(),
            tokens: word_tokens[index].clone(),
            start: jump_times[from],
            end: jump_times[to],
            probability,
        });
    }
    Ok(result)
}

/// Merges punctuation into its neighboring word: prepended marks attach to
/// the following word, appended marks to the previous one.
pub(crate) fn merge_punctuations(
    alignment: &mut [WordTiming],
    prepended: &str,
    appended: &str,
) {
    if alignment.is_empty() {
        return;
    }

    // Merge prepended punctuations.
    let mut i = alignment.len() as i64 - 2;
    let mut j = alignment.len() - 1;
    while i >= 0 {
        let idx = i as usize;
        let previous_word = alignment[idx].word.clone();
        if previous_word.starts_with(' ') &&
            prepended.contains(previous_word.trim())
        {
            // Prepend it to the following word.
            let previous_tokens = std::mem::take(&mut alignment[idx].tokens);
            alignment[idx].word.clear();
            alignment[j].word = format!("{previous_word}{}", alignment[j].word);
            let mut tokens = previous_tokens;
            tokens.extend_from_slice(&alignment[j].tokens);
            alignment[j].tokens = tokens;
        } else {
            j = idx;
        }
        i -= 1;
    }

    // Merge appended punctuations.
    let mut i = 0usize;
    let mut j = 1usize;
    while j < alignment.len() {
        let following_word = alignment[j].word.clone();
        if !alignment[i].word.ends_with(' ') &&
            appended.contains(following_word.as_str())
        {
            // Append it to the previous word.
            alignment[i].word.push_str(&following_word);
            let following_tokens = std::mem::take(&mut alignment[j].tokens);
            alignment[i].tokens.extend_from_slice(&following_tokens);
            alignment[j].word.clear();
        } else {
            i = j;
        }
        j += 1;
    }
}

/// Aligns the window's words and distributes them over `segments`, applying
/// the reference's boundary corrections. Segment start/end times move to the
/// word boundaries where that is more plausible.
///
/// # Errors
///
/// Fails when the backend fails.
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_word_timestamps<P: ForwardProvider>(
    segments: &mut [Segment],
    provider: &mut P,
    tokenizer: &Tokenizer,
    features: &P::AudioFeatures,
    num_frames: usize,
    prepend_punctuations: &str,
    append_punctuations: &str,
    mut last_speech_timestamp: f64,
    alignment_heads: &[(usize, usize)],
) -> Result<(), WhisperError> {
    if segments.is_empty() {
        return Ok(());
    }

    let text_tokens_per_segment: Vec<Vec<TokenId>> = segments
        .iter()
        .map(|segment| {
            segment
                .tokens
                .iter()
                .copied()
                .filter(|&token| token < tokenizer.eot())
                .collect()
        })
        .collect();

    let text_tokens: Vec<TokenId> =
        text_tokens_per_segment.iter().flatten().copied().collect();
    let mut alignment = find_alignment(
        provider,
        tokenizer,
        &text_tokens,
        features,
        num_frames,
        alignment_heads,
    )?;

    let word_durations: Vec<f64> = alignment
        .iter()
        .map(|timing| timing.end - timing.start)
        .filter(|&duration| duration != 0.0)
        .collect();
    let median_duration = median(&word_durations).min(0.7);
    let max_duration = median_duration * 2.0;

    // Hack: truncate long words at sentence boundaries.
    if !word_durations.is_empty() {
        let sentence_end_marks = [".", "。", "!", "！", "?", "？"];
        for i in 1..alignment.len() {
            if alignment[i].end - alignment[i].start > max_duration {
                if sentence_end_marks.contains(&alignment[i].word.as_str()) {
                    alignment[i].end = alignment[i].start + max_duration;
                } else if sentence_end_marks
                    .contains(&alignment[i - 1].word.as_str())
                {
                    alignment[i].start = alignment[i].end - max_duration;
                }
            }
        }
    }

    merge_punctuations(
        &mut alignment,
        prepend_punctuations,
        append_punctuations,
    );

    let time_offset =
        segments[0].seek as f64 * HOP_LENGTH as f64 / SAMPLE_RATE as f64;
    let mut word_index = 0usize;

    for (segment, segment_text_tokens) in
        segments.iter_mut().zip(&text_tokens_per_segment)
    {
        let mut saved_tokens = 0usize;
        let mut words: Vec<Word> = Vec::new();

        while word_index < alignment.len() &&
            saved_tokens < segment_text_tokens.len()
        {
            let timing = &alignment[word_index];
            if !timing.word.is_empty() {
                words.push(Word {
                    word: timing.word.clone(),
                    start: round2(time_offset + timing.start),
                    end: round2(time_offset + timing.end),
                    probability: timing.probability,
                });
            }
            saved_tokens += timing.tokens.len();
            word_index += 1;
        }

        // Hack: truncate long words at segment boundaries.
        if !words.is_empty() {
            // Ensure the first and second word after a pause is not longer
            // than twice the median word duration.
            if words[0].end - last_speech_timestamp > median_duration * 4.0 &&
                (words[0].end - words[0].start > max_duration ||
                    (words.len() > 1 &&
                        words[1].end - words[0].start >
                            max_duration * 2.0))
            {
                if words.len() > 1 &&
                    words[1].end - words[1].start > max_duration
                {
                    let boundary =
                        (words[1].end / 2.0).max(words[1].end - max_duration);
                    words[0].end = boundary;
                    words[1].start = boundary;
                }
                words[0].start = (words[0].end - max_duration).max(0.0);
            }

            // Prefer the segment-level start timestamp if the first word is
            // too long.
            if segment.start < words[0].end &&
                segment.start - 0.5 > words[0].start
            {
                words[0].start = (words[0].end - median_duration)
                    .min(segment.start)
                    .max(0.0);
            } else {
                segment.start = words[0].start;
            }

            // Prefer the segment-level end timestamp if the last word is too
            // long.
            let last = words.len() - 1;
            if segment.end > words[last].start &&
                segment.end + 0.5 < words[last].end
            {
                words[last].end =
                    (words[last].start + median_duration).max(segment.end);
            } else {
                segment.end = words[last].end;
            }

            last_speech_timestamp = segment.end;
        }

        segment.words = words;
    }

    Ok(())
}

/// The end of the last word in `segments`, falling back to the last
/// segment's end when no words are present.
pub(crate) fn get_end(segments: &[Segment]) -> Option<f64> {
    segments
        .iter()
        .rev()
        .flat_map(|segment| segment.words.iter().rev())
        .map(|word| word.end)
        .next()
        .or_else(|| segments.last().map(|segment| segment.end))
}

/// Rounds to centiseconds, as the reference does for word boundaries.
fn round2(value: f64) -> f64 { (value * 100.0).round() / 100.0 }

/// The median of `values`; zero for an empty list.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}
