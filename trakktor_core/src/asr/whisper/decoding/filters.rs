//! Logit filters, applied in a fixed order to each step's distributions
//! before token selection.
//!
//! `ApplyTimestampRules` is central to coherent timestamps and to loop
//! prevention: timestamps must appear in pairs, may never decrease, and each
//! segment is forced to have a nonzero length; when the probability mass on
//! timestamps outweighs every text token, a timestamp is forced.

use super::{
    super::tokenizer::{TokenId, Tokenizer},
    math,
};

/// In-place masking of the next-token distributions.
///
/// `logits` holds one vocabulary row per sequence; `tokens` are the full
/// contexts so far (including the initial tokens).
pub(crate) trait LogitFilter {
    fn apply(&self, logits: &mut [Vec<f32>], tokens: &[Vec<TokenId>]);
}

/// Forbids a blank-only start: at the first sampled position, a lone space
/// and an immediate end-of-text are suppressed.
pub(crate) struct SuppressBlank {
    suppress: Vec<TokenId>,
    sample_begin: usize,
}

impl SuppressBlank {
    pub(crate) fn new(tokenizer: &Tokenizer, sample_begin: usize) -> Self {
        let mut suppress = tokenizer.encode(" ");
        suppress.push(tokenizer.eot());
        Self {
            suppress,
            sample_begin,
        }
    }
}

impl LogitFilter for SuppressBlank {
    fn apply(&self, logits: &mut [Vec<f32>], tokens: &[Vec<TokenId>]) {
        if tokens.first().is_some_and(|t| t.len() == self.sample_begin) {
            for row in logits.iter_mut() {
                for &token in &self.suppress {
                    row[token as usize] = f32::NEG_INFINITY;
                }
            }
        }
    }
}

/// Unconditionally suppresses a fixed token set (non-speech annotations and
/// the special tokens that must never be sampled).
pub(crate) struct SuppressTokens {
    suppress: Vec<TokenId>,
}

impl SuppressTokens {
    pub(crate) fn new(suppress: Vec<TokenId>) -> Self { Self { suppress } }
}

impl LogitFilter for SuppressTokens {
    fn apply(&self, logits: &mut [Vec<f32>], _tokens: &[Vec<TokenId>]) {
        for row in logits.iter_mut() {
            for &token in &self.suppress {
                row[token as usize] = f32::NEG_INFINITY;
            }
        }
    }
}

/// The timestamp grammar and its anti-looping constraints.
pub(crate) struct ApplyTimestampRules {
    sample_begin: usize,
    eot: TokenId,
    no_timestamps: TokenId,
    timestamp_begin: TokenId,
    max_initial_timestamp_index: Option<usize>,
}

impl ApplyTimestampRules {
    pub(crate) fn new(
        tokenizer: &Tokenizer,
        sample_begin: usize,
        max_initial_timestamp_index: Option<usize>,
    ) -> Self {
        Self {
            sample_begin,
            eot: tokenizer.eot(),
            no_timestamps: tokenizer.no_timestamps(),
            timestamp_begin: tokenizer.timestamp_begin(),
            max_initial_timestamp_index,
        }
    }
}

impl LogitFilter for ApplyTimestampRules {
    fn apply(&self, logits: &mut [Vec<f32>], tokens: &[Vec<TokenId>]) {
        let ts_begin = self.timestamp_begin as usize;

        // `<|notimestamps|>` is expressed via the option, never sampled.
        for row in logits.iter_mut() {
            row[self.no_timestamps as usize] = f32::NEG_INFINITY;
        }

        for (row, context) in logits.iter_mut().zip(tokens) {
            let sampled = &context[self.sample_begin.min(context.len())..];
            let last_was_timestamp =
                sampled.last().is_some_and(|&t| t >= self.timestamp_begin);
            let penultimate_was_timestamp = sampled.len() < 2 ||
                sampled[sampled.len() - 2] >= self.timestamp_begin;

            // Timestamps come in pairs: after a closing timestamp only text
            // or eot may follow; after an opening one, only a timestamp.
            if last_was_timestamp {
                if penultimate_was_timestamp {
                    for x in &mut row[ts_begin..] {
                        *x = f32::NEG_INFINITY;
                    }
                } else {
                    for x in &mut row[..self.eot as usize] {
                        *x = f32::NEG_INFINITY;
                    }
                }
            }

            // Timestamps never decrease, and each segment must have nonzero
            // length — the explicit anti-looping constraint.
            if let Some(&last_timestamp) = sampled
                .iter()
                .filter(|&&t| t >= self.timestamp_begin)
                .next_back()
            {
                let timestamp_last =
                    if last_was_timestamp && !penultimate_was_timestamp {
                        last_timestamp
                    } else {
                        last_timestamp + 1
                    };
                for x in &mut row[ts_begin..timestamp_last as usize] {
                    *x = f32::NEG_INFINITY;
                }
            }
        }

        if tokens.first().is_some_and(|t| t.len() == self.sample_begin) {
            // The window must start with a timestamp...
            for row in logits.iter_mut() {
                for x in &mut row[..ts_begin] {
                    *x = f32::NEG_INFINITY;
                }
                // ...and not one beyond the initial-timestamp limit.
                if let Some(max_index) = self.max_initial_timestamp_index {
                    let last_allowed = ts_begin + max_index;
                    if last_allowed + 1 < row.len() {
                        for x in &mut row[last_allowed + 1..] {
                            *x = f32::NEG_INFINITY;
                        }
                    }
                }
            }
        }

        // When the total probability of timestamps outweighs every single
        // text token, force a timestamp.
        for row in logits.iter_mut() {
            let logprobs = math::log_softmax(row);
            let timestamp_logprob = math::logsumexp(&logprobs[ts_begin..]);
            let max_text_token_logprob = logprobs[..ts_begin]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            if timestamp_logprob > max_text_token_logprob {
                for x in &mut row[..ts_begin] {
                    *x = f32::NEG_INFINITY;
                }
            }
        }
    }
}
