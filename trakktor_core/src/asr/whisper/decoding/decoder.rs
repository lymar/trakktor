//! Token selection: greedy sampling and beam search, plus sequence ranking.
//!
//! Faithful port of the reference decoders. Sequences are kept as plain
//! token vectors, one per batch row; cumulative log-probabilities ride along
//! in a parallel array. Beam search reproduces the reference's bookkeeping
//! exactly, including its dictionary semantics: duplicate candidate
//! sequences keep their first-insertion position while the score of the
//! last producer wins, and candidate ordering is a stable sort on the score.

use std::collections::HashMap;

use rand::{SeedableRng, rngs::StdRng};

use super::{
    super::{error::WhisperError, tokenizer::TokenId},
    math,
};

/// How the next token is selected each step.
pub(crate) trait TokenDecoder {
    /// Clears any per-window state before a new decode.
    fn reset(&mut self);

    /// Appends one token to every sequence based on this step's logits.
    ///
    /// Returns whether all sequences are finished, and — under beam search —
    /// the cache-reorder indices to apply before the next forward pass.
    ///
    /// # Errors
    ///
    /// Beam search fails when the batch size is not a multiple of the beam
    /// size.
    fn update(
        &mut self,
        tokens: &mut Vec<Vec<TokenId>>,
        logits: &[Vec<f32>],
        sum_logprobs: &mut [f32],
    ) -> Result<(bool, Option<Vec<usize>>), WhisperError>;

    /// Finalizes the search: per audio, the candidate sequences (each ending
    /// with `eot`) and their cumulative log-probabilities.
    #[allow(clippy::type_complexity)]
    fn finalize(
        &mut self,
        tokens: Vec<Vec<Vec<TokenId>>>,
        sum_logprobs: Vec<Vec<f32>>,
    ) -> (Vec<Vec<Vec<TokenId>>>, Vec<Vec<f32>>);
}

/// Greedy selection: argmax at zero temperature, categorical sampling above.
pub(crate) struct GreedyDecoder {
    temperature: f32,
    eot: TokenId,
    rng: StdRng,
}

impl GreedyDecoder {
    /// `seed` fixes the sampling stream (tests); `None` seeds from entropy.
    pub(crate) fn new(
        temperature: f32,
        eot: TokenId,
        seed: Option<u64>,
    ) -> Self {
        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };
        Self {
            temperature,
            eot,
            rng,
        }
    }
}

impl TokenDecoder for GreedyDecoder {
    fn reset(&mut self) {}

    fn update(
        &mut self,
        tokens: &mut Vec<Vec<TokenId>>,
        logits: &[Vec<f32>],
        sum_logprobs: &mut [f32],
    ) -> Result<(bool, Option<Vec<usize>>), WhisperError> {
        for (row, sequence) in tokens.iter_mut().enumerate() {
            let next = if self.temperature == 0.0 {
                math::argmax(&logits[row]) as TokenId
            } else {
                let scaled: Vec<f32> =
                    logits[row].iter().map(|&x| x / self.temperature).collect();
                math::sample_categorical(&scaled, &mut self.rng) as TokenId
            };

            let logprobs = math::log_softmax(&logits[row]);
            let last = *sequence.last().expect("sequences are never empty");
            if last != self.eot {
                sum_logprobs[row] += logprobs[next as usize];
            }
            sequence.push(if last == self.eot { self.eot } else { next });
        }

        let completed = tokens
            .iter()
            .all(|sequence| *sequence.last().unwrap() == self.eot);
        Ok((completed, None))
    }

    fn finalize(
        &mut self,
        tokens: Vec<Vec<Vec<TokenId>>>,
        sum_logprobs: Vec<Vec<f32>>,
    ) -> (Vec<Vec<Vec<TokenId>>>, Vec<Vec<f32>>) {
        // Make sure each sequence has at least one eot at the end.
        let tokens = tokens
            .into_iter()
            .map(|group| {
                group
                    .into_iter()
                    .map(|mut sequence| {
                        sequence.push(self.eot);
                        sequence
                    })
                    .collect()
            })
            .collect();
        (tokens, sum_logprobs)
    }
}

/// Beam search with optional patience (the candidate pool grows to
/// `round(beam_size * patience)` finished sequences).
pub(crate) struct BeamSearchDecoder {
    beam_size: usize,
    eot: TokenId,
    max_candidates: usize,
    /// Per audio: finished sequences in insertion order with their scores.
    finished_sequences: Option<Vec<Vec<(Vec<TokenId>, f32)>>>,
}

impl BeamSearchDecoder {
    pub(crate) fn new(
        beam_size: usize,
        eot: TokenId,
        patience: Option<f64>,
    ) -> Result<Self, WhisperError> {
        let patience = patience.unwrap_or(1.0);
        let max_candidates = (beam_size as f64 * patience).round() as usize;
        if max_candidates == 0 {
            return Err(WhisperError::InvalidOptions(format!(
                "invalid beam size ({beam_size}) or patience ({patience})"
            )));
        }
        Ok(Self {
            beam_size,
            eot,
            max_candidates,
            finished_sequences: None,
        })
    }
}

impl TokenDecoder for BeamSearchDecoder {
    fn reset(&mut self) { self.finished_sequences = None; }

    fn update(
        &mut self,
        tokens: &mut Vec<Vec<TokenId>>,
        logits: &[Vec<f32>],
        sum_logprobs: &mut [f32],
    ) -> Result<(bool, Option<Vec<usize>>), WhisperError> {
        if tokens.len() % self.beam_size != 0 {
            return Err(WhisperError::InvalidOptions(format!(
                "batch of {} rows is not a multiple of beam size {}",
                tokens.len(),
                self.beam_size
            )));
        }
        let n_audio = tokens.len() / self.beam_size;
        let finished_sequences = self
            .finished_sequences
            .get_or_insert_with(|| vec![Vec::new(); n_audio]);

        let logprobs: Vec<Vec<f32>> =
            logits.iter().map(|row| math::log_softmax(row)).collect();

        let mut next_tokens: Vec<Vec<TokenId>> = Vec::new();
        let mut source_indices: Vec<usize> = Vec::new();
        let mut new_sum_logprobs: Vec<f32> = Vec::new();

        for i in 0..n_audio {
            // Candidate sequences in first-insertion order; on a duplicate
            // the score and source of the later producer win, but the
            // position does not move (dictionary semantics).
            let mut order: Vec<Vec<TokenId>> = Vec::new();
            let mut info: HashMap<Vec<TokenId>, (f32, usize)> = HashMap::new();

            for j in 0..self.beam_size {
                let idx = i * self.beam_size + j;
                for (logprob, token) in
                    math::topk(&logprobs[idx], self.beam_size + 1)
                {
                    let new_logprob = sum_logprobs[idx] + logprob;
                    let mut sequence = tokens[idx].clone();
                    sequence.push(token as TokenId);
                    if !info.contains_key(&sequence) {
                        order.push(sequence.clone());
                    }
                    info.insert(sequence, (new_logprob, idx));
                }
            }

            // Rank candidates: stable sort keeps insertion order on ties.
            let mut ranked: Vec<&Vec<TokenId>> = order.iter().collect();
            ranked.sort_by(|a, b| info[*b].0.total_cmp(&info[*a].0));

            let mut saved = 0;
            let mut newly_finished: Vec<(Vec<TokenId>, f32)> = Vec::new();
            for sequence in ranked {
                let (score, source) = info[sequence];
                if *sequence.last().unwrap() == self.eot {
                    newly_finished.push((sequence.clone(), score));
                } else {
                    new_sum_logprobs.push(score);
                    next_tokens.push(sequence.clone());
                    source_indices.push(source);
                    saved += 1;
                    if saved == self.beam_size {
                        break;
                    }
                }
            }

            // Merge newly finished sequences (already in descending score
            // order) until the candidate pool is full.
            let previously = &mut finished_sequences[i];
            for (sequence, score) in newly_finished {
                if previously.len() >= self.max_candidates {
                    break;
                }
                match previously.iter_mut().find(|(s, _)| *s == sequence) {
                    Some(entry) => entry.1 = score,
                    None => previously.push((sequence, score)),
                }
            }
        }

        *tokens = next_tokens;
        sum_logprobs.copy_from_slice(&new_sum_logprobs);

        let completed = finished_sequences
            .iter()
            .all(|sequences| sequences.len() >= self.max_candidates);
        Ok((completed, Some(source_indices)))
    }

    fn finalize(
        &mut self,
        preceding_tokens: Vec<Vec<Vec<TokenId>>>,
        sum_logprobs: Vec<Vec<f32>>,
    ) -> (Vec<Vec<Vec<TokenId>>>, Vec<Vec<f32>>) {
        // Collect all finished sequences, including patience, and add
        // unfinished ones if not enough.
        let finished = self
            .finished_sequences
            .as_mut()
            .expect("finalize follows at least one update");
        for (i, sequences) in finished.iter_mut().enumerate() {
            if sequences.len() < self.beam_size {
                // Highest cumulative log-probability first.
                let mut by_score: Vec<usize> =
                    (0..sum_logprobs[i].len()).collect();
                by_score.sort_by(|&a, &b| {
                    sum_logprobs[i][b].total_cmp(&sum_logprobs[i][a])
                });
                for j in by_score {
                    let mut sequence = preceding_tokens[i][j].clone();
                    sequence.push(self.eot);
                    let score = sum_logprobs[i][j];
                    match sequences.iter_mut().find(|(s, _)| *s == sequence) {
                        Some(entry) => entry.1 = score,
                        None => sequences.push((sequence, score)),
                    }
                    if sequences.len() >= self.beam_size {
                        break;
                    }
                }
            }
        }

        let tokens = finished
            .iter()
            .map(|sequences| sequences.iter().map(|(s, _)| s.clone()).collect())
            .collect();
        let scores = finished
            .iter()
            .map(|sequences| sequences.iter().map(|&(_, p)| p).collect())
            .collect();
        (tokens, scores)
    }
}

/// Selects the best sequence in each group by cumulative log-probability,
/// normalized by plain length or by the Google NMT length penalty.
pub(crate) struct MaximumLikelihoodRanker {
    length_penalty: Option<f64>,
}

impl MaximumLikelihoodRanker {
    pub(crate) fn new(length_penalty: Option<f64>) -> Self {
        Self { length_penalty }
    }

    /// Index of the winning sequence per audio.
    pub(crate) fn rank(
        &self,
        tokens: &[Vec<Vec<TokenId>>],
        sum_logprobs: &[Vec<f32>],
    ) -> Vec<usize> {
        tokens
            .iter()
            .zip(sum_logprobs)
            .map(|(group, scores)| {
                let normalized: Vec<f64> = group
                    .iter()
                    .zip(scores)
                    .map(|(sequence, &logprob)| {
                        let length = sequence.len() as f64;
                        let penalty = match self.length_penalty {
                            None => length,
                            Some(alpha) => ((5.0 + length) / 6.0).powf(alpha),
                        };
                        f64::from(logprob) / penalty
                    })
                    .collect();
                let mut best = 0;
                for (i, &score) in normalized.iter().enumerate() {
                    if score > normalized[best] {
                        best = i;
                    }
                }
                best
            })
            .collect()
    }
}
