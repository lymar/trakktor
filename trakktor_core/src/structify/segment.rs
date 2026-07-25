//! Windowing, overlap stitching, and paragraph boundaries — the pure logic of
//! the SaT inference pipeline, independent of candle.
//!
//! Port of the reference `extract` chunking/averaging (`wtpsplit/extract.py`)
//! and the paragraph split (`indices_to_sentences` in `wtpsplit/utils`). Kept
//! free of tensors so it can be unit-tested against fixed logit/probability
//! vectors.

#[cfg(test)]
mod tests;

/// Maximum tokens per window, excluding the `CLS`/`SEP` added around each one
/// (the reference caps a 512-wide block at 510 for the two special tokens).
pub const MAX_BLOCK: usize = 510;

/// A window over the token stream: the half-open token range `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub start: usize,
    pub end: usize,
}

/// The overlap-averaging weight profile across a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weighting {
    /// Every position weighted equally.
    Uniform,
    /// Triangular (`1 − |x|`) — down-weights window edges, where context is
    /// one-sided.
    Hat,
}

/// The block size actually used for `n_tokens`: `min(n_tokens, MAX_BLOCK)`, so
/// every window is full and no sequence padding is ever needed (the reference
/// shrinks the block to the text when it is short).
#[must_use]
pub fn block_size(n_tokens: usize) -> usize { n_tokens.min(MAX_BLOCK) }

/// Plans the overlapping windows over `n_tokens`, stepping by `stride`; the
/// final window is pinned to the end. Mirrors the chunk loop in `extract`.
#[must_use]
pub fn plan_windows(n_tokens: usize, stride: usize) -> Vec<Window> {
    if n_tokens == 0 {
        return Vec::new();
    }
    let block = block_size(n_tokens);
    let stride = stride.max(1);
    let mut windows = Vec::new();
    let mut j = 0;
    loop {
        let mut start = j;
        let mut end = j + block;
        let mut done = false;
        if end >= n_tokens {
            end = n_tokens;
            start = end.saturating_sub(block);
            done = true;
        }
        windows.push(Window { start, end });
        if done {
            break;
        }
        j += stride;
    }
    windows
}

/// The per-position weights for a window of `block` tokens.
#[must_use]
pub fn weights(block: usize, weighting: Weighting) -> Vec<f32> {
    match weighting {
        Weighting::Uniform => vec![1.0; block],
        Weighting::Hat => {
            if block <= 1 {
                return vec![1.0; block];
            }
            // np.linspace(-(1 - 1/block), 1 - 1/block, block); w = 1 - |x|.
            let a = -(1.0 - 1.0 / block as f32);
            let b = 1.0 - 1.0 / block as f32;
            let step = (b - a) / (block - 1) as f32;
            (0..block)
                .map(|i| 1.0 - (a + i as f32 * step).abs())
                .collect()
        },
    }
}

/// Averages the per-window, per-token boundary logits back onto the token
/// stream with `weights`, exactly as `extract` sums-then-divides.
///
/// `per_window[w]` holds `windows[w].end − windows[w].start` logits. Returns
/// one averaged logit per token (`n_tokens`).
#[must_use]
pub fn stitch(
    n_tokens: usize,
    windows: &[Window],
    per_window: &[Vec<f32>],
    weights: &[f32],
) -> Vec<f32> {
    let mut acc = vec![0f32; n_tokens];
    let mut count = vec![0f32; n_tokens];
    for (window, logits) in windows.iter().zip(per_window) {
        for (i, &logit) in logits.iter().enumerate() {
            let w = weights[i];
            acc[window.start + i] += w * logit;
            count[window.start + i] += w;
        }
    }
    for (a, c) in acc.iter_mut().zip(&count) {
        if *c != 0.0 {
            *a /= *c;
        }
    }
    acc
}

/// Runs `forward_batch` over all windows of the token stream and stitches the
/// overlaps — the runtime-independent driver of the batched window forward,
/// shared by every backend.
///
/// Each window is `[cls] + ids[start..end] + [sep]`; since the block size is
/// [`block_size`], every window is full and needs no padding. Windows run in
/// batches of `batch_size`: `forward_batch(buffer, n_batch, seq_len)` receives
/// one batch's windows flattened row-major into `buffer` and returns, per
/// window, the logits of its `seq_len − 2` real tokens (`CLS`/`SEP` already
/// dropped). Overlaps are averaged with the `weighting` profile; the result is
/// one boundary logit per token (`ids.len()` values).
pub fn windowed_logits<E>(
    ids: &[u32],
    cls_id: u32,
    sep_id: u32,
    stride: usize,
    batch_size: usize,
    weighting: Weighting,
    mut forward_batch: impl FnMut(&[u32], usize, usize) -> Result<Vec<Vec<f32>>, E>,
) -> Result<Vec<f32>, E> {
    let n_tokens = ids.len();
    if n_tokens == 0 {
        return Ok(Vec::new());
    }
    let block = block_size(n_tokens);
    let windows = plan_windows(n_tokens, stride);
    let seq_len = block + 2;
    let batch_size = batch_size.max(1);

    let mut per_window: Vec<Vec<f32>> = Vec::with_capacity(windows.len());
    let mut buffer =
        Vec::with_capacity(batch_size.min(windows.len()) * seq_len);
    for batch in windows.chunks(batch_size) {
        buffer.clear();
        for window in batch {
            buffer.push(cls_id);
            buffer.extend_from_slice(&ids[window.start..window.end]);
            buffer.push(sep_id);
        }
        per_window.extend(forward_batch(&buffer, batch.len(), seq_len)?);
    }

    let weights = weights(block, weighting);
    Ok(stitch(n_tokens, &windows, &per_window, &weights))
}

/// The logistic sigmoid in f32 (the reference upcasts to f32 for precision).
#[must_use]
pub fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

/// Maps per-token boundary logits to a per-character boundary probability,
/// placing each token's `sigmoid(logit)` on the **last character** of its span
/// (`char_probs[c1 − 1]`, as in `token_to_char_probs`); characters that are not
/// a token's final character stay `0` (`sigmoid(−inf)`).
///
/// `offsets[t]` is token `t`'s half-open character span `[c0, c1)` in the text;
/// `n_chars` is the text length in characters.
#[must_use]
pub fn char_probs(
    n_chars: usize,
    offsets: &[(usize, usize)],
    token_logits: &[f32],
) -> Vec<f32> {
    let mut probs = vec![0f32; n_chars];
    for (&(_, c1), &logit) in offsets.iter().zip(token_logits) {
        if c1 == 0 {
            continue;
        }
        let last = c1 - 1;
        if last < n_chars {
            probs[last] = sigmoid(logit);
        }
    }
    probs
}

/// A paragraph as a half-open character range `[start, end)` in the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// Splits `text` into paragraph character ranges at every position whose
/// boundary probability exceeds `threshold`. Port of `indices_to_sentences`:
/// the cut falls **after** the boundary character, then swallows any following
/// whitespace into the preceding paragraph. Ranges tile the text with no gaps.
#[must_use]
pub fn paragraph_spans(
    text: &[char],
    char_probs: &[f32],
    threshold: f32,
) -> Vec<Span> {
    let n = text.len();
    let mut spans = Vec::new();
    let mut offset = 0usize;
    let mut idx = 0usize;
    for (i, &p) in char_probs.iter().enumerate() {
        if p <= threshold {
            continue;
        }
        idx = i + 1;
        while idx < n && text[idx].is_whitespace() {
            idx += 1;
        }
        if idx > offset {
            spans.push(Span {
                start: offset,
                end: idx,
            });
        }
        offset = idx;
    }
    if idx != n {
        spans.push(Span { start: idx, end: n });
    }
    spans
}

/// Threshold-search steps. Twenty halvings resolve the interval far finer than
/// the probabilities themselves distinguish, and each step is arithmetic over
/// a vector the network already produced.
const THRESHOLD_STEPS: usize = 20;

/// Splits text into pieces that each fit `budget`, with as few cuts as
/// possible.
///
/// Two steps, and they answer different questions. **Where may a cut fall** —
/// lowering the threshold can only *add* boundaries, so "does this threshold
/// fit the budget" is monotone in it, and a binary search finds the highest
/// threshold that still fits (starting from `max_threshold`, the caller's
/// ordinary paragraph threshold — nothing coarser is wanted). Every cut is then
/// as strong a boundary as the text offers. **How many cuts are actually
/// needed** — the threshold that makes the worst piece fit usually chops the
/// rest finer than necessary, so neighbouring pieces are greedily merged back
/// while they still fit. Left to right, that is the fewest pieces those
/// boundaries admit.
///
/// When even the model's finest boundaries leave a piece over budget — one long
/// sentence, a run without punctuation — that piece is cut without the model
/// (see [`hard_split`]). The guarantee that every piece fits rests on that
/// step, not on the network.
#[must_use]
pub fn split_to_budget(
    chars: &[char],
    char_probs: &[f32],
    max_threshold: f32,
    budget: usize,
    cost: &dyn Fn(&str) -> usize,
) -> Vec<String> {
    // Fewest cuts of all is none: text that fits comes back whole.
    let whole: String = chars.iter().collect();
    let whole = whole.trim();
    if whole.is_empty() {
        return Vec::new();
    }
    if cost(whole) <= budget {
        return vec![whole.to_owned()];
    }

    let fits = |threshold: f32| -> Option<Vec<String>> {
        let pieces = pieces_at(chars, char_probs, threshold);
        pieces
            .iter()
            .all(|piece| cost(piece) <= budget)
            .then_some(pieces)
    };

    if let Some(pieces) = fits(max_threshold) {
        return merge_to_budget(pieces, budget, cost);
    }

    let (mut low, mut high) = (0.0f32, max_threshold);
    let mut best = fits(low);
    for _ in 0..THRESHOLD_STEPS {
        let middle = (low + high) / 2.0;
        match fits(middle) {
            Some(pieces) => {
                best = Some(pieces);
                low = middle;
            },
            None => high = middle,
        }
    }

    let pieces = best.unwrap_or_else(|| {
        pieces_at(chars, char_probs, 0.0)
            .iter()
            .flat_map(|piece| hard_split(piece, budget, cost))
            .collect()
    });
    merge_to_budget(pieces, budget, cost)
}

/// Merges neighbouring pieces while the result still fits, left to right.
fn merge_to_budget(
    pieces: Vec<String>,
    budget: usize,
    cost: &dyn Fn(&str) -> usize,
) -> Vec<String> {
    let mut merged: Vec<String> = Vec::with_capacity(pieces.len());
    for piece in pieces {
        let Some(last) = merged.last_mut() else {
            merged.push(piece);
            continue;
        };
        let joined = format!("{last} {piece}");
        if cost(&joined) <= budget {
            *last = joined;
        } else {
            merged.push(piece);
        }
    }
    merged
}

/// The pieces a threshold cuts the text into, trimmed and without empties.
fn pieces_at(
    chars: &[char],
    char_probs: &[f32],
    threshold: f32,
) -> Vec<String> {
    paragraph_spans(chars, char_probs, threshold)
        .into_iter()
        .map(|span| trim_span(chars, span))
        .filter(|span| span.end > span.start)
        .map(|span| chars[span.start..span.end].iter().collect())
        .collect()
}

/// Halves a piece until every part fits, cutting at the sentence end nearest
/// the middle, else the nearest space, else the middle itself. The last resort
/// when the model offers no boundary at all.
fn hard_split(
    text: &str,
    budget: usize,
    cost: &dyn Fn(&str) -> usize,
) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 2 || cost(text) <= budget {
        return vec![text.to_owned()];
    }

    let at = cut_point(&chars);
    let (head, tail) = chars.split_at(at);
    let mut pieces = hard_split(&head.iter().collect::<String>(), budget, cost);
    pieces.extend(hard_split(&tail.iter().collect::<String>(), budget, cost));
    pieces
}

/// Where to cut a piece in two: after the sentence end closest to the middle,
/// else after the closest space, else at the middle. Always strictly inside the
/// piece, so halving terminates.
fn cut_point(chars: &[char]) -> usize {
    let middle = chars.len() / 2;
    let sentence_end = |index: usize| {
        matches!(chars[index], '.' | '!' | '?' | '…' | ';' | ':') &&
            chars.get(index + 1).is_none_or(|next| next.is_whitespace())
    };
    let at = nearest(chars.len(), middle, sentence_end)
        .or_else(|| {
            nearest(chars.len(), middle, |index| chars[index].is_whitespace())
        })
        .map_or(middle, |index| index + 1);
    at.clamp(1, chars.len() - 1)
}

/// The index nearest `middle` that satisfies `matches`, scanning outward in
/// both directions.
fn nearest(
    len: usize,
    middle: usize,
    matches: impl Fn(usize) -> bool,
) -> Option<usize> {
    (0..len).find_map(|step| {
        let before = middle.checked_sub(step);
        let after = middle + step;
        before
            .filter(|&index| matches(index))
            .or_else(|| (after < len && matches(after)).then_some(after))
    })
}

/// Trims leading/trailing whitespace off a character range, returning the
/// tightened range (empty if all whitespace).
#[must_use]
pub fn trim_span(text: &[char], span: Span) -> Span {
    let mut start = span.start;
    let mut end = span.end;
    while start < end && text[start].is_whitespace() {
        start += 1;
    }
    while end > start && text[end - 1].is_whitespace() {
        end -= 1;
    }
    Span { start, end }
}
