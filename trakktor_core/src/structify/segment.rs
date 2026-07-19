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
