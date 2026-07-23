//! Windowing over long inputs, the batched-forward driver, and overlap
//! stitching — the runtime-independent core of the inference pipeline.
//!
//! Port of the reference chunking (`punctuators/data/infer_dataset.py`) and
//! overlap merge (`punctuators/collectors/pcs_collector.py`). Kept free of
//! tensors so it can be unit-tested against fixed predictions.

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

use super::model::MAX_SUBWORD_LEN;

/// The per-position predictions of the four heads for one token — the unit the
/// forward produces, the stitcher merges, and the decoder consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenPred {
    /// Pre-punctuation label id (index into `PRE_LABELS`).
    pub pre: u8,
    /// Post-punctuation label id (index into `POST_LABELS`).
    pub post: u8,
    /// Sentence-boundary flag (full stop after this token).
    pub sbd: bool,
    /// Per-character upper-case flags, indexed by character position **within
    /// the subtoken piece** (position 0 is the `▁` for word-initial pieces and
    /// is unused). Positions past the piece length are ignored.
    pub cap: [bool; MAX_SUBWORD_LEN],
}

/// A window over the token stream: the half-open content range `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub start: usize,
    pub end: usize,
}

/// Plans the overlapping windows over `n_tokens` content tokens. Each window
/// holds at most `max_content` tokens; every window after the first begins
/// `overlap` tokens earlier than the previous window's (unclamped) end, so
/// consecutive windows overlap by `overlap`. Mirrors `_tokenize_inputs`.
#[must_use]
pub fn plan_windows(
    n_tokens: usize,
    max_content: usize,
    overlap: usize,
) -> Vec<Window> {
    if n_tokens == 0 || max_content == 0 {
        return Vec::new();
    }
    let mut windows = Vec::new();
    let mut start = 0usize;
    let mut idx = 0usize;
    while start < n_tokens {
        let adjusted = if idx == 0 {
            0
        } else {
            start.saturating_sub(overlap)
        };
        let stop = adjusted + max_content;
        windows.push(Window {
            start: adjusted,
            end: stop.min(n_tokens),
        });
        start = stop;
        idx += 1;
    }
    windows
}

/// Runs `forward_batch` over all windows of the token stream and stitches the
/// overlaps — the runtime-independent driver, shared by every backend.
///
/// Each window is `[bos] + ids[start..end] + [eos]`. Windows of the same length
/// are batched together (up to `batch_size`), so padding — and therefore any
/// attention mask — is never needed. `forward_batch` receives one batch of
/// equal-length windows and returns, per window, the predictions for **every**
/// position (BOS/EOS included). Overlaps are merged by dropping `overlap/2`
/// tokens from each interior seam; the result is one [`TokenPred`] per input
/// token, paired with that token's id.
///
/// `overlap` is clamped below the window size and rounded **down to even**: the
/// seam split drops `overlap/2` from each side, so an odd overlap would leave
/// one token covered by both windows (the reference duplicates that token in
/// its output — a quirk, not a behavior worth keeping).
///
/// # Errors
///
/// Propagates any error from `forward_batch`.
pub fn windowed_predictions<E>(
    ids: &[u32],
    bos: u32,
    eos: u32,
    max_content: usize,
    overlap: usize,
    batch_size: usize,
    mut forward_batch: impl FnMut(&[Vec<u32>]) -> Result<Vec<Vec<TokenPred>>, E>,
) -> Result<Vec<(u32, TokenPred)>, E> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let overlap = overlap.min(max_content.saturating_sub(1)) & !1;
    let windows = plan_windows(ids.len(), max_content, overlap);

    // Each window's model input: BOS + content + EOS.
    let inputs: Vec<Vec<u32>> = windows
        .iter()
        .map(|w| {
            let mut v = Vec::with_capacity(w.end - w.start + 2);
            v.push(bos);
            v.extend_from_slice(&ids[w.start..w.end]);
            v.push(eos);
            v
        })
        .collect();

    // Bucket window indices by input length so every batch is uniform.
    let mut buckets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, input) in inputs.iter().enumerate() {
        buckets.entry(input.len()).or_default().push(i);
    }

    let batch_size = batch_size.max(1);
    let mut per_window: Vec<Vec<TokenPred>> = vec![Vec::new(); windows.len()];
    for indices in buckets.values() {
        for chunk in indices.chunks(batch_size) {
            let batch: Vec<Vec<u32>> =
                chunk.iter().map(|&i| inputs[i].clone()).collect();
            let outputs = forward_batch(&batch)?;
            for (output, &i) in outputs.into_iter().zip(chunk) {
                // Drop BOS (first) and EOS (last), keeping the content tokens.
                let content = output.len().saturating_sub(2);
                per_window[i] =
                    output.into_iter().skip(1).take(content).collect();
            }
        }
    }

    Ok(stitch(&windows, &per_window, ids, overlap))
}

/// Merges the per-window predictions back onto the token stream, dropping
/// `overlap/2` tokens from each interior seam so every token is predicted once,
/// by whichever window held it most centered. Port of `produce`'s merge loop.
///
/// `per_window[i]` holds the content predictions of window `i` (BOS/EOS already
/// stripped), aligned to `windows[i]`. Returns `(token id, prediction)` in
/// stream order.
#[must_use]
pub fn stitch(
    windows: &[Window],
    per_window: &[Vec<TokenPred>],
    ids: &[u32],
    overlap: usize,
) -> Vec<(u32, TokenPred)> {
    let half = overlap / 2;
    let last = windows.len().saturating_sub(1);
    let mut merged = Vec::with_capacity(ids.len());
    for (i, window) in windows.iter().enumerate() {
        let preds = &per_window[i];
        let start = if i > 0 { half } else { 0 };
        let stop = preds.len().saturating_sub(if i < last { half } else { 0 });
        for (k, pred) in preds.iter().enumerate().take(stop).skip(start) {
            let global = window.start + k;
            if let Some(&id) = ids.get(global) {
                merged.push((id, *pred));
            }
        }
    }
    merged
}
