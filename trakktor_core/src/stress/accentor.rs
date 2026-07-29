//! The word-level accentor: the character n-grams a word is looked up by, and
//! the rules that turn one word's two predictions into marked-up text.
//!
//! This is a faithful port of the reference's `AccentorNgram`, edge cases
//! included; where it departs from the reference on purpose, the comment says
//! so. No tensors here — the network's output arrives as [`WordScores`].

use super::{
    model::WordScores,
    tables::Tables,
    text::{STRESS, Token, is_vowel, lower, upper},
};

/// The confidence a head must clear for its prediction to be used. It belongs
/// to the reference's model, not to us, so it is not a knob: below it the word
/// is simply left unmarked, which is the right failure for speech synthesis —
/// a miss beats a confident mistake.
const CONFIDENCE: f32 = 0.5;

/// The rows of the embedding table a word's n-grams hit, in the reference's
/// order.
///
/// The word is wrapped in `<…>` and every substring of length `1..=len+3` is
/// looked up; unknown n-grams are skipped and repeats are kept (the bag is
/// averaged, so they carry weight). A word matching nothing at all falls back
/// to the table's single `UNK` row.
#[must_use]
pub(super) fn bag(word: &str, tables: &Tables) -> Vec<u32> {
    let mut padded = String::with_capacity(word.len() + 2);
    padded.push('<');
    padded.push_str(word);
    padded.push('>');
    let chars: Vec<char> = padded.chars().collect();
    let length = chars.len() - 2;

    let mut rows = Vec::new();
    let mut gram = String::new();
    for size in 1..=length + 3 {
        if size > chars.len() {
            break;
        }
        for start in 0..=chars.len() - size {
            gram.clear();
            gram.extend(&chars[start..start + size]);
            if let Some(row) = tables.ngrams.get(gram.as_str()) {
                rows.push(*row);
            }
        }
    }
    if rows.is_empty() {
        rows.push(tables.unk);
    }
    rows
}

/// What marking one word did.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Outcome {
    /// The word carries a stress mark now (whether we put it there or the
    /// caller did).
    pub stressed: bool,
    /// A letter `е` became `ё`.
    pub restored_yo: bool,
}

/// One marked word: its text and what happened to it.
#[derive(Debug)]
pub(super) struct Marked {
    pub text: String,
    pub outcome: Outcome,
}

/// Marks one word, given the network's prediction for it (`None` for a word the
/// rules never reach, such as punctuation).
///
/// `put_yo` turns the whole `ё` layer off — including the exception table's,
/// which is where this port **departs from the reference on purpose**: upstream
/// writes `ё` from that table even when its `put_yo` flag is off, which would
/// make our `--yo off` (documented as "your letters are not touched") false for
/// some 3 000 very ordinary words.
pub(super) fn mark(
    token: &Token,
    scores: Option<&WordScores>,
    tables: &Tables,
    put_yo: bool,
) -> Marked {
    let unchanged = |outcome: Outcome| Marked {
        text: token.raw.iter().collect(),
        outcome,
    };
    if !token.process {
        return unchanged(Outcome::default());
    }

    let has_stress = token.has_stress();
    let has_yo = token.has_yo();
    let stressed = Outcome {
        stressed: true,
        restored_yo: false,
    };

    // The caller marked everything about this word already.
    if has_stress && has_yo {
        return unchanged(stressed);
    }
    // A written `ё` is always stressed, so it needs no model to mark it. The
    // caller may have written more than one.
    if !has_stress && has_yo {
        let mut out = String::with_capacity(token.raw.len() + 2);
        for c in &token.raw {
            if lower(*c) == 'ё' {
                out.push(STRESS);
            }
            out.push(*c);
        }
        return Marked {
            text: out,
            outcome: stressed,
        };
    }

    // From here the caller has written no `ё`.
    if let Some(exception) = tables.exceptions.get(token.clean.as_str()) {
        return from_exception(token, *exception, has_stress, put_yo);
    }

    let Some(scores) = scores else {
        return unchanged(Outcome {
            stressed: has_stress,
            restored_yo: false,
        });
    };

    let mut set_stress = scores.stress_prob > CONFIDENCE && !has_stress;
    let set_yo = put_yo && scores.yo_prob > CONFIDENCE;

    let vowels = token.vowels();
    let ye = token.ye();
    if vowels.is_empty() {
        return unchanged(Outcome {
            stressed: has_stress,
            restored_yo: false,
        });
    }

    // Which vowels count as stressed. Normally the head's own pick; when the
    // caller has marked the word, the vowels on each side of every `+` — the
    // reference computes it this way so the `ё` check below has something to
    // agree with.
    let stressed_ids: Vec<usize> = if has_stress {
        split_counts(&token.lower)
    } else {
        vec![scores.stress_id]
    };
    let mut stress_positions: Vec<usize> = stressed_ids
        .iter()
        .filter_map(|index| vowels.get(*index).copied())
        .collect();
    // The `ё` head numbers the letters `е` from one; zero means "no `ё`".
    let yo_positions: Vec<usize> = [scores.yo_id]
        .iter()
        .filter(|index| **index > 0)
        .filter_map(|index| ye.get(index - 1).copied())
        .collect();

    // A restored `ё` has to agree with the stress: the letter is always
    // stressed, so a `ё` predicted anywhere else is a contradiction.
    let mut raw = token.raw.clone();
    let mut restored_yo = false;
    if set_yo {
        for position in yo_positions {
            if stress_positions.contains(&position) &&
                token.lower[position] == 'е'
            {
                raw[position] = if raw[position].is_lowercase() {
                    'ё'
                } else {
                    upper('ё')
                };
                restored_yo = true;
            }
        }
    }

    // A word with one vowel has no alternatives, so it is always marked.
    if vowels.len() == 1 {
        stress_positions = vec![vowels[0]];
        set_stress = true;
    }

    let mut text: String = raw.iter().collect();
    if !has_stress && set_stress {
        let mut out =
            String::with_capacity(text.len() + stress_positions.len());
        for (index, c) in raw.iter().enumerate() {
            if stress_positions.contains(&index) {
                out.push(STRESS);
            }
            out.push(*c);
        }
        text = out;
    }
    Marked {
        text,
        outcome: Outcome {
            stressed: has_stress || set_stress,
            restored_yo,
        },
    }
}

/// The number of vowels in each `+`-separated part of a marked word — the
/// reference's way of turning a caller's marks back into vowel ordinals.
fn split_counts(lower: &[char]) -> Vec<usize> {
    lower
        .split(|c| *c == STRESS)
        .map(|part| part.iter().copied().filter(|c| is_vowel(*c)).count())
        .collect()
}

/// Marks a word the reference keeps in its exception table. Both positions are
/// **character offsets**, and — as upstream — they are applied to the word as
/// written, not to its cleaned form; a word wrapped in quotes therefore gets
/// its mark in the wrong place. Only the bounds are guarded here, so the
/// behaviour stays the reference's.
fn from_exception(
    token: &Token,
    exception: super::tables::Exception,
    has_stress: bool,
    put_yo: bool,
) -> Marked {
    let mut raw = token.raw.clone();
    // See `mark`: upstream ignores its own `put_yo` here, we do not.
    let yo = exception.yo.filter(|_| put_yo);

    if has_stress {
        let marks: Vec<usize> = raw
            .iter()
            .enumerate()
            .filter(|(_, c)| **c == STRESS)
            .map(|(index, _)| index)
            .collect();
        let mut stripped: Vec<char> =
            raw.iter().copied().filter(|c| *c != STRESS).collect();
        let mut restored_yo = false;
        if let Some(position) = yo &&
            marks.contains(&(position + 1)) &&
            position < stripped.len()
        {
            stripped[position] = if stripped[position].is_lowercase() {
                'ё'
            } else {
                upper('ё')
            };
            restored_yo = true;
        }
        for mark in marks {
            if mark <= stripped.len() {
                stripped.insert(mark, STRESS);
            }
        }
        return Marked {
            text: stripped.iter().collect(),
            outcome: Outcome {
                stressed: true,
                restored_yo,
            },
        };
    }

    let mut restored_yo = false;
    if let Some(position) = yo &&
        position < raw.len()
    {
        raw[position] = if raw[position].is_lowercase() {
            'ё'
        } else {
            upper('ё')
        };
        restored_yo = true;
    }
    let mut text = String::with_capacity(raw.len() + 1);
    for (index, c) in raw.iter().enumerate() {
        if index == exception.stress {
            text.push(STRESS);
        }
        text.push(*c);
    }
    if exception.stress >= raw.len() {
        text.push(STRESS);
    }
    Marked {
        text,
        outcome: Outcome {
            stressed: true,
            restored_yo,
        },
    }
}

#[cfg(test)]
mod tests;
