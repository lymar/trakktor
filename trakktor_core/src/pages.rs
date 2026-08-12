//! Selecting pages of a document by number: the `--pages` value shared by
//! every feature that addresses pages — `convert pdf` picks which pages to
//! read, `pdf cut` picks which pages to keep.

use std::collections::BTreeSet;

/// A `--pages` value that does not name pages of any document.
///
/// The features that use [`Selection`] wrap this into their own error type at
/// the boundary; the message already names the flag and the offending spec.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct SelectionError(String);

/// A `--pages` value: the ranges as written, before the document is open.
///
/// Resolution is deferred because a range may be open at the end (`40-`) and
/// the length of the document is not known until it is read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    ranges: Vec<(u32, Option<u32>)>,
}

impl Selection {
    /// Parses `3`, `1-5`, `1-5,12,40-`.
    ///
    /// # Errors
    ///
    /// [`SelectionError`] when a part is not a page number or a range, when a
    /// page number is zero (pages count from one), or when a range runs
    /// backwards.
    pub fn parse(spec: &str) -> Result<Self, SelectionError> {
        let mut ranges = Vec::new();
        for part in spec.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let range = match part.split_once('-') {
                None => {
                    let page = page_number(part, spec)?;
                    (page, Some(page))
                },
                Some((first, "")) => (page_number(first, spec)?, None),
                Some((first, last)) => {
                    let (first, last) =
                        (page_number(first, spec)?, page_number(last, spec)?);
                    if last < first {
                        return Err(SelectionError(format!(
                            "--pages range runs backwards: `{part}` in \
                             `{spec}`"
                        )));
                    }
                    (first, Some(last))
                },
            };
            ranges.push(range);
        }
        if ranges.is_empty() {
            return Err(SelectionError(format!(
                "--pages selects nothing: `{spec}`"
            )));
        }
        Ok(Self { ranges })
    }

    /// The page numbers this selection stands for in a document of
    /// `page_count` pages: sorted, without repeats.
    ///
    /// # Errors
    ///
    /// [`SelectionError`] when the selection starts past the end of the
    /// document. A range that merely *runs* past the end is clamped — `40-` on
    /// a thirty-page document is a request for "the rest", and there is none.
    pub fn resolve(&self, page_count: u32) -> Result<Vec<u32>, SelectionError> {
        let mut pages = BTreeSet::new();
        for &(first, last) in &self.ranges {
            if first > page_count {
                return Err(SelectionError(format!(
                    "--pages asks for page {first}, but the document has \
                     {page_count}"
                )));
            }
            let last = last.unwrap_or(page_count).min(page_count);
            pages.extend(first..=last);
        }
        Ok(pages.into_iter().collect())
    }
}

/// Parses one page number, rejecting zero: PDF pages count from one, and `0`
/// is far more likely to be an off-by-one than an intention.
fn page_number(text: &str, spec: &str) -> Result<u32, SelectionError> {
    let page: u32 = text.trim().parse().map_err(|_| {
        SelectionError(format!(
            "--pages expects page numbers and ranges like `1-5,12,40-`, not \
             `{spec}`"
        ))
    })?;
    if page == 0 {
        return Err(SelectionError(format!(
            "--pages counts from 1, so page 0 does not exist: `{spec}`"
        )));
    }
    Ok(page)
}

#[cfg(test)]
mod tests;
