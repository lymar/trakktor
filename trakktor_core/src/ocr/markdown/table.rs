//! Turning a document model's cell markup into a Markdown table.
//!
//! Asked for a table, the generative engine does not answer with text — it
//! answers with the cell markup its training data is written in, one tag per
//! cell and one per row:
//!
//! | tag | means |
//! |---|---|
//! | `<fcel>` | a cell, its text runs until the next tag |
//! | `<ecel>` | an empty cell |
//! | `<lcel>` | this cell is merged with the one to its **left** |
//! | `<ucel>` | …with the one **above** |
//! | `<xcel>` | …with both |
//! | `<nl>` | end of row |
//!
//! (`<ched>`, `<rhed>` and `<srow>` mark header and section rows in the same
//! family; they carry no cell of their own and are skipped here.)
//!
//! That markup **is** the structure, which is the whole reason to ask for it —
//! and it is not Markdown. Dropped into a document as it stands it is a wall of
//! angle brackets, so it is converted here.
//!
//! Two things Markdown cannot express and this conversion therefore loses:
//!
//! - **spans.** A pipe table has no `colspan`. A cell merged with its neighbour
//!   comes out as an empty cell next to the one that carries the text, which
//!   reads correctly and lays out correctly, but a reader cannot tell it was
//!   one cell.
//! - **a newline inside a cell.** A pipe table row is one line, so an internal
//!   break becomes `<br>` — which every Markdown renderer understands and no
//!   parser mistakes for the end of the row.
//!
//! A table the model cut short mid-row still converts: the short row is padded.
//! Nothing here fails on bad input — the caller falls back to printing the
//! markup when this returns `None`, which happens only when there is no markup
//! to convert.

#[cfg(test)]
mod tests;

use std::fmt::Write;

/// Whether a block's text is cell markup rather than prose.
///
/// The test is the cell tags themselves: they cannot occur in recognized text,
/// and a table the model answered as plain lines — which is what the classic
/// engine's `table` blocks hold — is left alone.
pub fn is_markup(text: &str) -> bool {
    text.contains("<fcel>") || text.contains("<ecel>")
}

/// Converts cell markup into a Markdown table, or returns `None` when there is
/// nothing table-shaped in it.
pub fn to_markdown(markup: &str) -> Option<String> {
    let rows = parse(markup);
    let width = rows.iter().map(Vec::len).max()?;
    if width == 0 {
        return None;
    }

    let mut out = String::new();
    for (at, row) in rows.iter().enumerate() {
        out.push('|');
        for column in 0..width {
            let cell = row.get(column).map(String::as_str).unwrap_or("");
            let _ = write!(out, " {} |", escape(cell));
        }
        out.push('\n');
        // A pipe table needs its rule after the first row, and the first row is
        // its header whether or not the table meant to have one.
        if at == 0 {
            out.push('|');
            for _ in 0..width {
                out.push_str(" --- |");
            }
            out.push('\n');
        }
    }
    // A single row is a header with no body, which renders as an empty table;
    // give it one so the text is not lost.
    if rows.len() == 1 {
        out.push('|');
        for _ in 0..width {
            out.push_str("  |");
        }
        out.push('\n');
    }
    Some(out)
}

/// Splits the markup into rows of cells.
fn parse(markup: &str) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut rest = markup;

    // Anything before the first tag is not part of any cell: the model
    // sometimes opens with a stray space or newline.
    while let Some(start) = rest.find('<') {
        let after = &rest[start..];
        let Some(end) = after.find('>') else { break };
        let tag = &after[1..end];
        let mut text = &after[end + 1..];
        // The cell's text runs to the next tag.
        if let Some(next) = text.find('<') {
            text = &text[..next];
        }
        match tag {
            "nl" => {
                if !row.is_empty() {
                    rows.push(std::mem::take(&mut row));
                }
            },
            "fcel" | "ecel" | "lcel" | "ucel" | "xcel" => {
                // A merged cell carries no text of its own; the one it is
                // merged with already did.
                row.push(if tag == "fcel" {
                    text.trim().to_string()
                } else {
                    String::new()
                });
            },
            // Header and section markers, and anything a later version adds.
            _ => {},
        }
        let consumed = start + end + 1;
        rest = &rest[consumed..];
    }
    if !row.is_empty() {
        rows.push(row);
    }
    rows
}

/// Makes a cell's text safe to sit between two pipes.
fn escape(cell: &str) -> String {
    let mut out = String::with_capacity(cell.len());
    let mut chars = cell.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '|' => out.push_str("\\|"),
            '\n' | '\r' => out.push_str("<br>"),
            // The model writes a line break inside a cell as the two
            // characters `\` and `n` as often as it writes a real newline.
            '\\' if chars.peek() == Some(&'n') => {
                chars.next();
                out.push_str("<br>");
            },
            _ => out.push(c),
        }
    }
    out.trim().to_string()
}
