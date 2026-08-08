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
//! **Those six are the whole of the markup.** The model has one token for each
//! of them and none for anything else, so an angle bracket in an answer that is
//! not one of the six is not a tag the reader here has to know about — it is
//! text the model read off the page.
//!
//! That markup **is** the structure, which is the whole reason to ask for it —
//! and it is not Markdown. Dropped into a document as it stands it is a wall of
//! angle brackets, so it is converted here, into one of two shapes:
//!
//! - **a pipe table** while the markup has no merged cell, which is the great
//!   majority of tables. It is the Markdown a reader expects, and it can say
//!   everything such a table says.
//! - **an HTML `<table>`** as soon as one cell covers two rows or two columns.
//!   A pipe table has neither `rowspan` nor `colspan`, so a merged cell can
//!   only come out as a blank beside the cell that carries the text — and a
//!   blank is what the markup already means by `<ecel>`. Reading a table is
//!   reading its structure; where Markdown's own table cannot hold it, the HTML
//!   that Markdown also accepts is written instead.
//!
//! Two things are deliberately **not** invented here.
//!
//! - **A header.** The pipe table has to name one — the format puts a rule
//!   under the first row and there is no way to decline — and that guess is
//!   wrong often enough to matter: a table whose column heads take two rows (a
//!   head spanning three columns, its three sub-heads beneath) gets a rule
//!   through the middle of its own heading. The HTML table is under no such
//!   obligation, so it makes every cell a `<td>` and leaves the question alone
//!   rather than answering it wrongly. The markup carries no header tag of any
//!   kind, so there is nothing better to go on.
//! - **A merge the model did not report.** The same table can come back with
//!   the row head against the first of its rows and `<ucel>` under it — the
//!   spelling this converts into a `rowspan` — or with the head against the row
//!   it is *printed* against and `<ecel>` above and below, which is a merge
//!   already lost by the time it arrives. Guessing that a blank continues the
//!   cell above it would recover the second case and wreck every table with a
//!   genuinely empty cell in it, so the blank stays a blank.
//!
//! A newline inside a cell becomes `<br>`, in both shapes: a pipe table row is
//! one line, and `<br>` is understood by every Markdown renderer and mistaken
//! for the end of the row by no parser.
//!
//! A table the model cut short mid-row still converts, and says so. The pipe
//! table fills the short row out, because a row of the wrong width is not a row
//! of a pipe table; the HTML table leaves it short, because HTML allows it and
//! stopping is what the answer did. Either way a comment marks the table: the
//! last cell of a cut-off answer stops wherever the ceiling fell — `29` where
//! the page reads `29.4` — and nothing about the number itself shows that.
//!
//! Nothing here fails on bad input; the caller falls back to printing the
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
///
/// `cut_short` says that the answer this markup came from ran into the length
/// ceiling: the table is converted either way, and marked.
pub fn to_markdown(markup: &str, cut_short: bool) -> Option<String> {
    let grid = Grid::of(&parse(markup))?;
    let mut out = if grid.merged() {
        grid.html()
    } else {
        grid.pipe()
    };
    if cut_short {
        out.push_str(CUT_SHORT);
        out.push('\n');
    }
    Some(out)
}

/// What marks a table the model did not finish. A comment, like the page
/// marker: it is there for whoever reads the Markdown source and does not
/// disturb the rendered document.
const CUT_SHORT: &str = "<!-- table cut short -->";

/// What a tag says about the position it opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tag {
    /// A cell of its own, with or without text: `<fcel>` and `<ecel>`.
    Own,
    /// Merged with the cell to its left.
    Left,
    /// Merged with the cell above.
    Up,
    /// Merged with both.
    Cross,
    /// End of row.
    Newline,
}

/// The markup's whole vocabulary.
const TAGS: [(&str, Tag); 6] = [
    ("<fcel>", Tag::Own),
    ("<ecel>", Tag::Own),
    ("<lcel>", Tag::Left),
    ("<ucel>", Tag::Up),
    ("<xcel>", Tag::Cross),
    ("<nl>", Tag::Newline),
];

/// One tagged position of one row, before the merges are resolved.
struct Slot {
    tag: Tag,
    /// The text between this tag and the next, which belongs to this cell when
    /// it has one of its own and is the empty string otherwise.
    text: String,
}

/// The table with its merges resolved: the cells, and which cell occupies each
/// position of the grid.
struct Grid {
    cells: Vec<Cell>,
    /// One entry per position, in rows of `width`. `None` where a row of the
    /// markup ran out before the widest row did.
    at: Vec<Vec<Option<usize>>>,
}

/// One cell of the table and the block of grid positions it covers.
struct Cell {
    text: String,
    top: usize,
    left: usize,
    bottom: usize,
    right: usize,
}

impl Cell {
    fn new(text: &str, row: usize, column: usize) -> Self {
        Self {
            text: text.to_string(),
            top: row,
            left: column,
            bottom: row,
            right: column,
        }
    }

    /// Notes that this cell reaches one position further.
    fn covers(&mut self, row: usize, column: usize) {
        self.bottom = self.bottom.max(row);
        self.right = self.right.max(column);
    }

    fn rows(&self) -> usize { self.bottom - self.top + 1 }

    fn columns(&self) -> usize { self.right - self.left + 1 }
}

impl Grid {
    /// Lays the tagged rows out as a grid, giving every merged position to the
    /// cell it continues.
    fn of(rows: &[Vec<Slot>]) -> Option<Self> {
        let width = rows.iter().map(Vec::len).max()?;
        if width == 0 {
            return None;
        }
        let mut cells: Vec<Cell> = Vec::new();
        let mut at: Vec<Vec<Option<usize>>> =
            vec![vec![None; width]; rows.len()];
        for (row, slots) in rows.iter().enumerate() {
            for (column, slot) in slots.iter().enumerate() {
                let left = column.checked_sub(1).and_then(|it| at[row][it]);
                let above = row.checked_sub(1).and_then(|it| at[it][column]);
                let owner = match slot.tag {
                    // A cell of its own. `<nl>` never reaches here: it ends
                    // the row rather than taking a place in it.
                    Tag::Own | Tag::Newline => None,
                    Tag::Left => left,
                    Tag::Up => above,
                    Tag::Cross => left.or(above),
                };
                // A merge with nothing to merge into — the first row, the
                // first column, a row the model opened with a continuation —
                // becomes a cell of its own instead of being dropped.
                let index = owner.unwrap_or_else(|| {
                    cells.push(Cell::new(&slot.text, row, column));
                    cells.len() - 1
                });
                cells[index].covers(row, column);
                at[row][column] = Some(index);
            }
        }
        Some(Self { cells, at })
    }

    /// Whether any cell came out covering more than its own position, which is
    /// the whole of the question "can a pipe table hold this table".
    fn merged(&self) -> bool {
        self.cells
            .iter()
            .any(|cell| cell.rows() > 1 || cell.columns() > 1)
    }

    /// The cell that begins at this position, if one does. A position a merged
    /// cell reaches into has no text of its own to write.
    fn opens(&self, row: usize, column: usize) -> Option<&Cell> {
        let cell = &self.cells[self.at[row][column]?];
        (cell.top == row && cell.left == column).then_some(cell)
    }

    /// The table as a pipe table.
    fn pipe(&self) -> String {
        let mut out = String::new();
        for (row, positions) in self.at.iter().enumerate() {
            out.push('|');
            for column in 0..positions.len() {
                let text =
                    self.opens(row, column).map_or("", |cell| &cell.text);
                let _ = write!(out, " {} |", escape(text, Form::Pipe));
            }
            out.push('\n');
            // A pipe table needs its rule after the first row, and the first
            // row is its header whether or not the table meant to have one.
            if row == 0 {
                out.push('|');
                for _ in 0..positions.len() {
                    out.push_str(" --- |");
                }
                out.push('\n');
            }
        }
        // A single row is a header with no body, which renders as an empty
        // table; give it one so the text is not lost.
        if self.at.len() == 1 {
            out.push('|');
            for _ in 0..self.at[0].len() {
                out.push_str("  |");
            }
            out.push('\n');
        }
        out
    }

    /// The table as HTML, one row per line so that the source stays readable.
    /// No blank line may fall inside it: that would end the HTML block and
    /// leave the rest of the table to be read as Markdown.
    fn html(&self) -> String {
        let mut out = String::from("<table>\n");
        for (row, positions) in self.at.iter().enumerate() {
            out.push_str("<tr>");
            for column in 0..positions.len() {
                let Some(cell) = self.opens(row, column) else {
                    continue;
                };
                out.push_str("<td");
                if cell.rows() > 1 {
                    let _ = write!(out, " rowspan=\"{}\"", cell.rows());
                }
                if cell.columns() > 1 {
                    let _ = write!(out, " colspan=\"{}\"", cell.columns());
                }
                let _ = write!(out, ">{}</td>", escape(&cell.text, Form::Html));
            }
            out.push_str("</tr>\n");
        }
        out.push_str("</table>\n");
        out
    }
}

/// Splits the markup into rows of tagged positions.
fn parse(markup: &str) -> Vec<Vec<Slot>> {
    let mut rows: Vec<Vec<Slot>> = Vec::new();
    let mut row: Vec<Slot> = Vec::new();
    let mut rest = markup;

    // Anything before the first tag is not part of any cell: the model
    // sometimes opens with a stray space or newline.
    while let Some((at, tag, length)) = find_tag(rest) {
        let after = &rest[at + length..];
        // The cell's text runs to the next tag.
        let text = match find_tag(after) {
            Some((next, _, _)) => &after[..next],
            None => after,
        };
        match tag {
            Tag::Newline => {
                if !row.is_empty() {
                    rows.push(std::mem::take(&mut row));
                }
            },
            // A merged cell carries no text of its own; the one it is merged
            // with already did.
            _ => row.push(Slot {
                tag,
                text: if tag == Tag::Own {
                    text.trim().to_string()
                } else {
                    String::new()
                },
            }),
        }
        rest = &after[text.len()..];
    }
    if !row.is_empty() {
        rows.push(row);
    }
    rows
}

/// Where the first tag of `text` begins, which one it is, and how long it is.
fn find_tag(text: &str) -> Option<(usize, Tag, usize)> {
    TAGS.iter()
        .filter_map(|(token, tag)| Some((text.find(token)?, *tag, token.len())))
        .min_by_key(|(at, _, _)| *at)
}

/// The two shapes a cell's text has to be made safe for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    Pipe,
    Html,
}

/// Makes a cell's text safe to sit in a cell of that shape.
fn escape(cell: &str, form: Form) -> String {
    let mut out = String::with_capacity(cell.len());
    let mut chars = cell.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, form) {
            ('|', Form::Pipe) => out.push_str("\\|"),
            ('&', Form::Html) => out.push_str("&amp;"),
            ('<', Form::Html) => out.push_str("&lt;"),
            ('>', Form::Html) => out.push_str("&gt;"),
            ('\n' | '\r', _) => out.push_str("<br>"),
            // The model writes a line break inside a cell as the two
            // characters `\` and `n` as often as it writes a real newline.
            ('\\', _) if chars.peek() == Some(&'n') => {
                chars.next();
                out.push_str("<br>");
            },
            _ => out.push(c),
        }
    }
    out.trim().to_string()
}
