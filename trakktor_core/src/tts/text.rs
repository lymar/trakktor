//! Turning an input document into the paragraphs a run speaks.
//!
//! Paragraph boundaries are read off the document itself — no model needed.
//! Plain text puts one paragraph per line; Markdown separates them with blank
//! lines and carries markup that has to come off first, because hashes and
//! asterisks either get spoken aloud or wreck the prosody.
//!
//! The Markdown cleanup is deliberately shallow: it removes *markup*, not
//! content. Tables and code blocks keep their text — deciding that the user did
//! not want to hear them is not this module's call.

use std::path::Path;

/// How the input text is laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextFormat {
    /// Decide from the source's extension, then from the text itself.
    #[default]
    Auto,
    /// Plain text: one paragraph per line, no markup.
    Plain,
    /// Markdown: paragraphs separated by blank lines, markup stripped.
    Markdown,
}

impl TextFormat {
    /// Resolves [`Auto`](Self::Auto) against the source path and the text;
    /// the other two answer with themselves. Never returns `Auto`.
    ///
    /// The extension decides when it says anything at all. Otherwise the text
    /// does: markup or blank-line separation means Markdown, and so does an
    /// undecidable case — a paragraph-per-line reading of wrapped prose chops
    /// it into fragments, while treating clean text as Markdown changes almost
    /// nothing.
    #[must_use]
    pub fn resolve(self, source: Option<&Path>, text: &str) -> TextFormat {
        if self != TextFormat::Auto {
            return self;
        }
        if let Some(format) = source.and_then(by_extension) {
            return format;
        }
        by_content(text)
    }

    /// The name this format is known by on the command line.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            TextFormat::Auto => "auto",
            TextFormat::Plain => "txt",
            TextFormat::Markdown => "md",
        }
    }
}

/// The format a file name claims, when it claims one.
fn by_extension(path: &Path) -> Option<TextFormat> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    match extension.as_str() {
        "txt" | "text" => Some(TextFormat::Plain),
        "md" | "markdown" | "mdown" | "mkd" | "mdx" => {
            Some(TextFormat::Markdown)
        },
        _ => None,
    }
}

/// Line length past which a break is taken to end a paragraph rather than to
/// wrap one. Below it, breaks are far more likely to be hard wrapping.
const WRAPPED_LINE_CHARS: usize = 120;

/// The format the text itself suggests.
fn by_content(text: &str) -> TextFormat {
    let lines: Vec<&str> = text.lines().collect();
    if lines.iter().any(|line| is_markup(line)) {
        return TextFormat::Markdown;
    }
    // A blank line is how Markdown separates paragraphs; plain text that uses
    // them is read the same way either path.
    let filled: Vec<&&str> = lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if filled.len() < lines.len() {
        return TextFormat::Markdown;
    }
    // No blank lines: long lines are paragraphs of their own, short ones are
    // wrapped prose that must not be cut at every break.
    let long = filled.len() > 1 &&
        filled
            .iter()
            .all(|line| line.chars().count() >= WRAPPED_LINE_CHARS);
    if long {
        TextFormat::Plain
    } else {
        TextFormat::Markdown
    }
}

/// Whether a line opens with something only markup would put there.
fn is_markup(line: &str) -> bool {
    let trimmed = line.trim_start();
    heading_text(trimmed).is_some() ||
        list_item(trimmed).is_some() ||
        trimmed.starts_with("> ") ||
        trimmed.starts_with("```") ||
        trimmed.starts_with("~~~") ||
        trimmed.starts_with('|') ||
        trimmed.contains("**") ||
        // An inline link, the one markup form that rarely opens a line.
        (trimmed.contains("](") && trimmed.contains('['))
}

/// Splits `text` into the paragraphs to speak, in order.
///
/// Each paragraph comes out as running prose: line breaks and repeated spaces
/// inside it collapse to single spaces, because the models are trained on
/// single-line text and raw breaks only confuse their prosody. Empty
/// paragraphs are dropped.
#[must_use]
pub fn paragraphs(text: &str, format: TextFormat) -> Vec<String> {
    match format.resolve(None, text) {
        TextFormat::Plain => plain_paragraphs(text),
        _ => markdown_paragraphs(text),
    }
}

/// One paragraph per line.
fn plain_paragraphs(text: &str) -> Vec<String> {
    text.lines()
        .map(collapse)
        .filter(|line| !line.is_empty())
        .collect()
}

/// Collapses every run of whitespace to a single space and trims the ends.
#[must_use]
pub fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Paragraphs of a Markdown document, with the markup taken off.
///
/// Blank lines separate paragraphs; a heading or a list item also starts one,
/// so a list is read as several utterances rather than one breathless run.
fn markdown_paragraphs(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut fence: Option<char> = None;
    let mut in_comment = false;

    let flush = |current: &mut String, out: &mut Vec<String>| {
        let paragraph = collapse(current);
        current.clear();
        if !paragraph.is_empty() {
            out.push(paragraph);
        }
    };

    let lines: Vec<&str> = text.lines().collect();
    for line in &lines[front_matter_end(&lines)..] {
        let line = *line;
        let trimmed = line.trim();

        // Inside a fenced block the text is kept but not cleaned: markup
        // characters there are content.
        if let Some(marker) = fence {
            if is_fence(trimmed).is_some_and(|found| found == marker) {
                fence = None;
                flush(&mut current, &mut out);
            } else {
                current.push('\n');
                current.push_str(line);
            }
            continue;
        }
        if let Some(marker) = is_fence(trimmed) {
            fence = Some(marker);
            flush(&mut current, &mut out);
            continue;
        }

        // A line left empty by an HTML comment reads as a blank one, which is
        // what a comment between two paragraphs should be.
        let line = strip_comments(line, &mut in_comment);
        let trimmed = line.trim();

        if trimmed.is_empty() ||
            is_thematic_break(trimmed) ||
            is_table_rule(trimmed) ||
            is_link_definition(trimmed)
        {
            flush(&mut current, &mut out);
            continue;
        }

        let body = strip_quote(trimmed);
        if let Some(heading) = heading_text(body) {
            flush(&mut current, &mut out);
            current.push_str(&clean_inline(heading));
            flush(&mut current, &mut out);
            continue;
        }
        if let Some(item) = list_item(body) {
            flush(&mut current, &mut out);
            current.push_str(&clean_inline(item));
            continue;
        }

        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(&clean_inline(body));
    }
    flush(&mut current, &mut out);
    out
}

/// The first line past a YAML front-matter block — metadata, never speech.
/// Zero when the document has none: only a *closed* block counts, an unclosed
/// `---` was a thematic break with ordinary text under it.
fn front_matter_end(lines: &[&str]) -> usize {
    if lines.first().map(|line| line.trim_end()) != Some("---") {
        return 0;
    }
    lines
        .iter()
        .skip(1)
        .position(|line| matches!(line.trim_end(), "---" | "..."))
        .map_or(0, |offset| offset + 2)
}

/// The fence character of a code-fence line (``` or ~~~), if it is one.
fn is_fence(trimmed: &str) -> Option<char> {
    for marker in ['`', '~'] {
        let run = trimmed.chars().take_while(|c| *c == marker).count();
        if run >= 3 {
            return Some(marker);
        }
    }
    None
}

/// Whether the line is a horizontal rule or a setext underline — a separator
/// with nothing to say.
fn is_thematic_break(trimmed: &str) -> bool {
    for marker in ['-', '*', '_', '='] {
        let run = trimmed.chars().filter(|c| *c == marker).count();
        if run >= 3 && trimmed.chars().all(|c| c == marker || c.is_whitespace())
        {
            return true;
        }
    }
    false
}

/// Whether the line is a table's header rule (`|---|:--:|`), which is layout
/// rather than content.
fn is_table_rule(trimmed: &str) -> bool {
    trimmed.contains('-') &&
        trimmed.starts_with('|') &&
        trimmed
            .chars()
            .all(|c| matches!(c, '|' | '-' | ':' | ' ' | '\t'))
}

/// Whether the line defines a reference link (`[ref]: https://…`) — markup
/// that carries no prose.
fn is_link_definition(trimmed: &str) -> bool {
    let Some(rest) = trimmed.strip_prefix('[') else {
        return false;
    };
    rest.find("]:")
        .is_some_and(|end| !rest[..end].contains('['))
}

/// The text of an ATX heading (`## Title ##`), if the line is one.
fn heading_text(trimmed: &str) -> Option<&str> {
    let hashes = trimmed.chars().take_while(|c| *c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &trimmed[hashes..];
    if !rest.is_empty() && !rest.starts_with([' ', '\t']) {
        return None;
    }
    Some(rest.trim().trim_end_matches('#').trim_end())
}

/// The text of a list item (`- `, `* `, `+ `, `1. `, `1) `), if the line is
/// one.
fn list_item(trimmed: &str) -> Option<&str> {
    if let Some(rest) = trimmed.strip_prefix(['-', '*', '+']) {
        // A bare marker with no text is a rule, not an item.
        return rest.starts_with([' ', '\t']).then(|| rest.trim_start());
    }
    let digits = trimmed.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 {
        return None;
    }
    let rest = &trimmed[digits..];
    let rest = rest.strip_prefix(['.', ')'])?;
    rest.starts_with([' ', '\t']).then(|| rest.trim_start())
}

/// Removes leading block-quote markers, however deeply nested.
fn strip_quote(trimmed: &str) -> &str {
    let mut rest = trimmed;
    while let Some(inner) = rest.strip_prefix('>') {
        rest = inner.trim_start();
    }
    rest
}

/// Removes HTML comments, carrying the open/closed state across lines.
fn strip_comments(line: &str, in_comment: &mut bool) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    loop {
        if *in_comment {
            match rest.find("-->") {
                Some(end) => {
                    *in_comment = false;
                    rest = &rest[end + 3..];
                },
                None => break,
            }
        } else {
            match rest.find("<!--") {
                Some(start) => {
                    out.push_str(&rest[..start]);
                    *in_comment = true;
                    rest = &rest[start + 4..];
                },
                None => {
                    out.push_str(rest);
                    break;
                },
            }
        }
    }
    out
}

/// Takes the inline markup off one line, leaving what it wrapped.
fn clean_inline(line: &str) -> String {
    let text = strip_links(line);
    let text = strip_emphasis(&text);
    if line.trim_start().starts_with('|') {
        // A table row: the pipes are column rules, not speech.
        return text.replace('|', " ");
    }
    text
}

/// Replaces links and images with their text: `[text](url)` → `text`,
/// `![alt](url)` → `alt`, `[[note|alias]]` → `alias`. A bare URL is left
/// alone — it is content, not markup.
fn strip_links(line: &str) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < chars.len() {
        // An image's `!` belongs to the markup, not to the alt text.
        if chars[i] == '!' && chars.get(i + 1) == Some(&'[') {
            i += 1;
            continue;
        }
        if chars[i] != '[' {
            out.push(chars[i]);
            i += 1;
            continue;
        }
        // A wiki link: `[[target|alias]]`, spoken as the alias when it has one.
        if chars.get(i + 1) == Some(&'[') &&
            let Some(end) = find_from(&chars, i + 2, "]]")
        {
            let inner: String = chars[i + 2..end].iter().collect();
            let shown = inner.rsplit('|').next().unwrap_or(&inner);
            out.push_str(shown);
            i = end + 2;
            continue;
        }
        let Some(close) = chars[i + 1..].iter().position(|c| *c == ']') else {
            out.push(chars[i]);
            i += 1;
            continue;
        };
        let close = i + 1 + close;
        out.extend(&chars[i + 1..close]);
        i = close + 1;
        // Drop the target that follows the text, inline or by reference.
        match chars.get(i) {
            Some('(') => {
                if let Some(end) = skip_balanced(&chars, i) {
                    i = end;
                }
            },
            Some('[') => {
                if let Some(end) = find_from(&chars, i + 1, "]") {
                    i = end + 1;
                }
            },
            _ => {},
        }
    }
    out
}

/// The index of `needle` in `chars` at or after `from`.
fn find_from(chars: &[char], from: usize, needle: &str) -> Option<usize> {
    let pattern: Vec<char> = needle.chars().collect();
    (from..chars.len().saturating_sub(pattern.len() - 1))
        .find(|&start| chars[start..start + pattern.len()] == pattern[..])
}

/// One past the `)` closing the `(` at `open`, honoring nesting (URLs contain
/// parentheses often enough to matter).
fn skip_balanced(chars: &[char], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, c) in chars[open..].iter().enumerate() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + offset + 1);
                }
            },
            _ => {},
        }
    }
    None
}

/// Removes emphasis, strikethrough, and inline-code markers, plus the
/// backslashes that escape them.
///
/// Underscores are only markup at a word boundary: inside a word they are
/// part of an identifier, and `snake_case` should stay readable.
fn strip_emphasis(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '\\' if chars
                .get(i + 1)
                .is_some_and(|next| next.is_ascii_punctuation()) =>
            {
                out.push(chars[i + 1]);
                i += 2;
            },
            '*' | '`' => i += 1,
            '~' if chars.get(i + 1) == Some(&'~') => i += 2,
            '_' => {
                let before = i
                    .checked_sub(1)
                    .and_then(|prev| chars.get(prev))
                    .is_some_and(|c| c.is_alphanumeric());
                let after =
                    chars.get(i + 1).is_some_and(|c| c.is_alphanumeric());
                if before && after {
                    out.push(c);
                }
                i += 1;
            },
            _ => {
                out.push(c);
                i += 1;
            },
        }
    }
    out
}

/// Cuts `text` down to pieces of at most `budget` bytes, at punctuation where
/// there is any and at a word boundary otherwise.
///
/// This is the floor under a caller's own splitter, not a replacement for it: a
/// sentence longer than the budget cannot be split by a sentence model at all,
/// and on the short budgets some engines work with such a sentence is ordinary
/// rather than exceptional.
pub fn fit(text: &str, budget: usize) -> Vec<String> {
    let text = text.trim();
    if text.len() <= budget || budget == 0 {
        return vec![text.to_owned()];
    }
    let mut pieces = Vec::new();
    let mut rest = text;
    while rest.len() > budget {
        // The last break that fits: a clause boundary if there is one, else a
        // space, else the budget itself.
        let window = &rest[..char_boundary(rest, budget)];
        let cut = window
            .rfind([',', ';', ':', '.', '!', '?', '—', '–'])
            .map(|index| {
                index + window[index..].chars().next().map_or(1, char::len_utf8)
            })
            .or_else(|| window.rfind(' '))
            .unwrap_or(window.len());
        // A cut at zero would hand the whole remainder back and loop forever.
        // Only a budget narrower than one character gets there, which the
        // budget rules do not produce — but a loop is not something to
        // leave resting on a caller's arithmetic.
        let cut =
            cut.max(rest.chars().next().map_or(rest.len(), char::len_utf8));
        let (piece, tail) = rest.split_at(cut.min(rest.len()));
        let piece = piece.trim();
        if !piece.is_empty() {
            pieces.push(piece.to_owned());
        }
        rest = tail.trim_start();
        if rest.is_empty() {
            break;
        }
    }
    if !rest.is_empty() {
        pieces.push(rest.to_owned());
    }
    pieces
}

/// The largest character boundary at or below `at`.
fn char_boundary(text: &str, at: usize) -> usize {
    let mut at = at.min(text.len());
    while at > 0 && !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

#[cfg(test)]
mod tests;
