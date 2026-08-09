//! Reading the model description PaddleOCR writes as YAML.
//!
//! Every published model directory carries its description twice — once as
//! JSON and once as YAML — and this port read the JSON, which cost it no
//! dependency at all. The newest generation ships **only** the YAML, so the
//! choice is gone; what is left is how much of YAML to accept.
//!
//! The answer here is: the part that appears in these files, and nothing else.
//! They are machine-written dumps of a Python dictionary, so they use a small,
//! stable corner of the language — block mappings, block sequences, plain and
//! quoted scalars, one anchor and one alias — and this module reads that corner
//! into the same [`serde_json::Value`] the JSON path produces, so everything
//! downstream stays as it was. Anything outside the corner is an error with the
//! offending line in it, which is the honest failure for a reader that does not
//! claim to be a YAML parser.
//!
//! Two details are worth stating because they are where a hand-rolled reader
//! normally goes wrong:
//!
//! - **A block sequence may sit at its key's own indentation.** The character
//!   dictionary is written that way, so an indentation-driven reader that
//!   insists a value be indented deeper than its key reads eighteen thousand
//!   characters as zero.
//! - **A quoted scalar is not its own text.** The dictionary quotes the
//!   thirty-odd characters YAML would otherwise read as syntax, and one of them
//!   is the quote itself, written `''''`. Take the line at face value and the
//!   model gains four characters that are really one.

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::ocr::error::OcrError;

/// Reads a document into the value tree the rest of the port expects.
pub fn parse(text: &str) -> Result<Value, OcrError> {
    let lines: Vec<Line> = text
        .lines()
        .enumerate()
        .filter_map(|(at, raw)| Line::read(at + 1, raw))
        .collect();
    let mut reader = Reader {
        lines: &lines,
        at: 0,
        anchors: HashMap::new(),
    };
    let value = if reader.lines.is_empty() {
        Value::Null
    } else {
        reader.block(reader.lines[0].indent)?
    };
    if reader.at < lines.len() {
        return Err(unexpected(&lines[reader.at]));
    }
    Ok(value)
}

/// One line that carries something: how far it is indented and what is on it.
struct Line<'a> {
    number: usize,
    indent: usize,
    body: &'a str,
}

impl<'a> Line<'a> {
    /// Blank lines and whole-line comments carry nothing and are dropped here
    /// rather than checked for everywhere below.
    ///
    /// Only the ASCII space counts as indentation, and only it is trimmed.
    /// That is not pedantry: the character dictionary carries the ideographic
    /// space U+3000 as an entry of its own, and a reader that trims Unicode
    /// whitespace reads that line as an empty one.
    fn read(number: usize, raw: &'a str) -> Option<Self> {
        let body = trim_end(raw);
        let indent = body.len() - trim(body).len();
        let body = trim(body);
        if body.is_empty() || body.starts_with('#') {
            return None;
        }
        Some(Self {
            number,
            indent,
            body,
        })
    }

    /// The `- ` that opens a sequence entry, and what follows it.
    fn entry(&self) -> Option<&'a str> {
        if self.body == "-" {
            Some("")
        } else {
            self.body.strip_prefix("- ").map(trim)
        }
    }
}

/// Leading ASCII spaces removed. See [`Line::read`].
fn trim(text: &str) -> &str { text.trim_start_matches(' ') }

/// Trailing ASCII spaces, tabs and carriage returns removed.
fn trim_end(text: &str) -> &str { text.trim_end_matches([' ', '\t', '\r']) }

struct Reader<'a> {
    lines: &'a [Line<'a>],
    at: usize,
    anchors: HashMap<String, Value>,
}

impl<'a> Reader<'a> {
    fn peek(&self) -> Option<&'a Line<'a>> { self.lines.get(self.at) }

    /// Reads whatever block starts at the current line and is indented by at
    /// least `indent` — a sequence if the line opens with a dash, a mapping
    /// otherwise.
    fn block(&mut self, indent: usize) -> Result<Value, OcrError> {
        match self.peek() {
            Some(line) if line.entry().is_some() => self.sequence(indent),
            Some(_) => self.mapping(indent),
            None => Ok(Value::Null),
        }
    }

    fn sequence(&mut self, indent: usize) -> Result<Value, OcrError> {
        let mut items = Vec::new();
        while let Some(line) = self.peek() {
            if line.indent < indent {
                break;
            }
            if line.indent > indent {
                return Err(unexpected(line));
            }
            let Some(rest) = line.entry() else { break };
            // The dash and the space it takes up are indentation for whatever
            // continues the entry on the lines below.
            let inner = line.indent + 2;
            self.at += 1;
            items.push(self.item(rest, inner, line)?);
        }
        Ok(Value::Array(items))
    }

    /// One sequence entry, given what followed its dash.
    fn item(
        &mut self,
        rest: &'a str,
        inner: usize,
        line: &'a Line<'a>,
    ) -> Result<Value, OcrError> {
        if rest.is_empty() {
            return match self.peek() {
                Some(next) if next.indent >= inner => self.block(next.indent),
                _ => Ok(Value::Null),
            };
        }
        // A dash straight after a dash opens a nested sequence on the same
        // line, whose own first entry is the rest of that line.
        if rest == "-" || rest.starts_with("- ") {
            return self.nested_sequence(rest, inner, line);
        }
        match split_key(rest) {
            // A mapping whose first key shares the dash's line.
            Some((key, value)) => {
                let mut map = Map::new();
                let entry = self.value(value, inner, line)?;
                map.insert(key.to_string(), entry);
                self.mapping_into(&mut map, inner)?;
                Ok(Value::Object(map))
            },
            None => self.scalar(rest, line),
        }
    }

    /// A sequence written inline after another dash: its first entry is the
    /// rest of the outer dash's line, and its remaining entries are the lines
    /// below at the inner column.
    fn nested_sequence(
        &mut self,
        first: &'a str,
        inner: usize,
        line: &'a Line<'a>,
    ) -> Result<Value, OcrError> {
        let head = trim(first.strip_prefix("- ").unwrap_or(""));
        let mut items = vec![self.item(head, inner + 2, line)?];
        while let Some(next) = self.peek() {
            if next.indent != inner {
                break;
            }
            let Some(rest) = next.entry() else { break };
            self.at += 1;
            items.push(self.item(rest, inner + 2, next)?);
        }
        Ok(Value::Array(items))
    }

    fn mapping(&mut self, indent: usize) -> Result<Value, OcrError> {
        let mut map = Map::new();
        self.mapping_into(&mut map, indent)?;
        Ok(Value::Object(map))
    }

    fn mapping_into(
        &mut self,
        map: &mut Map<String, Value>,
        indent: usize,
    ) -> Result<(), OcrError> {
        while let Some(line) = self.peek() {
            if line.indent < indent || line.entry().is_some() {
                break;
            }
            if line.indent > indent {
                return Err(unexpected(line));
            }
            let Some((key, value)) = split_key(line.body) else {
                return Err(unexpected(line));
            };
            self.at += 1;
            let value = self.value(value, indent, line)?;
            map.insert(key.to_string(), value);
        }
        Ok(())
    }

    /// The value of a key, which is either on the key's own line or in the
    /// block below it.
    ///
    /// A block sequence is allowed to sit at the key's own indentation, which
    /// is why the test below is `>=` for a sequence and `>` for everything
    /// else.
    fn value(
        &mut self,
        written: &'a str,
        indent: usize,
        line: &'a Line<'a>,
    ) -> Result<Value, OcrError> {
        if let Some(anchor) = written.strip_prefix('&') {
            let value = self.nested(indent)?;
            self.anchors
                .insert(anchor.trim().to_string(), value.clone());
            return Ok(value);
        }
        if let Some(alias) = written.strip_prefix('*') {
            let name = alias.trim();
            return self.anchors.get(name).cloned().ok_or_else(|| {
                bad(format!(
                    "line {}: `*{name}` refers to an anchor this file has not \
                     defined",
                    line.number
                ))
            });
        }
        if written.is_empty() {
            return self.nested(indent);
        }
        self.scalar(written, line)
    }

    /// Whatever block follows a key that carried no value of its own.
    fn nested(&mut self, indent: usize) -> Result<Value, OcrError> {
        match self.peek() {
            Some(next)
                if next.indent > indent ||
                    (next.indent == indent && next.entry().is_some()) =>
            {
                self.block(next.indent)
            },
            _ => Ok(Value::Null),
        }
    }

    fn scalar(
        &mut self,
        written: &str,
        line: &'a Line<'a>,
    ) -> Result<Value, OcrError> {
        scalar(written).ok_or_else(|| {
            bad(format!("line {}: cannot read `{written}`", line.number))
        })
    }
}

/// Splits `key: value`, leaving the value empty when the key carries none.
///
/// A colon only ends a key when a space or the end of the line follows it, so
/// a plain scalar with a colon inside it stays whole.
fn split_key(body: &str) -> Option<(&str, &str)> {
    if body.starts_with(['\'', '"']) {
        return None;
    }
    let mut from = 0;
    while let Some(cut) = body[from..].find(':') {
        let at = from + cut;
        let rest = &body[at + 1..];
        if rest.is_empty() || rest.starts_with(' ') {
            let key = trim_end(&body[..at]);
            return (!key.is_empty()).then_some((key, trim(rest)));
        }
        from = at + 1;
    }
    None
}

/// One scalar, in the four spellings these files use: single-quoted,
/// double-quoted, a keyword, or plain.
fn scalar(written: &str) -> Option<Value> {
    if let Some(inner) = quoted(written, '\'') {
        // Inside single quotes the only escape is a doubled quote.
        return Some(Value::String(inner.replace("''", "'")));
    }
    if let Some(inner) = quoted(written, '"') {
        return Some(Value::String(unescape(inner)));
    }
    // A plain scalar cannot open with an indicator: a flow collection, a block
    // scalar, a tag or a directive is a shape this reader does not read, and
    // taking it at face value would turn `{a: 1}` into the text of itself. The
    // published dictionaries quote every such character, so nothing legitimate
    // is refused here.
    if written.starts_with(['{', '[', '|', '>', '!', '%', '@', '`', ',']) {
        return None;
    }
    Some(match written {
        "null" | "~" | "Null" | "NULL" => Value::Null,
        "true" | "True" | "TRUE" => Value::Bool(true),
        "false" | "False" | "FALSE" => Value::Bool(false),
        // A plain scalar is a number only if it reads as one whole; PaddleOCR
        // writes `1./255.` as a scalar and means the text, not a quotient.
        _ => match written.parse::<i64>() {
            Ok(int) => Value::Number(int.into()),
            Err(_) => match written.parse::<f64>().ok().and_then(|f| {
                (f.is_finite() && looks_numeric(written))
                    .then(|| serde_json::Number::from_f64(f))
                    .flatten()
            }) {
                Some(number) => Value::Number(number),
                None => Value::String(written.to_string()),
            },
        },
    })
}

/// Whether a plain scalar is written the way a number is, rather than merely
/// being parseable as one. `1.` and `inf` parse; neither is a number here.
fn looks_numeric(written: &str) -> bool {
    let body = written.strip_prefix(['+', '-']).unwrap_or(written);
    let mut chars = body.chars();
    chars.next().is_some_and(|c| c.is_ascii_digit()) &&
        body.chars().all(|c| {
            c.is_ascii_digit() ||
                c == '.' ||
                c == 'e' ||
                c == 'E' ||
                c == '+' ||
                c == '-'
        }) &&
        !body.ends_with('.')
}

/// The body of a scalar written in `quote`s, if it is written that way and the
/// quotes match.
fn quoted(written: &str, quote: char) -> Option<&str> {
    let inner = written.strip_prefix(quote)?.strip_suffix(quote)?;
    // `''` is an empty scalar, not an unterminated one.
    (written.len() >= 2).then_some(inner)
}

/// The handful of escapes a double-quoted scalar can carry. These files do not
/// use them today; leaving the sequence as written would be a silent change of
/// content if they start to.
fn unescape(inner: &str) -> String {
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('r') => out.push('\r'),
            Some('0') => out.push('\0'),
            Some(other) => out.push(other),
            None => out.push('\\'),
        }
    }
    out
}

fn unexpected(line: &Line) -> OcrError {
    bad(format!(
        "line {}: `{}` is not a shape this reader knows",
        line.number, line.body
    ))
}

fn bad(message: String) -> OcrError { OcrError::Artifact(message) }
