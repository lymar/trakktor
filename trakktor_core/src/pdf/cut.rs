//! `pdf cut`: the selected pages of a PDF, written out as a new PDF.
//!
//! The approach is deletion in place, not copying into a fresh document. A
//! page may inherit its `/Resources`, `/MediaBox` or `/Rotate` from a parent
//! node of the page tree, and a page re-hung under a new root loses them —
//! blank pages, wrong margins. Deleting the unwanted pages leaves the tree
//! (and so the inheritance) intact, and the garbage collection that follows
//! keeps exactly what the surviving pages reach: fonts and images come
//! through byte for byte, with no re-subsetting and no guesswork.

use std::{collections::HashSet, path::Path};

use lopdf::{Document, LoadOptions};

use crate::{
    pages::Selection,
    pdf::{Dropped, PdfError, catalog},
};

/// The most a single compressed stream may inflate to while the document
/// loads. Real object and cross-reference streams are megabytes at the very
/// most; a gigabyte here is not a limit anyone honest will meet, only a
/// guard against decompression bombs.
const MAX_DECOMPRESSED_STREAM: usize = 1 << 30;

/// How to cut.
#[derive(Debug, Clone)]
pub struct Options {
    /// Which pages to keep.
    pub pages: Selection,
    /// The password of an encrypted document.
    pub password: Option<String>,
}

/// What a cut produced.
#[derive(Debug, Clone)]
pub struct Cut {
    /// Pages in the source document — not in the result.
    pub page_count: u32,
    /// The pages kept, by their numbers in the source document, in order.
    /// In the new file they are pages 1..N in this same order.
    pub pages: Vec<u32>,
    /// What had to be dropped to keep the result honest.
    pub dropped: Vec<Dropped>,
}

/// Cuts the selected pages out of `input` and writes them as a new PDF at
/// `out`.
///
/// # Errors
///
/// Returns a [`PdfError`] when the file cannot be read or is not a PDF
/// (`invalid_input`), when its structure does not parse (`parse_failed`),
/// when it is encrypted and the password is missing or wrong (`encrypted`),
/// when the selection does not fit the document (`invalid_options`), or when
/// the result cannot be written (`io_error`).
pub fn cut(
    input: &Path,
    out: &Path,
    options: &Options,
) -> Result<Cut, PdfError> {
    let bytes = std::fs::read(input).map_err(|source| {
        PdfError::InvalidInput(format!(
            "cannot read {}: {source}",
            input.display()
        ))
    })?;
    if !bytes.starts_with(b"%PDF-") {
        return Err(PdfError::InvalidInput(format!(
            "{} is not a PDF",
            input.display()
        )));
    }

    let load = LoadOptions {
        password: options.password.clone(),
        max_decompressed_size: Some(MAX_DECOMPRESSED_STREAM),
        ..LoadOptions::default()
    };
    let mut doc =
        Document::load_mem_with_options(&bytes, load).map_err(translate)?;
    // Loading decrypts when it can — the empty password, or the one given.
    // A document still encrypted after loading is one it could not open;
    // its object map is empty and nothing below would make sense.
    if doc.is_encrypted() {
        return Err(PdfError::Encrypted);
    }

    let page_ids = doc.get_pages();
    let page_count = u32::try_from(page_ids.len()).unwrap_or(u32::MAX);
    if page_count == 0 {
        return Err(PdfError::ParseFailed(
            "the document has no pages".to_string(),
        ));
    }

    let kept = options.pages.resolve(page_count)?;
    let kept_set: HashSet<u32> = kept.iter().copied().collect();
    let removed_numbers: Vec<u32> = (1..=page_count)
        .filter(|page| !kept_set.contains(page))
        .collect();

    let all_pages: HashSet<lopdf::ObjectId> =
        page_ids.values().copied().collect();
    let removed_pages: HashSet<lopdf::ObjectId> = removed_numbers
        .iter()
        .filter_map(|number| page_ids.get(number).copied())
        .collect();

    // Catalog hygiene comes before the deletion, while every reference is
    // still intact and can be read for what it points at.
    let dropped = catalog::clean(&mut doc, &all_pages, &removed_pages);

    remove_pages(&mut doc, &removed_pages);
    doc.prune_objects();
    doc.renumber_objects();

    doc.save(out).map_err(|source| PdfError::Io {
        path: out.display().to_string(),
        source,
    })?;

    Ok(Cut {
        page_count,
        pages: kept,
        dropped,
    })
}

/// Takes the removed pages out of the page tree by hand: every node's
/// `/Kids` is rebuilt without them, its `/Count` is set to what survives, and
/// the page objects themselves are dropped. Parent nodes stay where they are,
/// so inherited attributes keep working.
///
/// The library's own `delete_pages` walks the whole document once *per page*
/// to strip every reference to it, and the main scenario here — a couple of
/// dozen pages kept out of hundreds — is its worst case: minutes where this
/// takes milliseconds. The price is that a reference to a removed page may
/// survive outside the page tree. The catalog was cleaned of those before
/// this runs; what can still hold one is a link annotation sitting on a
/// *kept* page, and that one stays a well-formed link that leads nowhere —
/// the same dead end other page cutters leave.
fn remove_pages(doc: &mut Document, removed: &HashSet<lopdf::ObjectId>) {
    let root = doc
        .trailer
        .get(b"Root")
        .and_then(lopdf::Object::as_reference)
        .ok()
        .and_then(|id| doc.objects.get(&id))
        .and_then(|object| object.as_dict().ok())
        .and_then(|catalog| catalog.get(b"Pages").ok())
        .and_then(|pages| pages.as_reference().ok());
    if let Some(root) = root {
        prune_tree_node(doc, root, removed, &mut HashSet::new());
    }
    for id in removed {
        doc.objects.remove(id);
    }
}

/// The recursive step of [`remove_pages`]: prunes one `/Pages` node and
/// returns how many pages survive beneath it.
fn prune_tree_node(
    doc: &mut Document,
    node: lopdf::ObjectId,
    removed: &HashSet<lopdf::ObjectId>,
    visited: &mut HashSet<lopdf::ObjectId>,
) -> i64 {
    if !visited.insert(node) {
        return 0;
    }
    let kids = doc
        .objects
        .get(&node)
        .and_then(|object| object.as_dict().ok())
        .and_then(|dict| dict.get(b"Kids").ok())
        .and_then(|kids| match kids {
            lopdf::Object::Reference(id) => doc.objects.get(id),
            direct => Some(direct),
        })
        .and_then(|kids| kids.as_array().ok())
        .cloned();
    let Some(kids) = kids else { return 0 };

    let mut surviving = Vec::with_capacity(kids.len());
    let mut count: i64 = 0;
    for kid in kids {
        match kid.as_reference() {
            Ok(id) if removed.contains(&id) => continue,
            Ok(id) => {
                let is_tree_node = doc
                    .objects
                    .get(&id)
                    .and_then(|object| object.as_dict().ok())
                    .is_some_and(|dict| dict.has(b"Kids"));
                if is_tree_node {
                    count += prune_tree_node(doc, id, removed, visited);
                } else {
                    count += 1;
                }
                surviving.push(kid);
            },
            Err(_) => surviving.push(kid),
        }
    }
    if let Some(dict) = doc
        .objects
        .get_mut(&node)
        .and_then(|object| object.as_dict_mut().ok())
    {
        dict.set("Kids", lopdf::Object::Array(surviving));
        dict.set("Count", count);
    }
    count
}

/// The library's load-time error, in our contract.
fn translate(error: lopdf::Error) -> PdfError {
    match error {
        lopdf::Error::InvalidPassword | lopdf::Error::Decryption(_) => {
            PdfError::Encrypted
        },
        lopdf::Error::IO(source) => {
            PdfError::InvalidInput(format!("cannot read the file: {source}"))
        },
        other => PdfError::ParseFailed(other.to_string()),
    }
}

#[cfg(test)]
mod tests;
