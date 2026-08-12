//! Catalog hygiene ahead of page deletion.
//!
//! Deleting a page strips every reference to it, and a document-level
//! structure that pointed at the page survives that mutilated rather than
//! whole: an outline destination loses its page, a form field its widget's
//! home, page labels keep numbering positions that no longer exist. The rule
//! here is to drop such a structure honestly — before the deletion, while the
//! references are still intact and can be read — and to report every drop.
//!
//! Everything in this module is best-effort over hostile input: a catalog
//! entry that does not parse is left alone rather than guessed at, and no
//! failure here may fail the cut.

use std::collections::{BTreeMap, HashSet};

use lopdf::{Dictionary, Document, Object, ObjectId};

use crate::pdf::Dropped;

/// How many reference hops to follow before calling it a loop.
const MAX_DEREF_HOPS: usize = 8;

/// How deep a walk into catalog structures may recurse.
const MAX_WALK_DEPTH: usize = 64;

/// Cleans the catalog of structures that would not survive deleting the
/// `removed` pages, and says what was dropped.
///
/// `pages` is every page of the document — a walk never expands into a page
/// object, whether it stays or goes.
pub(super) fn clean(
    doc: &mut Document,
    pages: &HashSet<ObjectId>,
    removed: &HashSet<ObjectId>,
) -> Vec<Dropped> {
    if removed.is_empty() {
        return Vec::new();
    }
    let mut dropped = Vec::new();

    // Named destinations go first: whether an outline or the open action is
    // broken depends on which names survive.
    let removed_names = clean_named_destinations(doc, removed);

    if outlines_reach_removed(doc, pages, removed, &removed_names) {
        catalog_remove(doc, b"Outlines");
        dropped.push(Dropped::Outlines);
    }

    // Page labels number positions, not pages: after any deletion every label
    // past the first removed page tells a lie.
    if catalog_remove(doc, b"PageLabels") {
        dropped.push(Dropped::PageLabels);
    }

    // The structure tree holds references into content across the whole
    // document; there is no cheap way to cut it consistently.
    if catalog_remove(doc, b"StructTreeRoot") {
        dropped.push(Dropped::StructTree);
    }

    if open_action_reaches_removed(doc, removed, &removed_names) {
        catalog_remove(doc, b"OpenAction");
        dropped.push(Dropped::OpenAction);
    }

    if clean_form_fields(doc, removed) {
        dropped.push(Dropped::FormFields);
    }

    if !removed_names.is_empty() {
        dropped.push(Dropped::NamedDestinations);
    }

    dropped
}

/// Follows references until an actual object (or a loop) is reached.
fn deref<'a>(doc: &'a Document, mut object: &'a Object) -> &'a Object {
    for _ in 0..MAX_DEREF_HOPS {
        match object {
            Object::Reference(id) => match doc.objects.get(id) {
                Some(next) => object = next,
                None => break,
            },
            _ => break,
        }
    }
    object
}

/// The id of the document catalog, when the trailer names one.
fn catalog_id(doc: &Document) -> Option<ObjectId> {
    doc.trailer.get(b"Root").and_then(Object::as_reference).ok()
}

/// The catalog dictionary, read-only.
fn catalog(doc: &Document) -> Option<&Dictionary> {
    let id = catalog_id(doc)?;
    doc.objects.get(&id)?.as_dict().ok()
}

/// Removes `key` from the catalog; whether there was one to remove.
fn catalog_remove(doc: &mut Document, key: &[u8]) -> bool {
    let Some(id) = catalog_id(doc) else {
        return false;
    };
    let Some(dict) = doc
        .objects
        .get_mut(&id)
        .and_then(|object| object.as_dict_mut().ok())
    else {
        return false;
    };
    dict.remove(key).is_some()
}

/// Whether a destination value — an explicit `[page /XYZ …]` array, or a
/// dictionary wrapping one under `/D` — points at a removed page.
fn dest_reaches_removed(
    doc: &Document,
    value: &Object,
    removed: &HashSet<ObjectId>,
) -> bool {
    let value = deref(doc, value);
    let array = match value {
        Object::Array(array) => Some(array),
        Object::Dictionary(dict) => dict
            .get(b"D")
            .ok()
            .map(|inner| deref(doc, inner))
            .and_then(|inner| inner.as_array().ok()),
        _ => None,
    };
    matches!(
        array.and_then(|array| array.first()),
        Some(Object::Reference(id)) if removed.contains(id)
    )
}

/// The name a destination value carries when it is a *named* destination —
/// a name or string instead of an explicit array.
fn dest_name(value: &Object) -> Option<Vec<u8>> {
    match value {
        Object::Name(name) => Some(name.clone()),
        Object::String(bytes, _) => Some(bytes.clone()),
        _ => None,
    }
}

/// Rewrites the two homes of named destinations — the old-style `/Dests`
/// dictionary and the `/Names` → `/Dests` name tree — keeping only the names
/// whose destination survives. Returns the names that did not.
fn clean_named_destinations(
    doc: &mut Document,
    removed: &HashSet<ObjectId>,
) -> HashSet<Vec<u8>> {
    let mut removed_names = HashSet::new();

    // Old-style: catalog /Dests is a plain dictionary of name → destination.
    if let Some(value) = catalog(doc).and_then(|dict| dict.get(b"Dests").ok()) {
        let mut survivors: BTreeMap<Vec<u8>, Object> = BTreeMap::new();
        if let Ok(dests) = deref(doc, value).as_dict() {
            for (name, dest) in dests.iter() {
                if dest_reaches_removed(doc, dest, removed) {
                    removed_names.insert(name.clone());
                } else {
                    survivors.insert(name.clone(), dest.clone());
                }
            }
        }
        if !removed_names.is_empty() {
            replace_catalog_entry(doc, b"Dests", dests_dictionary(survivors));
        }
    }

    // Name tree: catalog /Names → /Dests, nodes of /Kids and /Names pairs.
    let tree_root = catalog(doc)
        .and_then(|dict| dict.get(b"Names").ok())
        .map(|value| deref(doc, value))
        .and_then(|value| value.as_dict().ok())
        .and_then(|names| names.get(b"Dests").ok().cloned());
    if let Some(root) = tree_root {
        let mut pairs: BTreeMap<Vec<u8>, Object> = BTreeMap::new();
        collect_name_tree(doc, &root, &mut pairs, 0, &mut HashSet::new());
        let before = pairs.len();
        pairs.retain(|name, dest| {
            let keep = !dest_reaches_removed(doc, dest, removed);
            if !keep {
                removed_names.insert(name.clone());
            }
            keep
        });
        if pairs.len() != before {
            replace_names_dests(doc, pairs);
        }
    }

    removed_names
}

/// A direct dictionary of surviving old-style destinations, or `None` to
/// remove the entry when nothing survived.
fn dests_dictionary(survivors: BTreeMap<Vec<u8>, Object>) -> Option<Object> {
    if survivors.is_empty() {
        return None;
    }
    let mut dict = Dictionary::new();
    for (name, dest) in survivors {
        dict.set(name, dest);
    }
    Some(Object::Dictionary(dict))
}

/// Sets or removes a catalog entry in place.
fn replace_catalog_entry(
    doc: &mut Document,
    key: &[u8],
    value: Option<Object>,
) {
    let Some(id) = catalog_id(doc) else { return };
    let Some(dict) = doc
        .objects
        .get_mut(&id)
        .and_then(|object| object.as_dict_mut().ok())
    else {
        return;
    };
    match value {
        Some(value) => dict.set(key, value),
        None => {
            dict.remove(key);
        },
    }
}

/// Gathers every name → destination pair of a name (sub)tree.
fn collect_name_tree(
    doc: &Document,
    node: &Object,
    pairs: &mut BTreeMap<Vec<u8>, Object>,
    depth: usize,
    visited: &mut HashSet<ObjectId>,
) {
    if depth > MAX_WALK_DEPTH {
        return;
    }
    if let Object::Reference(id) = node &&
        !visited.insert(*id)
    {
        return;
    }
    let Ok(dict) = deref(doc, node).as_dict() else {
        return;
    };
    if let Ok(kids) = dict.get(b"Kids").map(|value| deref(doc, value)) &&
        let Ok(kids) = kids.as_array()
    {
        for kid in kids {
            collect_name_tree(doc, kid, pairs, depth + 1, visited);
        }
    }
    if let Ok(names) = dict.get(b"Names").map(|value| deref(doc, value)) &&
        let Ok(names) = names.as_array()
    {
        for pair in names.chunks(2) {
            if let [key, value] = pair &&
                let Some(name) = dest_name(deref(doc, key))
            {
                pairs.insert(name, value.clone());
            }
        }
    }
}

/// Rewrites `/Names` → `/Dests` as one flat, sorted node of the surviving
/// pairs — or removes it (and an emptied `/Names`) when nothing survived.
fn replace_names_dests(doc: &mut Document, pairs: BTreeMap<Vec<u8>, Object>) {
    // The /Names dictionary itself may sit in the catalog directly or behind
    // a reference; find where to mutate.
    let Some(value) = catalog(doc).and_then(|dict| dict.get(b"Names").ok())
    else {
        return;
    };
    let names_home = match value {
        Object::Reference(id) => Some(*id),
        _ => None,
    };

    let replacement = if pairs.is_empty() {
        None
    } else {
        let mut names = Vec::with_capacity(pairs.len() * 2);
        for (name, dest) in pairs {
            names.push(Object::String(name, lopdf::StringFormat::Literal));
            names.push(dest);
        }
        let mut node = Dictionary::new();
        node.set("Names", Object::Array(names));
        Some(Object::Dictionary(node))
    };

    let mutate = |dict: &mut Dictionary| {
        match replacement {
            Some(node) => dict.set("Dests", node),
            None => {
                dict.remove(b"Dests");
            },
        }
        dict.is_empty()
    };

    let emptied = match names_home {
        Some(id) => doc
            .objects
            .get_mut(&id)
            .and_then(|object| object.as_dict_mut().ok())
            .map(mutate),
        None => {
            let Some(root) = catalog_id(doc) else { return };
            doc.objects
                .get_mut(&root)
                .and_then(|object| object.as_dict_mut().ok())
                .and_then(|catalog| {
                    catalog
                        .get_mut(b"Names")
                        .ok()?
                        .as_dict_mut()
                        .ok()
                        .map(mutate)
                })
        },
    };
    if emptied == Some(true) {
        catalog_remove(doc, b"Names");
    }
}

/// Whether anything under `/Outlines` — an item, its destination, its action —
/// reaches a removed page or a removed name.
///
/// The walk expands every reference except the ones that leave the outline
/// tree upward or sideways (`/Parent`, `/Prev`, `/SE` into the structure
/// tree), and never expands into a page object: reaching a *kept* page is
/// what an outline is for.
fn outlines_reach_removed(
    doc: &Document,
    pages: &HashSet<ObjectId>,
    removed: &HashSet<ObjectId>,
    removed_names: &HashSet<Vec<u8>>,
) -> bool {
    let Some(start) = catalog(doc).and_then(|dict| dict.get(b"Outlines").ok())
    else {
        return false;
    };
    let mut visited = HashSet::new();
    walk_reaches_removed(
        doc,
        start,
        pages,
        removed,
        removed_names,
        0,
        &mut visited,
    )
}

/// The recursive step of [`outlines_reach_removed`].
fn walk_reaches_removed(
    doc: &Document,
    object: &Object,
    pages: &HashSet<ObjectId>,
    removed: &HashSet<ObjectId>,
    removed_names: &HashSet<Vec<u8>>,
    depth: usize,
    visited: &mut HashSet<ObjectId>,
) -> bool {
    if depth > MAX_WALK_DEPTH {
        return false;
    }
    match object {
        Object::Reference(id) => {
            if removed.contains(id) {
                return true;
            }
            if pages.contains(id) || !visited.insert(*id) {
                return false;
            }
            match doc.objects.get(id) {
                Some(next) => walk_reaches_removed(
                    doc,
                    next,
                    pages,
                    removed,
                    removed_names,
                    depth + 1,
                    visited,
                ),
                None => false,
            }
        },
        Object::Array(array) => array.iter().any(|item| {
            walk_reaches_removed(
                doc,
                item,
                pages,
                removed,
                removed_names,
                depth + 1,
                visited,
            )
        }),
        Object::Dictionary(dict) => dict.iter().any(|(key, value)| {
            match key.as_slice() {
                // Backlinks and the structure tree: expanding them either
                // loops back or wanders into content that references every
                // page of the document.
                b"Parent" | b"Prev" | b"SE" => false,
                // A destination given by name is checked against the names
                // that did not survive, not walked.
                b"Dest" | b"D" => match dest_name(deref(doc, value)) {
                    Some(name) => removed_names.contains(&name),
                    None => walk_reaches_removed(
                        doc,
                        value,
                        pages,
                        removed,
                        removed_names,
                        depth + 1,
                        visited,
                    ),
                },
                _ => walk_reaches_removed(
                    doc,
                    value,
                    pages,
                    removed,
                    removed_names,
                    depth + 1,
                    visited,
                ),
            }
        }),
        _ => false,
    }
}

/// Whether the document's open action lands on a removed page or name.
fn open_action_reaches_removed(
    doc: &Document,
    removed: &HashSet<ObjectId>,
    removed_names: &HashSet<Vec<u8>>,
) -> bool {
    let Some(value) =
        catalog(doc).and_then(|dict| dict.get(b"OpenAction").ok())
    else {
        return false;
    };
    let action = deref(doc, value);
    let dest = match action {
        Object::Dictionary(dict) => match dict.get(b"D") {
            Ok(dest) => dest,
            Err(_) => return false,
        },
        _ => action,
    };
    if let Some(name) = dest_name(deref(doc, dest)) {
        return removed_names.contains(&name);
    }
    dest_reaches_removed(doc, dest, removed)
}

/// Filters the interactive form down to the fields that still have a widget
/// on a surviving page; whether anything was dropped.
fn clean_form_fields(doc: &mut Document, removed: &HashSet<ObjectId>) -> bool {
    let Some(form_value) =
        catalog(doc).and_then(|dict| dict.get(b"AcroForm").ok())
    else {
        return false;
    };
    let form_home = form_value.as_reference().ok();
    let Ok(form) = deref(doc, form_value).as_dict() else {
        return false;
    };
    let Ok(fields_value) = form.get(b"Fields") else {
        return false;
    };
    let fields_home = fields_value.as_reference().ok();
    let Ok(fields) = deref(doc, fields_value).as_array() else {
        return false;
    };

    let mut survivors = Vec::with_capacity(fields.len());
    let mut any_dropped = false;
    for field in fields {
        if field_survives(doc, field, removed, 0, &mut HashSet::new()) {
            survivors.push(field.clone());
        } else {
            any_dropped = true;
        }
    }
    if !any_dropped {
        return false;
    }

    if survivors.is_empty() {
        catalog_remove(doc, b"AcroForm");
        return true;
    }

    // Prune dead branches inside every surviving field tree first, then put
    // the surviving roots back where the /Fields array lives.
    let ids: Vec<ObjectId> = survivors
        .iter()
        .filter_map(|field| field.as_reference().ok())
        .collect();
    for id in ids {
        filter_field_kids(doc, id, removed, 0, &mut HashSet::new());
    }
    let fields = Object::Array(survivors);
    match fields_home {
        Some(id) => {
            if let Some(array) = doc
                .objects
                .get_mut(&id)
                .and_then(|object| object.as_array_mut().ok())
            {
                *array = match fields {
                    Object::Array(array) => array,
                    _ => unreachable!(),
                };
            }
        },
        None => {
            let target = form_home.or_else(|| catalog_id(doc));
            if let Some(id) = target &&
                let Some(object) = doc.objects.get_mut(&id)
            {
                let dict = match form_home {
                    Some(_) => object.as_dict_mut().ok(),
                    None => object
                        .as_dict_mut()
                        .ok()
                        .and_then(|catalog| catalog.get_mut(b"AcroForm").ok())
                        .and_then(|form| form.as_dict_mut().ok()),
                };
                if let Some(dict) = dict {
                    dict.set("Fields", fields);
                }
            }
        },
    }
    true
}

/// Whether a field (or any widget below it) still sits on a surviving page.
///
/// A field that names no page at all is kept: nothing says it is broken, and
/// the rule is to drop on evidence, not on doubt.
fn field_survives(
    doc: &Document,
    field: &Object,
    removed: &HashSet<ObjectId>,
    depth: usize,
    visited: &mut HashSet<ObjectId>,
) -> bool {
    if depth > MAX_WALK_DEPTH {
        return true;
    }
    if let Object::Reference(id) = field &&
        !visited.insert(*id)
    {
        return false;
    }
    let Ok(dict) = deref(doc, field).as_dict() else {
        return true;
    };
    if let Ok(kids) = dict.get(b"Kids").map(|value| deref(doc, value)) &&
        let Ok(kids) = kids.as_array()
    {
        return kids
            .iter()
            .any(|kid| field_survives(doc, kid, removed, depth + 1, visited));
    }
    match dict.get(b"P") {
        Ok(Object::Reference(page)) => !removed.contains(page),
        _ => true,
    }
}

/// Rewrites the `/Kids` of a field so that only surviving branches remain.
fn filter_field_kids(
    doc: &mut Document,
    id: ObjectId,
    removed: &HashSet<ObjectId>,
    depth: usize,
    visited: &mut HashSet<ObjectId>,
) {
    if depth > MAX_WALK_DEPTH || !visited.insert(id) {
        return;
    }
    let Some(kids) = doc
        .objects
        .get(&id)
        .and_then(|object| object.as_dict().ok())
        .and_then(|dict| dict.get(b"Kids").ok())
        .map(|value| deref(doc, value))
        .and_then(|value| value.as_array().ok())
        .cloned()
    else {
        return;
    };

    let survivors: Vec<Object> = kids
        .into_iter()
        .filter(|kid| {
            field_survives(doc, kid, removed, depth + 1, &mut HashSet::new())
        })
        .collect();
    let kid_ids: Vec<ObjectId> = survivors
        .iter()
        .filter_map(|kid| kid.as_reference().ok())
        .collect();

    if let Some(dict) = doc
        .objects
        .get_mut(&id)
        .and_then(|object| object.as_dict_mut().ok())
    {
        dict.set("Kids", Object::Array(survivors));
    }
    for kid in kid_ids {
        filter_field_kids(doc, kid, removed, depth + 1, visited);
    }
}
