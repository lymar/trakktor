use lopdf::{
    Dictionary, Document, EncryptionState, EncryptionVersion, Object,
    Permissions, Stream,
    content::{Content, Operation},
    dictionary,
};

use super::*;

/// Cuts `spec` out of a PDF given as bytes; the result and its saved bytes.
fn run(
    bytes: &[u8],
    spec: &str,
    password: Option<&str>,
) -> Result<(Cut, Vec<u8>), PdfError> {
    let directory = tempfile::tempdir().expect("temp dir");
    let input = directory.path().join("in.pdf");
    std::fs::write(&input, bytes).expect("write the test PDF");
    let out = directory.path().join("out.pdf");
    let cut = cut(
        &input,
        &out,
        &Options {
            pages: Selection::parse(spec).expect(spec),
            password: password.map(str::to_string),
        },
    )?;
    let saved = std::fs::read(&out).expect("read the result");
    Ok((cut, saved))
}

/// [`run`] for the tests that expect success, with the result parsed back.
fn run_ok(bytes: &[u8], spec: &str) -> (Cut, Document) {
    let (cut, saved) = run(bytes, spec, None).expect("cut");
    let reloaded = Document::load_mem(&saved).expect("parse the result");
    (cut, reloaded)
}

/// One line of text for page `number`.
fn page_content(number: usize) -> Vec<u8> {
    let content = Content {
        operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 24.into()]),
            Operation::new("Td", vec![100.into(), 700.into()]),
            Operation::new(
                "Tj",
                vec![Object::string_literal(format!("Page {number}"))],
            ),
            Operation::new("ET", vec![]),
        ],
    };
    content.encode().expect("encode content")
}

/// A document of `count` one-line pages whose resources and media box live on
/// the page-tree node — the inherited case, common in the wild.
fn document(count: usize) -> (Document, Vec<lopdf::ObjectId>) {
    let mut doc = Document::with_version("1.5");
    let font = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });
    let resources = doc.add_object(dictionary! {
        "Font" => dictionary! { "F1" => font },
    });
    let pages_id = doc.new_object_id();
    let mut kids: Vec<Object> = Vec::with_capacity(count);
    let mut page_ids = Vec::with_capacity(count);
    for number in 1..=count {
        let contents =
            doc.add_object(Stream::new(dictionary! {}, page_content(number)));
        let id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => contents,
        });
        page_ids.push(id);
        kids.push(id.into());
    }
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => kids,
            "Count" => count as i64,
            "Resources" => resources,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        }),
    );
    let catalog = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog);
    (doc, page_ids)
}

fn save(mut doc: Document) -> Vec<u8> {
    let mut bytes = Vec::new();
    doc.save_to(&mut bytes).expect("write the test PDF");
    bytes
}

/// The catalog of a (re)loaded document.
fn catalog(doc: &Document) -> &Dictionary {
    let root = doc
        .trailer
        .get(b"Root")
        .and_then(Object::as_reference)
        .expect("Root");
    doc.get_dictionary(root).expect("catalog")
}

/// Mutates the catalog of a document under construction.
fn set_catalog_entry(doc: &mut Document, key: &str, value: Object) {
    let root = doc
        .trailer
        .get(b"Root")
        .and_then(Object::as_reference)
        .expect("Root");
    doc.objects
        .get_mut(&root)
        .and_then(|object| object.as_dict_mut().ok())
        .expect("catalog")
        .set(key, value);
}

/// An explicit destination to a page.
fn dest(page: lopdf::ObjectId) -> Object {
    Object::Array(vec![page.into(), "Fit".into()])
}

/// A one-item outline tree pointing at `page`.
fn outlines(doc: &mut Document, page: lopdf::ObjectId) {
    let item_id = doc.new_object_id();
    let outlines_id = doc.add_object(dictionary! {
        "Type" => "Outlines",
        "First" => item_id,
        "Last" => item_id,
        "Count" => 1,
    });
    doc.objects.insert(
        item_id,
        Object::Dictionary(dictionary! {
            "Title" => Object::string_literal("Chapter"),
            "Parent" => outlines_id,
            "Dest" => dest(page),
        }),
    );
    set_catalog_entry(doc, "Outlines", outlines_id.into());
}

#[test]
fn the_selected_pages_come_out_in_order_with_their_text() {
    let (doc, _) = document(5);
    let (cut, output) = run_ok(&save(doc), "2,4");

    assert_eq!(cut.page_count, 5);
    assert_eq!(cut.pages, vec![2, 4]);
    assert!(cut.dropped.is_empty());

    assert_eq!(output.get_pages().len(), 2);
    let first = output.extract_text(&[1]).expect("text of page 1");
    assert!(first.contains("Page 2"), "{first:?}");
    let second = output.extract_text(&[2]).expect("text of page 2");
    assert!(second.contains("Page 4"), "{second:?}");
}

#[test]
fn resources_inherited_from_the_page_tree_survive() {
    let (doc, _) = document(3);
    let (_, output) = run_ok(&save(doc), "2");

    let (_, page_id) = output.get_pages().into_iter().next().expect("one page");
    let (own, inherited) =
        output.get_page_resources(page_id).expect("page resources");
    assert!(own.is_none(), "resources should still be inherited");
    let fonts = inherited.iter().any(|id| {
        output
            .get_dictionary(*id)
            .is_ok_and(|resources| resources.has(b"Font"))
    });
    assert!(fonts, "the inherited resources should carry the font");
    let text = output.extract_text(&[1]).expect("text");
    assert!(text.contains("Page 2"), "{text:?}");
}

#[test]
fn a_font_shared_with_a_removed_page_stays_and_one_exclusive_to_it_goes() {
    const ALPHA: &[u8] = b"FONT-PROGRAM-ALPHA";
    const BETA: &[u8] = b"FONT-PROGRAM-BETA";

    let mut doc = Document::with_version("1.5");
    let embedded_font = |doc: &mut Document, program: &[u8]| {
        let file =
            doc.add_object(Stream::new(dictionary! {}, program.to_vec()));
        let descriptor = doc.add_object(dictionary! {
            "Type" => "FontDescriptor",
            "FontName" => "Test",
            "FontFile" => file,
        });
        doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Test",
            "FontDescriptor" => descriptor,
        })
    };
    let alpha = embedded_font(&mut doc, ALPHA);
    let beta = embedded_font(&mut doc, BETA);

    let pages_id = doc.new_object_id();
    let mut kids: Vec<Object> = Vec::new();
    for (number, font) in [(1, alpha), (2, alpha), (3, beta)] {
        let contents =
            doc.add_object(Stream::new(dictionary! {}, page_content(number)));
        let id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Contents" => contents,
            "Resources" => dictionary! {
                "Font" => dictionary! { "F1" => font },
            },
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        });
        kids.push(id.into());
    }
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => kids,
            "Count" => 3,
        }),
    );
    let catalog = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog);

    let (_, output) = run_ok(&save(doc), "1-2");
    let programs: Vec<&[u8]> = output
        .objects
        .values()
        .filter_map(|object| object.as_stream().ok())
        .map(|stream| stream.content.as_slice())
        .filter(|content| content.starts_with(b"FONT-PROGRAM"))
        .collect();
    assert_eq!(
        programs,
        vec![ALPHA],
        "the shared program byte for byte, the exclusive one gone"
    );
}

#[test]
fn bookmarks_leading_to_a_removed_page_are_dropped_and_reported() {
    let (mut doc, pages) = document(3);
    outlines(&mut doc, pages[2]);
    let (cut, output) = run_ok(&save(doc), "1-2");

    assert_eq!(cut.dropped, vec![Dropped::Outlines]);
    assert!(!catalog(&output).has(b"Outlines"));
}

#[test]
fn bookmarks_wholly_inside_the_kept_pages_survive() {
    let (mut doc, pages) = document(3);
    outlines(&mut doc, pages[0]);
    let (cut, output) = run_ok(&save(doc), "1-2");

    assert!(cut.dropped.is_empty(), "{:?}", cut.dropped);
    assert!(catalog(&output).has(b"Outlines"));
}

#[test]
fn page_labels_and_the_structure_tree_do_not_survive_a_cut() {
    let (mut doc, _) = document(3);
    set_catalog_entry(
        &mut doc,
        "PageLabels",
        Object::Dictionary(dictionary! { "Nums" => Vec::<Object>::new() }),
    );
    let tree = doc.add_object(dictionary! { "Type" => "StructTreeRoot" });
    set_catalog_entry(&mut doc, "StructTreeRoot", tree.into());
    let (cut, output) = run_ok(&save(doc), "1-2");

    assert_eq!(cut.dropped, vec![Dropped::PageLabels, Dropped::StructTree]);
    assert!(!catalog(&output).has(b"PageLabels"));
    assert!(!catalog(&output).has(b"StructTreeRoot"));
}

#[test]
fn a_full_copy_drops_nothing() {
    let (mut doc, pages) = document(3);
    outlines(&mut doc, pages[2]);
    set_catalog_entry(
        &mut doc,
        "PageLabels",
        Object::Dictionary(dictionary! { "Nums" => Vec::<Object>::new() }),
    );
    let (cut, output) = run_ok(&save(doc), "1-");

    assert_eq!(cut.pages, vec![1, 2, 3]);
    assert!(cut.dropped.is_empty(), "{:?}", cut.dropped);
    assert!(catalog(&output).has(b"Outlines"));
    assert!(catalog(&output).has(b"PageLabels"));
    assert_eq!(output.get_pages().len(), 3);
}

#[test]
fn named_destinations_are_filtered_name_by_name() {
    let (mut doc, pages) = document(3);
    let node = doc.add_object(dictionary! {
        "Names" => vec![
            Object::string_literal("alpha"),
            dest(pages[0]),
            Object::string_literal("omega"),
            dest(pages[2]),
        ],
    });
    set_catalog_entry(
        &mut doc,
        "Names",
        Object::Dictionary(dictionary! { "Dests" => node }),
    );
    let (cut, output) = run_ok(&save(doc), "1-2");

    assert_eq!(cut.dropped, vec![Dropped::NamedDestinations]);
    let names = catalog(&output)
        .get(b"Names")
        .and_then(Object::as_dict)
        .and_then(|names| names.get(b"Dests"))
        .and_then(Object::as_dict)
        .and_then(|dests| dests.get(b"Names"))
        .and_then(Object::as_array)
        .expect("the surviving name tree");
    assert_eq!(names.len(), 2, "{names:?}");
    assert!(
        matches!(&names[0], Object::String(name, _) if name == b"alpha"),
        "{names:?}"
    );
}

#[test]
fn the_open_action_dies_with_its_page() {
    let (mut doc, pages) = document(3);
    set_catalog_entry(&mut doc, "OpenAction", dest(pages[2]));
    let (cut, output) = run_ok(&save(doc), "1-2");

    assert_eq!(cut.dropped, vec![Dropped::OpenAction]);
    assert!(!catalog(&output).has(b"OpenAction"));
}

#[test]
fn form_fields_on_removed_pages_are_dropped_and_reported() {
    let (mut doc, pages) = document(3);
    let field = |doc: &mut Document, name: &str, page: lopdf::ObjectId| {
        let id = doc.add_object(dictionary! {
            "FT" => "Tx",
            "T" => Object::string_literal(name),
            "Type" => "Annot",
            "Subtype" => "Widget",
            "Rect" => vec![0.into(), 0.into(), 100.into(), 20.into()],
            "P" => page,
        });
        if let Some(dict) = doc
            .objects
            .get_mut(&page)
            .and_then(|object| object.as_dict_mut().ok())
        {
            dict.set("Annots", vec![id.into()]);
        }
        id
    };
    let kept = field(&mut doc, "kept", pages[0]);
    let gone = field(&mut doc, "gone", pages[2]);
    set_catalog_entry(
        &mut doc,
        "AcroForm",
        Object::Dictionary(dictionary! {
            "Fields" => vec![kept.into(), gone.into()],
        }),
    );
    let (cut, output) = run_ok(&save(doc), "1-2");

    assert_eq!(cut.dropped, vec![Dropped::FormFields]);
    let fields = catalog(&output)
        .get(b"AcroForm")
        .and_then(Object::as_dict)
        .and_then(|form| form.get(b"Fields"))
        .and_then(Object::as_array)
        .expect("the surviving fields");
    assert_eq!(fields.len(), 1, "{fields:?}");
}

#[test]
fn an_encrypted_document_opens_with_its_password_and_is_written_open() {
    let (doc, _) = document(3);
    let mut doc = doc;
    // Computing encryption keys needs the file identifier every real
    // producer writes.
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::string_literal(vec![7u8; 16]),
            Object::string_literal(vec![7u8; 16]),
        ]),
    );
    let state = EncryptionState::try_from(EncryptionVersion::V1 {
        document: &doc,
        owner_password: "owner",
        user_password: "user",
        permissions: Permissions::all(),
    })
    .expect("encryption state");
    doc.encrypt(&state).expect("encrypt");
    let bytes = save(doc);

    let (cut, saved) = run(&bytes, "1-2", Some("user")).expect("cut");
    assert_eq!(cut.pages, vec![1, 2]);
    let output = Document::load_mem(&saved).expect("parse the result");
    assert!(!output.is_encrypted(), "the result must be written open");
    let text = output.extract_text(&[1]).expect("text");
    assert!(text.contains("Page 1"), "{text:?}");
}

#[test]
fn an_encrypted_document_without_its_password_is_refused() {
    let (doc, _) = document(3);
    let mut doc = doc;
    // Computing encryption keys needs the file identifier every real
    // producer writes.
    doc.trailer.set(
        "ID",
        Object::Array(vec![
            Object::string_literal(vec![7u8; 16]),
            Object::string_literal(vec![7u8; 16]),
        ]),
    );
    let state = EncryptionState::try_from(EncryptionVersion::V1 {
        document: &doc,
        owner_password: "owner",
        user_password: "user",
        permissions: Permissions::all(),
    })
    .expect("encryption state");
    doc.encrypt(&state).expect("encrypt");
    let bytes = save(doc);

    let missing = run(&bytes, "1", None).unwrap_err();
    assert!(matches!(missing, PdfError::Encrypted), "{missing}");
    let wrong = run(&bytes, "1", Some("nope")).unwrap_err();
    assert!(matches!(wrong, PdfError::Encrypted), "{wrong}");
}

#[test]
fn a_file_that_is_not_a_pdf_is_invalid_input() {
    let err = run(b"plain text, no header", "1", None).unwrap_err();
    assert!(matches!(err, PdfError::InvalidInput(_)), "{err}");
}

#[test]
fn a_selection_past_the_end_is_invalid_options() {
    let (doc, _) = document(3);
    let err = run(&save(doc), "9", None).unwrap_err();
    assert!(matches!(err, PdfError::InvalidOptions(_)), "{err}");
    assert!(err.to_string().contains("has 3"), "{err}");
}
