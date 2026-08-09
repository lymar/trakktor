use std::io::Write;

use lopdf::{Document, Object, Stream, dictionary};

use super::*;

/// Writes `bytes` to a temporary file and converts it.
fn convert_bytes(
    bytes: &[u8],
    options: &Options,
) -> Result<Converted, ConvertError> {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("document.pdf");
    std::fs::File::create(&path)
        .and_then(|mut file| file.write_all(bytes))
        .expect("write the test PDF");
    convert(&path, options)
}

#[test]
fn a_typeset_document_converts_page_by_page() {
    let converted = convert_bytes(
        &paged_pdf(&[Page::text(1), Page::text(2)]),
        &Options::default(),
    )
    .expect("convert");

    assert_eq!(converted.kind, Kind::Text);
    assert_eq!(converted.page_count, 2);
    assert_eq!(converted.pages.len(), 2);
    assert_eq!(converted.pages[0].number, 1);
    assert_eq!(converted.pages[1].number, 2);
    assert!(converted.pages_needing_ocr().is_empty());
    assert!(converted.issues.is_empty());
    assert_eq!(converted.fonts_repaired, 0);

    let first = converted.pages[0].markdown.as_deref().unwrap_or_default();
    assert!(first.contains("page one"), "{first:?}");
    let second = converted.pages[1].markdown.as_deref().unwrap_or_default();
    assert!(second.contains("page two"), "{second:?}");
}

#[test]
fn the_assembled_markdown_separates_pages_and_marks_the_missing_ones() {
    let converted = convert_bytes(
        &paged_pdf(&[Page::text(1), Page::blank(), Page::text(3)]),
        &Options::default(),
    )
    .expect("convert");

    let markdown = converted.markdown();
    assert!(markdown.contains("<!-- page 1 -->"), "{markdown}");
    assert!(markdown.contains("<!-- page 3 -->"), "{markdown}");
    // A hole in the document is visible to whoever reads the source and
    // invisible in the rendered text.
    assert!(markdown.contains("page 2: no text layer"), "{markdown}");
    assert!(markdown.contains("trakktor ocr"), "{markdown}");
    assert_eq!(converted.pages_needing_ocr(), vec![2]);
}

#[test]
fn a_single_page_document_gets_no_separator() {
    let converted =
        convert_bytes(&paged_pdf(&[Page::text(1)]), &Options::default())
            .expect("convert");
    assert!(!converted.markdown().contains("<!-- page"));
}

#[test]
fn pages_selects_and_keeps_the_documents_own_numbering() {
    let pages: Vec<Page> = (1..=5).map(Page::text).collect();
    let converted = convert_bytes(
        &paged_pdf(&pages),
        &Options {
            pages: Some(Selection::parse("2-3").unwrap()),
            ..Options::default()
        },
    )
    .expect("convert");

    assert_eq!(
        converted
            .pages
            .iter()
            .map(|page| page.number)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
    // The document is still five pages long; only the result is two.
    assert_eq!(converted.page_count, 5);
    let second = converted.pages[0].markdown.as_deref().unwrap_or_default();
    assert!(second.contains("page two"), "{second:?}");
}

#[test]
fn a_page_range_past_the_end_is_rejected_before_anything_is_read() {
    let error = convert_bytes(
        &paged_pdf(&[Page::text(1)]),
        &Options {
            pages: Some(Selection::parse("4").unwrap()),
            ..Options::default()
        },
    )
    .expect_err("should refuse");
    assert!(matches!(error, ConvertError::InvalidOptions(_)), "{error}");
}

#[test]
fn a_document_with_nothing_to_read_says_to_use_ocr() {
    let blank: Vec<Page> = (0..3).map(|_| Page::blank()).collect();
    let error = convert_bytes(&paged_pdf(&blank), &Options::default())
        .expect_err("should refuse");
    assert!(matches!(error, ConvertError::NoTextLayer { .. }), "{error}");
    assert!(error.to_string().contains("trakktor ocr"), "{error}");
}

#[test]
fn a_file_that_is_not_a_pdf_is_rejected_by_its_first_bytes() {
    let error = convert_bytes(b"just some text\n", &Options::default())
        .expect_err("should refuse");
    assert!(matches!(error, ConvertError::InvalidInput(_)), "{error}");
    assert!(error.to_string().contains("not a PDF"), "{error}");
}

#[test]
fn a_missing_file_is_an_input_error() {
    let error = convert(
        std::path::Path::new("/nonexistent/nothing-here.pdf"),
        &Options::default(),
    )
    .expect_err("should refuse");
    assert!(matches!(error, ConvertError::InvalidInput(_)), "{error}");
}

#[test]
fn a_page_carrying_only_the_documents_watermark_is_not_content() {
    // Six pages stamped alike and one with something on it: the stamp is not
    // what the reader came for, and a page that is only the stamp has nothing
    // to give.
    let mut pages: Vec<Page> = (0..6).map(|_| Page::stamped()).collect();
    pages.push(Page::text(9));
    let converted = convert_bytes(&paged_pdf(&pages), &Options::default())
        .expect("convert");

    assert_eq!(converted.pages_needing_ocr(), vec![1, 2, 3, 4, 5, 6]);
    for page in &converted.pages[..6] {
        assert_eq!(page.reason, Some(Reason::NoUniqueText));
    }
    assert!(converted.pages[6].markdown.is_some());
}

#[test]
fn text_in_a_private_use_area_is_reported_rather_than_refused() {
    // The page converts — whatever else is on it is real — but the characters
    // the font maps into a private use area are gone for good, and saying so
    // is the only honest thing available.
    let converted = convert_bytes(
        &paged_pdf(&[Page::private_use(), Page::text(2)]),
        &Options::default(),
    )
    .expect("convert");

    assert!(
        converted.pages[0].markdown.is_some(),
        "the page still converts"
    );
    let issue = converted
        .issues
        .iter()
        .find(|issue| issue.code == "private_use_text")
        .expect("the damage should be reported");
    assert_eq!(issue.pages, vec![1]);
    assert!(issue.message.contains("trakktor ocr"), "{}", issue.message);
}

#[test]
fn an_encrypted_document_asks_for_its_password_rather_than_looking_like_a_scan()
{
    let lines = prose_page();
    let encrypted = encrypt(&paged_pdf(&[lines]), "owner", "user");

    // Without one, and with the wrong one, the answer is the same and it is
    // the true one.
    for password in [None, Some("wrong")] {
        let error = convert_bytes(
            &encrypted,
            &Options {
                password: password.map(str::to_string),
                ..Options::default()
            },
        )
        .expect_err("should refuse");
        assert!(matches!(error, ConvertError::Encrypted), "{error}");
    }

    // The owner password is the trap: it opens the file and leaves the text
    // locked, so the engine reports an empty scan. Answering "run OCR on it"
    // would send the caller down a road that cannot work.
    let error = convert_bytes(
        &encrypted,
        &Options {
            password: Some("owner".to_string()),
            ..Options::default()
        },
    )
    .expect_err("should refuse");
    assert!(matches!(error, ConvertError::Encrypted), "{error}");

    // The other half — that the *user* password opens the document — is not
    // asserted here: the engine declines to decrypt what this test can
    // produce, though other readers open it, so a failing assertion would be
    // about the fixture rather than about trakktor. It is checked by hand
    // against a document from an ordinary PDF producer instead.
}

/// The same document, locked with a user and an owner password.
fn encrypt(source: &[u8], owner: &str, user: &str) -> Vec<u8> {
    use lopdf::encryption::{EncryptionState, EncryptionVersion, Permissions};

    let mut document = Document::load_mem(source).expect("load");
    // The file identifier is part of the encryption key, so a document without
    // one cannot be encrypted at all.
    let id = Object::String(vec![0x42; 16], lopdf::StringFormat::Hexadecimal);
    document.trailer.set("ID", vec![id.clone(), id]);
    let version = EncryptionVersion::V2 {
        document: &document,
        owner_password: owner,
        user_password: user,
        key_length: 16,
        permissions: Permissions::all(),
    };
    let state = EncryptionState::try_from(version).expect("encryption state");
    document.encrypt(&state).expect("encrypt");

    // `/Length` in the encryption dictionary is in **bits**; lopdf writes the
    // byte count it was given, which leaves a file most readers cannot open.
    if let Ok(Object::Reference(id)) = document.trailer.get(b"Encrypt") &&
        let Ok(encrypt) =
            document.get_object_mut(*id).and_then(Object::as_dict_mut)
    {
        encrypt.set("Length", Object::Integer(128));
    }

    let mut bytes = Vec::new();
    document.save_to(&mut bytes).expect("write");
    bytes
}

/// Four lines of prose, which is what it takes for a page to count as carrying
/// text rather than a stray mark.
fn prose_page() -> Page { Page::text(1) }

/// What one page of a test document holds.
enum Page {
    /// Ordinary prose, with the page's number spelled out in it.
    Text(u32),
    /// Nothing at all.
    Blank,
    /// The same few words every stamped page carries.
    Stamped,
    /// Prose plus characters that map nowhere.
    PrivateUse,
}

impl Page {
    fn text(number: u32) -> Self { Page::Text(number) }

    fn blank() -> Self { Page::Blank }

    fn stamped() -> Self { Page::Stamped }

    fn private_use() -> Self { Page::PrivateUse }

    /// The page's content stream. Four lines of it, because a page with fewer
    /// text operators than that is classified as having no text at all.
    fn content(&self) -> Vec<u8> {
        let lines: Vec<String> = match self {
            Page::Blank => return Vec::new(),
            Page::Stamped => {
                vec!["DRAFT COPY".to_string(); 2]
            },
            Page::PrivateUse => (1..=4)
                .map(|line| format!("line {line} with a glyph"))
                .collect(),
            Page::Text(number) => {
                let spelled = match number {
                    1 => "one",
                    2 => "two",
                    3 => "three",
                    4 => "four",
                    5 => "five",
                    _ => "later",
                };
                (1..=4)
                    .map(|line| {
                        format!("This is page {spelled}, line {line} of prose.")
                    })
                    .collect()
            },
        };
        let mut content = Vec::new();
        for (at, line) in lines.iter().enumerate() {
            let y = 720 - 20 * at;
            let font = if matches!(self, Page::PrivateUse) {
                "F2"
            } else {
                "F1"
            };
            content.extend_from_slice(
                format!("BT /{font} 12 Tf 72 {y} Td ({line}) Tj ET\n")
                    .as_bytes(),
            );
        }
        content
    }
}

/// A PDF of the given pages, in order.
///
/// Two fonts are declared on every page: `F1`, plain Helvetica, and `F2`, a
/// composite font whose `/ToUnicode` sends every code into a private use area —
/// the shape of a document that carries a script no reader can spell out.
fn paged_pdf(pages: &[Page]) -> Vec<u8> {
    let mut document = Document::with_version("1.5");

    let helvetica = document.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });
    let private_use = private_use_font(&mut document);
    let resources = document.add_object(dictionary! {
        "Font" => dictionary! { "F1" => helvetica, "F2" => private_use },
    });

    let pages_id = document.new_object_id();
    let mut kids: Vec<Object> = Vec::with_capacity(pages.len());
    for page in pages {
        let mut page_dictionary = dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "Resources" => resources,
        };
        let content = page.content();
        if !content.is_empty() {
            let contents =
                document.add_object(Stream::new(dictionary! {}, content));
            page_dictionary.set("Contents", contents);
        }
        kids.push(document.add_object(page_dictionary).into());
    }
    let count = kids.len() as i64;
    document.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => kids,
            "Count" => count,
        }),
    );
    let catalog = document.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    document.trailer.set("Root", catalog);

    let mut bytes = Vec::new();
    document.save_to(&mut bytes).expect("write the test PDF");
    bytes
}

/// A simple font whose `/ToUnicode` maps the printable ASCII codes into the
/// private use area, which is what an unrecoverable script looks like from the
/// outside.
fn private_use_font(document: &mut Document) -> lopdf::ObjectId {
    let mut cmap = String::from(
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n1 \
         begincodespacerange\n<00> <FF>\nendcodespacerange\n",
    );
    cmap.push_str("100 beginbfchar\n");
    for code in 0x20u32..0x84 {
        cmap.push_str(&format!(
            "<{code:02X}> <{:04X}>\n",
            0xE000 + code - 0x20
        ));
    }
    cmap.push_str(
        "endbfchar\nendcmap\nCMapName currentdict /CMap defineresource \
         pop\nend\nend\n",
    );
    let to_unicode =
        document.add_object(Stream::new(dictionary! {}, cmap.into_bytes()));
    document.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
        "ToUnicode" => to_unicode,
    })
}
