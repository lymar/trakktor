use super::*;

/// The shape of the encoding vector a TeX Type 1 font carries: an array is
/// built, then filled in one `dup … put` at a time, and everything after
/// `eexec` is binary.
const PROGRAM: &[u8] = b"%!PS-AdobeFont-1.0: TestFont\n\
    /Encoding 256 array\n\
    0 1 255 {1 index exch /.notdef put} for\n\
    dup 11 /ff put\n\
    dup 12 /fi put\n\
    dup 22 /macron put\n\
    dup 123 /endash put\n\
    readonly def\n\
    currentfile eexec\n\
    dup 200 /never put\x80\x01\x02\x03";

#[test]
fn the_encoding_vector_is_read_out_of_the_cleartext_part() {
    let vector = builtin_encoding(PROGRAM);
    assert_eq!(
        vector,
        vec![
            (11, "ff".to_string()),
            (12, "fi".to_string()),
            (22, "macron".to_string()),
            (123, "endash".to_string()),
        ]
    );
}

#[test]
fn nothing_after_eexec_is_read() {
    // The encrypted part is not PostScript and must not be scanned for
    // anything that happens to look like an encoding entry.
    assert!(
        builtin_encoding(PROGRAM)
            .iter()
            .all(|(code, _)| *code != 200)
    );
}

#[test]
fn the_notdef_filler_is_not_an_entry() {
    // `0 1 255 {1 index exch /.notdef put} for` is how the array is cleared;
    // taking it as a mapping would map every code to nothing.
    assert!(
        builtin_encoding(PROGRAM)
            .iter()
            .all(|(_, glyph)| glyph != ".notdef")
    );
}

#[test]
fn malformed_entries_are_skipped_rather_than_guessed_at() {
    let program = b"dup 5 /good put dup /bad put dup 999 /toobig put dup 7 \
                    bare put dup 8 /has[bracket put eexec"
        as &[u8];
    assert_eq!(builtin_encoding(program), vec![(5, "good".to_string())]);
}

#[test]
fn a_program_without_a_vector_yields_nothing() {
    assert!(
        builtin_encoding(b"%!PS-AdobeFont-1.0\ncurrentfile eexec\n").is_empty()
    );
}

#[test]
fn differences_are_written_in_run_length_form() {
    let vector = vec![
        (11, "ff".to_string()),
        (12, "fi".to_string()),
        (123, "endash".to_string()),
    ];
    let encoding = differences(&vector);
    let array = encoding.get(b"Differences").unwrap().as_array().unwrap();
    let rendered: Vec<String> = array
        .iter()
        .map(|object| match object {
            Object::Integer(code) => code.to_string(),
            Object::Name(name) => {
                format!("/{}", String::from_utf8_lossy(name))
            },
            other => format!("{other:?}"),
        })
        .collect();
    // 12 follows 11 and so needs no number of its own; 123 does.
    assert_eq!(rendered, vec!["11", "/ff", "/fi", "123", "/endash"]);
}

#[test]
fn an_accent_joins_the_letter_it_was_drawn_in_front_of() {
    assert_eq!(compose_diacritics("K\u{00AF}alacakra"), "Kālacakra");
    assert_eq!(compose_diacritics("r\u{00B4}en"), "rén");
    assert_eq!(compose_diacritics("\u{02C7}c"), "č");
}

#[test]
fn a_dotless_letter_gets_its_dot_back_under_an_accent() {
    // `ı` exists so that an accent can be set over it, so an accent is exactly
    // where it means an ordinary `i`.
    assert_eq!(compose_diacritics("y\u{02C7}\u{0131}"), "yǐ");
    assert_eq!(compose_diacritics("d\u{00AF}\u{0131}ng"), "dīng");
}

#[test]
fn an_accent_with_no_letter_after_it_stays_where_it_is() {
    assert_eq!(compose_diacritics("\u{00AF} x"), "\u{00AF} x");
    assert_eq!(compose_diacritics("2\u{00AF}"), "2\u{00AF}");
    assert_eq!(compose_diacritics("\u{00AF}5"), "\u{00AF}5");
}

#[test]
fn a_pair_unicode_has_no_single_character_for_keeps_both() {
    // `q` takes no macron in Unicode; the answer is the letter and the
    // combining mark, not the letter alone.
    assert_eq!(compose_diacritics("a\u{00AF}q"), "aq\u{0304}");
}

#[test]
fn ascii_lookalikes_are_never_treated_as_accents() {
    // These carry accents in one TeX font encoding and are ordinary characters
    // everywhere else; composing them would corrupt code and paths.
    for text in ["`ls`", "a^b", "~/dir", "\"quoted\""] {
        assert_eq!(compose_diacritics(text), text);
    }
}

#[test]
fn text_without_accents_comes_back_untouched() {
    let text = "Plain text — with an em dash, «quotes» and Кириллица.";
    assert_eq!(compose_diacritics(text), text);
}

#[test]
fn small_capitals_and_old_style_figures_become_ordinary_characters() {
    assert_eq!(
        normalize_presentation_forms("\u{F761}\u{F762}\u{F77A}"),
        "ABZ"
    );
    assert_eq!(normalize_presentation_forms("\u{F730}\u{F739}"), "09");
}

#[test]
fn the_slices_of_a_big_brace_are_drawing_and_are_dropped() {
    assert_eq!(
        normalize_presentation_forms("x\u{F8F1}\u{F8F2}\u{F8F3}\u{F8F4}y"),
        "xy"
    );
    assert_eq!(normalize_presentation_forms("\u{F8E8}\u{F8E9}"), "®©");
}

#[test]
fn private_use_outside_adobes_range_is_left_alone_to_be_reported() {
    // This is the unrecoverable kind: a font that maps its glyphs into the
    // private use area and tells nobody what they are. Rewriting it would hide
    // the damage instead of reporting it.
    let text = "\u{E2BC}\u{0F56}\u{0F0B}";
    assert_eq!(normalize_presentation_forms(text), text);
}

#[test]
fn a_document_with_no_type1_font_is_not_rewritten() {
    // Nothing to repair must cost nothing: no re-serialization, no new bytes.
    let repair = repair(&minimal_pdf(None));
    assert!(repair.bytes.is_none());
    assert_eq!(repair.fonts, 0);
    assert!(!repair.unmapped, "a standard font needs no warning");
    assert_eq!(repair.page_count, Some(1));
}

#[test]
fn a_font_that_hides_its_encoding_is_repaired_and_read() {
    let repair = repair(&minimal_pdf(Some(PROGRAM)));
    assert_eq!(repair.fonts, 1);
    let bytes = repair.bytes.expect("the document should be rewritten");

    // The vector went in as /Differences, and the engine reads it: code 12 is
    // the `fi` ligature, which StandardEncoding drops on the floor.
    let result = pdf_inspector::process_pdf_mem(&bytes).expect("convert");
    let markdown = result.markdown.unwrap_or_default();
    assert!(markdown.contains("first"), "{markdown:?}");
    assert!(markdown.contains('–'), "the en dash too: {markdown:?}");
}

#[test]
fn without_the_repair_the_same_document_loses_those_characters() {
    // The measurement the repair exists for, in miniature: this is what the
    // engine does on its own.
    let source = minimal_pdf(Some(PROGRAM));
    let result = pdf_inspector::process_pdf_mem(&source).expect("convert");
    let markdown = result.markdown.unwrap_or_default();
    assert!(markdown.contains("rst"), "{markdown:?}");
    assert!(!markdown.contains("first"), "{markdown:?}");
}

#[test]
fn a_font_beyond_recovery_is_reported_rather_than_repaired() {
    // A descriptor without a Type 1 program: subset, renamed, and nothing in
    // the file says what its codes mean.
    let repair = repair(&minimal_pdf(Some(b"")));
    assert_eq!(repair.fonts, 0);
    assert!(repair.unmapped);
}

/// A one-page PDF that says `first` and an en dash, with the `fi` ligature and
/// the dash written as the codes a TeX Type 1 font uses for them.
///
/// With `program`, the page's font is an embedded Type 1 font carrying that
/// font program and declaring no encoding — the case the repair exists for. An
/// empty program stands for a font whose descriptor embeds nothing. Without
/// one, the font is Helvetica, which needs no repair at all.
fn minimal_pdf(program: Option<&[u8]>) -> Vec<u8> {
    use lopdf::{Object, Stream, dictionary};

    let mut document = Document::with_version("1.5");
    let font = match program {
        None => document.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
        }),
        Some(program) => {
            let file = document.add_object(Stream::new(
                dictionary! { "Length1" => program.len() as i64 },
                program.to_vec(),
            ));
            let descriptor = document.add_object(dictionary! {
                "Type" => "FontDescriptor",
                "FontName" => "ABCDEF+TestFont",
                "Flags" => 4,
                "FontFile" => file,
            });
            document.add_object(dictionary! {
                "Type" => "Font",
                "Subtype" => "Type1",
                "BaseFont" => "ABCDEF+TestFont",
                "FontDescriptor" => descriptor,
            })
        },
    };
    let resources = document.add_object(dictionary! {
        "Font" => dictionary! { "F1" => font },
    });
    // Code 12 is the `fi` ligature and 123 the en dash in the font's own
    // vector; StandardEncoding has nothing at 12 and a brace at 123. Several
    // lines, because a page with one or two of them is classified as having no
    // text at all.
    let content = b"BT /F1 12 Tf 72 720 Td (\x0crst light) Tj ET\n\
        BT /F1 12 Tf 72 700 Td (a plain line of prose) Tj ET\n\
        BT /F1 12 Tf 72 680 Td (1841{1868 and more prose) Tj ET\n\
        BT /F1 12 Tf 72 660 Td (a fourth line to be sure) Tj ET\n"
        .to_vec();
    let contents = document.add_object(Stream::new(dictionary! {}, content));
    let pages_id = document.new_object_id();
    let page = document.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        "Contents" => contents,
        "Resources" => resources,
    });
    document.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page.into()],
            "Count" => 1,
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
