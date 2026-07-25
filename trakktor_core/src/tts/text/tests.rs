//! Tests for reading paragraphs off an input document.

use std::path::Path;

use super::*;

#[test]
fn plain_text_puts_one_paragraph_per_line() {
    let text = "Первый абзац.\n\nВторой абзац.\n   \nТретий.\n";

    let paragraphs = paragraphs(text, TextFormat::Plain);

    assert_eq!(paragraphs, ["Первый абзац.", "Второй абзац.", "Третий."]);
}

#[test]
fn a_paragraph_reads_as_running_prose() {
    // Markdown joins the wrapped lines of one block; both formats collapse
    // repeated spaces.
    let text = "Речь в этой\nкниге пойдет   главным\nобразом о хоббитах.";

    let paragraphs = paragraphs(text, TextFormat::Markdown);

    assert_eq!(
        paragraphs,
        ["Речь в этой книге пойдет главным образом о хоббитах."]
    );
}

#[test]
fn markdown_breaks_paragraphs_on_blank_lines() {
    let text = "Первый абзац,\nперенесённый по словам.\n\nВторой абзац.";

    let paragraphs = paragraphs(text, TextFormat::Markdown);

    assert_eq!(
        paragraphs,
        ["Первый абзац, перенесённый по словам.", "Второй абзац."]
    );
}

#[test]
fn markup_comes_off_but_the_text_stays() {
    // Built line by line: a long literal is at the mercy of the formatter,
    // and here the line breaks are the fixture.
    let text = [
        "# Заголовок",
        "",
        "Абзац с **жирным**, *курсивом*, `кодом` и \
         [ссылкой](https://example.com).",
        "",
        "> Цитата тоже читается.",
    ]
    .join("\n");
    let text = text.as_str();

    let paragraphs = paragraphs(text, TextFormat::Markdown);

    assert_eq!(
        paragraphs,
        [
            "Заголовок",
            "Абзац с жирным, курсивом, кодом и ссылкой.",
            "Цитата тоже читается.",
        ]
    );
}

#[test]
fn a_heading_and_every_list_item_is_its_own_paragraph() {
    // Neither is separated by a blank line, yet each is a separate utterance:
    // a list read in one breath is unlistenable.
    let text = "## Список\n- первый пункт\n- второй пункт\n1. третий\n";

    let paragraphs = paragraphs(text, TextFormat::Markdown);

    assert_eq!(
        paragraphs,
        ["Список", "первый пункт", "второй пункт", "третий",]
    );
}

#[test]
fn code_blocks_and_tables_keep_their_content() {
    let text = "\
Перед кодом.

```rust
let x = 1;
```

| Колонка | Значение |
|---|---|
| раз | два |
";

    let paragraphs = paragraphs(text, TextFormat::Markdown);

    assert_eq!(
        paragraphs,
        ["Перед кодом.", "let x = 1;", "Колонка Значение", "раз два",]
    );
}

#[test]
fn front_matter_comments_and_rules_are_dropped() {
    let text = "\
---
title: Заметка
---

Текст заметки.

<!-- скрытый комментарий -->

---

Хвост.
";

    let paragraphs = paragraphs(text, TextFormat::Markdown);

    assert_eq!(paragraphs, ["Текст заметки.", "Хвост."]);
}

#[test]
fn an_unclosed_opening_rule_is_not_front_matter() {
    let text = "---\n\nОбычный текст.\n";

    let paragraphs = paragraphs(text, TextFormat::Markdown);

    assert_eq!(paragraphs, ["Обычный текст."]);
}

#[test]
fn links_images_and_wiki_links_leave_their_text() {
    let text = [
        "См. ![схему](img/a.png) и [заметку](notes/a.md), а также \
         [[note|псевдоним]] и [[просто-заметку]].",
        "",
        "[ref]: https://example.com",
    ]
    .join("\n");
    let text = text.as_str();

    let paragraphs = paragraphs(text, TextFormat::Markdown);

    assert_eq!(
        paragraphs,
        ["См. схему и заметку, а также псевдоним и просто-заметку.",]
    );
}

#[test]
fn underscores_inside_a_word_survive() {
    let text = "Функция _важная_ зовётся some_long_name и всё.";

    let paragraphs = paragraphs(text, TextFormat::Markdown);

    assert_eq!(paragraphs, ["Функция важная зовётся some_long_name и всё."]);
}

#[test]
fn the_extension_decides_the_format() {
    let text = "Строка одна.\nСтрока два.";

    assert_eq!(
        TextFormat::Auto.resolve(Some(Path::new("a/b.txt")), text),
        TextFormat::Plain
    );
    assert_eq!(
        TextFormat::Auto.resolve(Some(Path::new("a/b.MD")), text),
        TextFormat::Markdown
    );
    // An explicit choice ignores both the extension and the text.
    assert_eq!(
        TextFormat::Markdown.resolve(Some(Path::new("a/b.txt")), text),
        TextFormat::Markdown
    );
}

#[test]
fn without_an_extension_the_text_decides() {
    // Markup of any kind, or blank-line separation: Markdown.
    assert_eq!(
        TextFormat::Auto.resolve(None, "# Заголовок\nтекст"),
        TextFormat::Markdown
    );
    assert_eq!(
        TextFormat::Auto.resolve(None, "Абзац.\n\nЕщё абзац."),
        TextFormat::Markdown
    );

    // Long unbroken lines: each is a paragraph of its own.
    let long = format!("{0}\n{0}", "слово ".repeat(30));
    assert_eq!(TextFormat::Auto.resolve(None, &long), TextFormat::Plain);

    // Short lines with no blank line are wrapped prose, not paragraphs — the
    // undecidable case falls to Markdown, which keeps them together.
    assert_eq!(
        TextFormat::Auto.resolve(None, "короткая строка\nи ещё одна"),
        TextFormat::Markdown
    );
}
