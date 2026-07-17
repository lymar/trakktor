use clap::{Arg, ArgAction, Command};

use super::*;

/// A miniature CLI standing in for the real one, exercising the traversal:
/// nested subcommands, a positional, a flag, and an option with a default.
fn sample() -> Command {
    Command::new("trakktor")
        .about("Helper commands for coding agents: feeds and more.")
        .long_about(
            "Helper commands for coding agents: feeds and more.\n\nUse when a \
             task needs one of them.",
        )
        .subcommand(
            Command::new("feed")
                .about("Work with feeds.")
                .long_about("Work with feeds.\n\ndiscover, read, mark-read.")
                .subcommand(
                    Command::new("read")
                        .about("Read a feed and return publications.")
                        .arg(Arg::new("url").required(true).value_name("url"))
                        .arg(
                            Arg::new("all")
                                .long("all")
                                .action(ArgAction::SetTrue)
                                .help("Include read items."),
                        )
                        .arg(
                            Arg::new("fields")
                                .long("fields")
                                .default_value("minimal")
                                .value_name("list")
                                .help("Fields to show."),
                        ),
                ),
        )
}

#[test]
fn narrative_lists_commands_without_reference() {
    let guide = render_guide(&sample(), false);
    assert!(guide.starts_with("# trakktor"));
    assert!(guide.contains("## Commands"));
    assert!(guide.contains("### trakktor feed"));
    // Long-about workflow text comes through.
    assert!(guide.contains("discover, read, mark-read."));
    // No reference section without `--full`.
    assert!(!guide.contains("## Reference"));
}

#[test]
fn full_reference_exposes_flags_values_and_defaults() {
    let guide = render_guide(&sample(), true);
    assert!(guide.contains("## Reference"));
    assert!(guide.contains("### trakktor feed read"));
    assert!(guide.contains("`<url>`"));
    assert!(guide.contains("required"));
    assert!(guide.contains("`--all`"));
    assert!(guide.contains("`--fields <list>`"));
    assert!(guide.contains("default: minimal"));
}

#[test]
fn stub_has_strict_frontmatter_and_points_at_the_binary() {
    let stub = render_stub(&sample());
    assert!(stub.starts_with("---\n"));
    assert!(stub.contains("name: trakktor\n"));
    assert!(stub.contains("allowed-tools: Bash(trakktor:*)\n"));
    // Description is the curated trigger line, NOT the root `about`.
    assert!(stub.contains(
        "description: \"Use trakktor, a predictable, automation-friendly CLI \
         toolbox"
    ));
    assert!(!stub.contains("feeds and more.")); // sample()'s `about`
    // Body is a pointer, not a reference dump.
    assert!(stub.contains("trakktor skill show"));
    assert!(stub.contains("trakktor skill show --full"));
    assert!(!stub.contains("--fields"));
}

#[test]
fn stub_description_is_a_bounded_trigger_line() {
    let desc = stub_description(&sample());
    // What it does + when to use it — the field an agent matches on.
    assert!(
        desc.starts_with(
            "Use trakktor, a predictable, automation-friendly CLI toolbox"
        ),
        "got: {desc:?}"
    );
    // The trigger line must name each capability so an agent discovers the
    // skill for every kind of task; keep it in sync by hand (STUB_DESCRIPTION).
    assert!(
        desc.contains("feed") && desc.contains("transcribe"),
        "got: {desc:?}"
    );
    assert!(desc.contains("Reach for it whenever"));
    // Strict-frontmatter guarantees: non-empty, bounded, one clean line.
    assert!(!desc.is_empty() && desc.chars().count() <= DESCRIPTION_MAX);
    assert!(!desc.chars().any(|c| c.is_control()));
}

#[test]
fn yaml_escape_escapes_quotes_and_backslashes() {
    // The only characters illegal in a double-quoted YAML scalar.
    assert_eq!(yaml_escape(r#"a "q" \ b"#), r#"a \"q\" \\ b"#);
    assert_eq!(yaml_escape("plain text"), "plain text");
}

#[test]
fn clamp_chars_truncates_long_input_with_ellipsis() {
    let clamped = clamp_chars(&"x".repeat(2000), DESCRIPTION_MAX);
    assert_eq!(clamped.chars().count(), DESCRIPTION_MAX);
    assert!(clamped.ends_with('…'));
    // Short input is returned unchanged.
    assert_eq!(clamp_chars("short", DESCRIPTION_MAX), "short");
}

#[test]
fn collapse_squeezes_whitespace_and_control_chars() {
    assert_eq!(collapse("a\n\tb   c\n"), "a b c");
    // Non-whitespace control chars (NUL, BEL, ESC, DEL) become spaces, so a
    // generated description stays valid inside a double-quoted YAML scalar.
    assert_eq!(collapse("a\u{0000}b\u{0007}\u{001b}c\u{007f}d"), "a b c d");
}

#[test]
fn skill_name_rules() {
    assert!(is_valid_skill_name("trakktor"));
    assert!(!is_valid_skill_name("Trakktor"));
    assert!(!is_valid_skill_name("claude"));
    assert!(!is_valid_skill_name(""));
}
