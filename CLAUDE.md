# Instructions for coding agents

## What this is

`trakktor` is a Rust command-line utility that exposes helper functions for
coding agents (Claude Code, OpenCode, etc.): RSS/Atom/JSON feeds, speech
(ASR/TTS), OCR, file/document conversion, and more. This repository holds the
**implementation**.

## Source of truth: the design docs

The project's design, specifications, and decisions live in a **separate
documentation repository**: `../trakktor_project` (a sibling directory; add it to
the session with `--add-dir` if it is not already available).

**This is spec-first development. The docs are the source of truth.**

- Before implementing or changing behavior, read the relevant design doc.
- If the implementation must diverge from the docs, update the docs **first**
  (or within the same logical change) — never let code and docs drift apart.
- **This `CLAUDE.md` is the only file in this repository that may reference the
  design docs.** Do not mention the docs repo, ADR identifiers, `design.md`,
  convention filenames, or `§`-sections anywhere else — not in code or
  doc-comments, not in the `README`, not in commit messages. This repository is
  (potentially) public and must read cleanly on its own; the docs repo is
  private. Doc-comments are especially sensitive: clap renders them into
  `--help`, and `trakktor skill show` reproduces them **verbatim** for agents.

Key documents (in `../trakktor_project/docs/`):

- `overview.md` — what the project is and its design principles.
- `architecture.md` — code organization (this workspace and its crates).
- `features/<feature>/design.md` — per-feature specs (e.g.
  `features/feed/design.md`).
- `adr/` — Architecture Decision Records (rationale for big decisions).
- `conventions/` — cross-cutting rules: `cli.md`, `output.md`,
  `error-handling.md`, `http.md`, `working-directory.md`, `language.md`.

> The docs are written in **Russian**. Read them as the spec, but all code and
> code-repo content here must be in **English** (see Language below).

## Language

**English only**, without exception, everywhere in this repository:

- comments and identifiers (types, functions, variables, modules);
- `README` and any in-repo documentation;
- commit messages.

(The Russian language is used only in the `trakktor_project` docs repo. See
`../trakktor_project/docs/conventions/language.md`.)

## Architecture

Cargo **workspace** with a flat crate layout at the root:

- `trakktor` — **bin**: CLI entry point, argument parsing, configuration,
  output formatting. A thin layer; delegates logic to `trakktor_core`.
- `trakktor_core` — **lib**: all functionality, independent of the CLI and
  reusable.

New crates are added as workspace members as needed (especially if parts are
published separately). Rationale: `../trakktor_project/docs/adr/0002-cargo-workspace.md`.

## Working conventions

- Format with `cargo fmt` before committing. `rustfmt.toml` uses unstable
  features (e.g. `format_strings`, `wrap_comments`, edition 2024), so formatting
  requires **nightly** rustfmt (`cargo +nightly fmt`).
- Keep `trakktor` (bin) thin; put real logic in `trakktor_core`.
- Machine-readable output (e.g. JSON) is a first-class requirement — the primary
  consumers are agents (see `conventions/output.md`).
- Describe errors with `thiserror` (typed enums); map them to the output contract
  (stable `code` + exit) at the bin boundary. See `conventions/error-handling.md`.
- The CLI is self-documenting and the skill is generated from it (ADR-0003): a
  command, flag, value, or help string lives once in `clap` and flows to both
  `--help` and `trakktor skill show`. **After changing any clap doc-comment or
  help text, you MUST verify what reached the agent-facing output** — run
  `trakktor skill show`, `trakktor skill show --full`, and the relevant
  `--help`, and confirm nothing internal leaked. The test
  `no_internal_doc_references_leak_into_user_facing_text` enforces this; see
  `conventions/cli.md`.

## Branches

- `agent-cli` — current direction (this pivot); the active line of work.
- `trunk` — default/integration branch.
- `legacy` — previous incarnation, kept for reference.
