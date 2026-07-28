# Instructions for coding agents

## What this is

`trakktor` is a Rust command-line utility that exposes helper functions for
coding agents (Claude Code, OpenCode, etc.): RSS/Atom/JSON feeds, speech
(ASR/TTS), OCR, file/document conversion, and more. This repository holds the
**implementation**.

## Committing: NEVER commit or push unprompted

**Never run `git commit` or `git push` on your own initiative.** Commit or push
**only** when the user explicitly asks for it **in that same request** ("commit
this", "commit and push", etc.). This is a hard rule, in this repo and the docs
repo.

- An earlier commit in the session grants **no** standing permission for later
  ones. Each commit requires its own fresh, explicit request — even if you just
  committed something a moment ago.
- Finishing a task, green tests, or "the change looks done" is **not** a reason
  to commit. Leave the work in the working tree, report what changed, and let
  the user decide when (and whether) to commit.
- When in doubt, do not commit — stop and ask.

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
- `core/<subsystem>/design.md` — specs for core-internal subsystems that are
  not CLI features (e.g. `core/audio/design.md`).
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
- **Build and test the bin with `-p trakktor`** when enabling the `metal`/`burn`
  features: they are features of the **bin crate**, so from the workspace root
  `cargo build --release --features metal,burn` fails ("package does not have
  these features") — and piping the output (e.g. through `tail`) swallows the
  non-zero exit, leaving a stale binary that looks freshly built; check the
  binary's mtime when in doubt. Use
  `cargo build --release -p trakktor --features metal,burn`. `trakktor_core`
  has per-engine feature names instead (e.g.
  `cargo test -p trakktor_core --features tts-burn`).
- Keep `trakktor` (bin) thin; put real logic in `trakktor_core`.
- Machine-readable output (e.g. JSON) is a first-class requirement — the primary
  consumers are agents (see `conventions/output.md`).
- **Every ASR engine drives the live transcription progress line through the
  shared `trakktor::asr::progress` module — never a per-engine copy.** That
  line MUST include the **time-remaining estimate** (`~mm:ss left`) whenever the
  audio length is known (it is omitted only for a length-less source such as an
  `ffmpeg` pipe). A new transcriber wires its progress callback into
  `progress::live_reporter` and closes with `progress::finish_line`; model
  downloads use `progress::download_progress`. Do not reintroduce a local
  `transcribe_progress`/`clock`/`download_progress`.
- **Every model, checkpoint, and downloaded asset comes through the shared
  `trakktor_core::download` module — never a per-engine copy.** One
  `Download::new(url, target).fetch(progress)` gets resume after an
  interruption, retries with backoff, an atomic rename, optional BLAKE3/MD5
  verification, and parallel range requests; a local `download.rs` with its own
  `reqwest` client would get none of it and would drift. Progress is reported
  **only** through that module's `Progress` type, which the CLI draws with
  `progress::download_progress`. Do not reintroduce a per-engine
  `download_file`/`http_client`, and do not point the feed HTTP client at large
  files — it is built for small, size-capped responses.
- **Any burn run on a GPU device must call `burn_notice::announce_cold_gpu_start`
  before loading the model** — it prints the kernel compile/autotune heads-up
  (burn autotunes per kernel shape, so a not-yet-tuned model re-tunes even when
  the cache holds another's) and installs the panic hook that mutes cubecl's
  harmless autotune-internal panics. The CPU backend does not autotune, so it is
  not called there.
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
- Some agent- and user-facing surfaces are **hand-written, not generated** from
  the command tree, so a new feature does not reach them on its own — update
  them **in the same change** whenever trakktor gains or changes a capability:
  the **`README`**, and the **skill's discovery `description`** (the trigger
  line an agent matches to decide whether to load the skill — the
  `STUB_DESCRIPTION` constant in `trakktor_core/src/skill/render.rs`, plus the
  top-level `about` that opens the narrative guide). The generated `skill show
  --full` reference lists a new command automatically, but it does **not** make
  the skill discoverable for it — that is what the trigger `description` is for.
- The GitHub repository **About description and topics** are hand-maintained
  too, and live outside the repo. Whenever trakktor gains or changes a
  capability, **propose an updated About text and topic set to the user** —
  do not change repository settings on your own; apply them only when the user
  agrees (`gh repo edit --description … --add-topic …`). Keep the About in sync
  with the README intro and the CLI's top-level `about`; mind GitHub's cap of
  20 topics (adding one may mean suggesting which existing topic to drop).

## Third-party attribution

trakktor reuses third-party work — **ported/vendored implementations** (e.g. a
model or algorithm reimplemented from a reference), **embedded or downloaded
assets and model weights**, and **crate dependencies**. The root **`NOTICE`**
file records these with their license, copyright holder, and how trakktor uses
them.

**Keep `NOTICE` in sync in the same change** whenever you add or change such a
dependency: a new ported implementation, a newly embedded asset, a model that is
downloaded at runtime, or a crate whose license requires attribution. Preserve
each work's copyright and permission notice as its license (MIT, Apache-2.0,
etc.) requires. Attribute ported/adapted code at its source too, in the module
`//!` doc-comment (name the upstream project and its license) — but **not** in
clap `///` doc-comments, which are agent-facing (see the leak rule above). When
a feature has a user-facing home, add a credit to `docs/acknowledgments.md`
(summarized in the `README`'s `## Acknowledgments`, which links there) and,
for academic models that request it, the citation.

## Rust module and test layout

- **No `mod.rs`, ever.** Every module is a single file named after the module
  (`some_module.rs`). If it has submodules, they live in a sibling directory of
  the same name (`some_module/`). For example, a `whisper` module with children
  is `whisper.rs` plus a `whisper/` directory — never `whisper/mod.rs`.
- **Extract non-trivial tests into their own file.** If a file's `#[cfg(test)]`
  code is more than 30 lines, move it into a dedicated child module in its own
  file: declare `#[cfg(test)] mod tests;` in the parent and put the tests in
  `some_module/tests.rs`. Only short test blocks may stay inline. (Combined with
  the no-`mod.rs` rule, adding a `tests` submodule to `foo.rs` means creating
  `foo/tests.rs`.)

## Branches

- `trunk` — default/integration branch; the active line of work.
- `boa` — the previous default branch, kept for reference.
- `legacy` — previous incarnation, kept for reference.
