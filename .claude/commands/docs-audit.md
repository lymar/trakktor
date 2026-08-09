---
description: Audit all user-facing documentation — docs/, README, CLI/skill text, GitHub About — for contradictions, gaps, and clarity
argument-hint: [feature]
---

# Documentation audit

Audit every user-facing documentation surface of this repository for three
failure classes:

- **contradictions** — a document disagrees with the binary, or with another
  document;
- **gaps** — something the CLI can do that no document describes, a missing
  example, a dead link, a missing attribution;
- **unclarity** — the intended reader cannot get from their problem to a
  working command.

The deliverable is a **report of findings**. Do not edit, fix, or commit
anything during the audit itself — but close the report by offering to apply
the fixes right away (step 6); the user decides what changes.

Scope: `$ARGUMENTS`. If it names a feature or command (`ocr`, `tts`, `feed`,
…), audit that feature's pages, its slice of the CLI tree, and each
cross-cutting surface where it appears (README, skill description, About).
If empty, audit the whole project.

## The surfaces and their readers

Judge each surface against its own audience: a finding is "this reader fails
here", not "I would have phrased it differently".

1. **`README.md`** — the front door; people and search engines. Must say what
   the project is, emphasize its strengths, and link every feature to its page
   under `docs/features/` — a capability without a line and a link here is
   invisible.
2. **`docs/features/**` and `docs/acknowledgments.md`** — the full story, for
   a person who wants to know what is in the project and how to use it (agents
   read it too). Every page's contract: a short statement of what the feature
   is first, then all its capabilities and how to use them, **with examples**.
3. **The CLI text** — clap doc-comments in `trakktor/src/cli.rs`, rendered
   into `--help` and `trakktor skill show [--full]`. Read by coding agents,
   and not always strong ones. The workflow it must survive: a user states a
   problem in their own words; the agent reads the skill text and from it
   alone must find the right command and build a correct call.
4. **Skill discovery** — `STUB_DESCRIPTION` in
   `trakktor_core/src/skill/render.rs` and the root command's `about`. This is
   the trigger an agent matches to decide whether to load the skill at all: a
   capability missing there does not exist for agents, however well the full
   reference describes it.
5. **GitHub About and topics** — outside the repo; people finding it. Read
   with `gh repo view --json description,repositoryTopics`.

## Step 0 — build what you audit

```sh
cargo build --release -p trakktor --features metal,burn
```

Drop `metal` on non-Apple hardware; use the fullest feature set that builds —
feature-gated commands only render when built in. Run every check against
`target/release/trakktor`, never an installed binary: an installed one is a
different version, and a piped build swallows a failure — check the exit
code, and the binary's mtime when in doubt.

## Step 1 — inventory

The generated reference is the complete, current command tree:

```sh
target/release/trakktor skill show --text        # narrative guide
target/release/trakktor skill show --full --text # + full reference
```

Save both to a scratch directory. The full reference lists every command,
flag, value, and default the binary exposes; it is the baseline the docs are
measured against, not the other way round. List `docs/features/` for the
pages inventory. Also grep `cli.rs` for `hide`: anything hidden from help
must be hidden on purpose — name it in the report.

## Step 2 — mechanical checks

Cheap, grep-shaped, and where most drift is caught. Record each finding with
file:line.

**1. CLI coverage is absolute.** Every command, subcommand, flag, and enum
value in the full reference is described on that feature's `docs/features/`
page(s) — every flag at least named with its meaning, the load-bearing ones
with an example — and every top-level command has its line and link in
`README.md`. The reverse as well: everything the docs describe exists in the
CLI under that name with that default. **No threshold: one undocumented flag
is a finding.** Report the counts (commands and flags documented / total).

**2. Examples are runnable as written.** Extract every `trakktor` invocation
from fenced blocks in `README.md` and `docs/**`. For each: the subcommand
path exists, every flag exists on that command, every value is legal for its
flag. Placeholder inputs (`page.png`, `article.md`) are fine. Sample
*outputs* shown in the docs (JSON fields, page markers, file layouts) must
match what the binary actually emits — verify against the output types in
the code, or by a real run when the needed models are already in the local
cache. Never trigger a large model download for an audit.

**3. Links and anchors resolve.** Every relative link in `README.md` and
`docs/**` points at an existing file; every `#fragment` at a real heading in
its target.

**4. Repeated facts agree.** Collect every fact stated in more than one
place — engine counts, language counts and lists, model sizes, timings,
defaults, the Rust version floor (against `rust-version` in `Cargo.toml`),
feature-flag names (against the crate manifests) — into a table of fact →
value → places, and flag every row whose values disagree across `README.md`,
feature pages, engine pages, the CLI `about`, `STUB_DESCRIPTION`, and the
GitHub About.

**5. Nothing internal, nothing foreign.** The guard test covers only CLI
surfaces:

```sh
cargo test -p trakktor --test cli no_internal_doc_references_leak_into_user_facing_text
```

Run it, then apply the same needles to what it does not cover — `README.md`
and `docs/**`: no references to a private docs repository, ADR identifiers,
`design.md`/convention filenames, or `§`-sections; no Cyrillic (this repo is
English-only); no paths or names from anyone's local machine; every example
invented for this repo rather than taken from user material.

**6. Attribution is complete.** Every model or ported implementation the
docs say trakktor downloads or contains appears in `NOTICE` and in
`docs/acknowledgments.md`, and the README `## Acknowledgments` paragraph
agrees with both.

## Step 3 — the skill text is valid

Install the stub **into a scratch directory, never into the repo**:

```sh
cd <scratch> && <repo>/target/release/trakktor skill install claude
```

Check the written `SKILL.md`: the YAML frontmatter parses, `name` and
`description` are present, the description is within the 1024-character limit
(`DESCRIPTION_MAX` in `render.rs`) and not clamped mid-sentence, and the stub
points back at `trakktor skill show`. In the saved `show` outputs: code
fences balanced, every top-level command present, no truncation — and the
discovery `description` names every capability area (compare it against the
top-level command list; check 1's zero-threshold rule applies here too).
Run the skill test suite: `cargo test -p trakktor_core skill`.

## Step 4 — read for clarity

Judgment, not grep. Read each surface as its reader.

- **README as the front door.** Does the first screen say what this is, who
  it serves, and why it is good? Is every strength stated true of the current
  binary — and are the strengths *stated*, not buried? Does every feature get
  its line and its link?
- **Each docs page against its contract.** A short statement of what it is
  first; then capabilities and usage; examples present for everything
  load-bearing. Flag pages that assume unexplained context, bury the main use
  case, use a term defined nowhere, or drift into implementation talk the
  reader cannot act on.
- **The modest-agent walkthrough.** Take at least one realistic user problem
  per top-level command ("transcribe this voice message", "read this
  photographed page", "turn this article into audio", "cut the silence out
  of this recording", "does this blog have a new post", …). For each, using
  **only** the skill text and `--help` output — no source code, no `docs/` —
  write down the exact command an agent would build, then verify it against
  the real CLI. Every hesitation is a finding: information that exists but
  must be inferred, two texts that disagree, a term only an implementer
  knows, a flag whose consequence is stated nowhere, a default you had to
  guess.

## Step 5 — GitHub About and topics

```sh
gh repo view --json description,repositoryTopics
```

Compare with the README intro and the root `about`: the description must name
the current capability set and match their emphasis; the topics must cover
the feature areas within GitHub's cap of 20 (adding one may mean proposing
which to drop). If drifted, put a ready
`gh repo edit --description '…' --add-topic …` line into the report.
**Never apply it yourself** — repository settings change only when the user
agrees.

## Step 6 — the report

Order findings by reader impact within three sections:

1. **Contradictions** — doc vs. binary, doc vs. doc. For each: file:line,
   both statements, which one is right, a one-line fix.
2. **Gaps** — undocumented CLI surface (with the coverage counts from
   check 1), missing examples, dead links, missing attribution, About drift.
3. **Clarity** — for each: the reader, the concrete scenario in which they
   fail, file:line, a one-line fix.

Close with a per-surface verdict (clean, or the finding numbers) and the
About proposal if any — and then **immediately offer to make the changes for
the found problems**: say which findings you would fix and in what order, so
the user only has to say yes (or pick a subset). Apply fixes only after the
user chooses; committing is never part of the audit.

Independent checks parallelize well: one subagent per feature for checks 1–2
and the per-page reads, with the cross-cutting checks (the facts table,
README, skill, About) kept in the main session.
