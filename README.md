# trakktor

`trakktor` is a Rust command-line utility that exposes helper functions for
coding agents (Claude Code, OpenCode, etc.). It is built to be **predictable and
automation-friendly**: stable commands and flags, machine-readable output, and
meaningful exit codes.

## Install

Requires a stable Rust toolchain **≥ 1.88** (edition 2024; nightly is only
needed for `cargo fmt`, not to build). The binary installs into `~/.cargo/bin`,
which must be on your `PATH`.

From git:

```sh
cargo install --locked --git https://github.com/lymar/trakktor.git \
  --branch agent-cli trakktor
```

- `trakktor` (the trailing word) is the package to install — the workspace root
  is virtual, so it must be named.
- `--branch agent-cli` is required for now: the code is not yet on the default
  `trunk` branch. Drop the flag once it lands there.
- `--locked` builds with the exact dependency versions pinned in `Cargo.lock`
  (reproducible); omit it to pull the latest semver-compatible versions.

From a local clone (installs whatever is checked out):

```sh
git clone -b agent-cli https://github.com/lymar/trakktor.git
cd trakktor
cargo install --locked --path trakktor
```

Verify, then remove if needed:

```sh
trakktor --version
cargo uninstall trakktor
```

On Linux, the build pulls in `reqwest`'s default TLS, which needs OpenSSL
(`pkg-config` plus `libssl-dev`/`openssl-devel`); macOS uses the system TLS and
needs nothing extra.

## Layout

A Cargo workspace with a flat crate layout:

- `trakktor` — the CLI binary: argument parsing, configuration, output
  formatting. A thin layer over the library.
- `trakktor_core` — the library: all functionality, independent of the CLI.

## Build & test

```sh
cargo build --workspace
cargo test --workspace
```

Formatting uses unstable rustfmt features, so it requires nightly:

```sh
cargo +nightly fmt --all
```

## Output and exit codes

The default output is human-readable text; `--json` switches to machine-readable
JSON (`--json --pretty` indents it). Data is written to stdout, errors to stderr.

- `0` — success.
- `1` — runtime/validation error (network, parsing, I/O, bad value). Respects
  `--json`: `{ "error": { "code": "…", "message": "…" } }`.
- `2` — usage error (unknown flag, missing argument); always text, from the
  argument parser.

Global options (usable before or after the command): `--work-dir <path>` (also
`TRAKKTOR_DIR`; default `./.trakktor`), `--json`, `--pretty`.

## `feed` — RSS / Atom / JSON Feed

### Discover feeds on a page

```sh
trakktor feed discover https://example.com
```

Returns the feeds declared on the page (`url`, `type`, `title`). An empty result
is success.

### Read a feed

```sh
trakktor feed read https://example.com/feed.xml
trakktor feed read https://example.com/feed.xml --all --fields all --json
```

Accepts a feed URL or a regular page (autodiscovery applies, reading the first
feed found). Each publication carries a stable `uid` and an `is_read` flag.

- By default only **unread** publications are returned; `--all` includes read
  ones.
- `--fields <list>` selects output fields: a comma-separated list of
  `uid,is_read,title,link,published,updated,summary,content,authors`, or the
  special values `minimal` (default, `uid,title,link`) and `all`.

The `uid` is `hex(BLAKE3(feed_key ‖ 0x00 ‖ tag ‖ 0x00 ‖ item_key))` and is
stable across runs for the same feed + entry.

### Mark publications read

```sh
trakktor feed mark-read <uid> [<uid>...]
```

Idempotent. Read state is stored as plain files under `<work-dir>/feed/`, sharded
by uid; nothing else is needed (no database).

## Typical agent workflow

```sh
trakktor feed discover https://example.com            # find a feed
trakktor feed read https://example.com/feed.xml --json # read unread items
# … process the items …
trakktor feed mark-read <uid1> <uid2>                  # mark them handled
```

On the next `read`, marked publications are no longer returned.

## `skill` — generate the agent skill

trakktor can describe itself to a coding agent as an Agent Skill. The content is
generated from the live `clap` definition, so it always matches the installed
binary — there is no hand-written reference to drift out of date.

### Show the skill

```sh
trakktor skill show          # narrative guide: what trakktor is, when, workflows
trakktor skill show --full   # plus the full command/flag/value reference
```

Markdown is printed to stdout; `--json` wraps it as `{ "content": "…" }`.

### Install the stub

```sh
trakktor skill install claude            # ./.claude/skills/trakktor/SKILL.md
trakktor skill install agents            # ./.agents/skills/trakktor/SKILL.md
trakktor skill install claude --global   # ~/.claude/skills/trakktor/SKILL.md
trakktor skill install claude --force    # overwrite an existing stub
```

Writes a thin discovery stub to `<dir>/skills/trakktor/SKILL.md`. The
destination is explicit: `claude` or `agents` selects the agent layout in the
current project, and `--global` (only valid with `claude`) targets `~/.claude`
in your home directory. A project install creates the whole path; a global
install requires `~/.claude` to already exist — it is never created, and the
command fails if it is missing. The stub only points back at `trakktor skill
show`, so it never goes stale between releases. An existing `SKILL.md` is left
untouched unless `--force` is given; with `--json` the result is a single object
`{ "path": "…", "status": "written" | "skipped" }`.
