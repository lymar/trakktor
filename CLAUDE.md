# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trakktor is an experimental Rust utility that runs AI tasks written in JavaScript. It embeds the Boa JavaScript engine (no Node.js required) and provides a host API for JS tasks to interact with OpenAI services, HTTP, file I/O, and interactive terminal previews. Single self-contained binary.

## Build & Run Commands

```bash
# Build
cargo build

# Build release (LTO enabled, optimized)
cargo build --release

# Run a JS task
cargo run --bin dev_run_js -- path/to/script.js
# Requires: export OPENAI_API_KEY=...

# Format code (requires nightly for unstable features in rustfmt.toml)
cargo +nightly fmt

# Check formatting
cargo +nightly fmt -- --check

# Lint
cargo clippy

# Run tests (no tests exist yet)
cargo test
```

## Code Formatting

Uses nightly rustfmt with: max width 80 chars, `group_imports = "StdExternalCrate"`, `imports_granularity = "Crate"`, Unix newlines. See `rustfmt.toml`.

## Architecture

The codebase is a Cargo workspace with one member crate: `trakktor_core`.

### Execution Flow

```
dev_run_js (bin) → TrkConfig::builder() → trk::run_trk()
  → Trk::run() → engine::run(Arc<Trk>)
    → Create Boa context with custom job executor (Queue)
    → Register global JS APIs
    → Load JS file as ES6 module
    → Call exported run() function
    → Await promise resolution via Queue
```

### Key Modules

- **`engine/`** — JavaScript engine runtime. `mod.rs` contains the `Queue` job executor that bridges Boa's microtask system with tokio async. Loads JS as ES modules and calls the exported `run()` function.

- **`engine/api/`** — Rust functions exposed to JavaScript as globals: `httpGetText`, `htmlToMarkdown`, `chat`, `readFile`, `preview`, `textToSpeech`. Async APIs use `NativeFunction::from_async_fn`. Each API is registered in `api/mod.rs`.

- **`trk/`** — Runtime orchestrator. `Trk` struct holds config and a preview mutex (ensures one TUI at a time). `TrkConfig` uses the `bon` builder pattern.

- **`preview/`** — Terminal UI rendering with ratatui/crossterm. Currently supports audio playback via rodio/symphonia with a full TUI player (play/pause, progress, scrollable text).

- **`artifact/`** — `Artifact` wraps binary data (`Arc<[u8]>`) with optional MIME type and filename. Used to pass data between JS APIs (e.g., `readFile` → `preview`, `textToSpeech` → `preview`).

- **`logger.rs`** — Custom `TrkLogger` with a mute guard to suppress output during TUI rendering. JS `console.*` calls route through `js_logger.rs` to this logger.

### Key Patterns

- **Async bridging**: The `Queue` struct implements Boa's `JobExecutor` trait, running async jobs on a `tokio::task::LocalSet`. This is the most complex part of the codebase.
- **Builder pattern**: `TrkConfig` and `Artifact` use the `bon` crate for builder macros.
- **JS↔Rust data flow**: JS APIs receive `JsValue` args, deserialize via `serde_json`, perform async Rust work, and return `JsValue` results. The `args.rs` helper extracts and converts arguments.
- **Single-threaded JS**: All Boa execution is `!Send`, confined to a `LocalSet`. Rust async work (HTTP, OpenAI calls) happens on the tokio runtime and results are bridged back.

## JavaScript API

Scripts are ES modules that export an async `run()` function. Available globals:

| Function | Returns | Notes |
|---|---|---|
| `httpGetText(url)` | `Promise<string>` | |
| `htmlToMarkdown(html)` | `string` | Synchronous |
| `chat(messages, options?)` | `Promise<string>` | `options.json_schema` for structured output |
| `readFile(path)` | `Promise<Artifact>` | Path relative to CWD |
| `textToSpeech(text, options?)` | `Promise<Artifact>` | `options`: model, voice, speed |
| `preview(artifact, text?)` | `Promise<void>` | TUI preview (audio only currently) |

See `examples/` for usage patterns.
