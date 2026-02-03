# Trakktor

Trakktor is an experimental Rust utility for running AI tasks. The tasks are
written in JavaScript and executed by a single binary.

Instead of relying on Node.js, Trakktor embeds the
[Boa](https://github.com/boa-dev/boa) JavaScript engine and provides a small
host API (HTTP, HTML→Markdown, AI chat, file IO, preview, Text-to-Speech) to your JS scripts.

The goal is a single, self-contained executable that can run AI automations
without external language runtimes.

## Status

Early development.

It works, but **please don’t use it for anything important yet**. The main
purpose right now is to explore the approach and iterate on the runtime API.

## Key ideas

- **Rust binary, JS tasks**: tasks are regular `.js` files.
- **Embedded JS engine**: runs on Boa (no Node.js).
- **Interactive previews in the terminal (TUI)**: tasks can emit artifacts via `preview`, and Trakktor can render them in an interactive terminal UI (currently audio), powered by the fantastic [Ratatui](https://ratatui.rs/).

	![TUI preview](docs/images/tts.gif)
- **Minimal (for now) host API** exposed to JS:
	- `httpGetText`
	- `htmlToMarkdown`
	- `chat`
	- `readFile`
	- `preview`
	- `textToSpeech`

	(6 functions today; more will be added as the project evolves).

## One executable, no language runtimes

Trakktor is designed to run as **a single Rust binary**:

- No Node.js / Python (no external language runtimes).
- JS execution is embedded via Boa.

In other words: you run one executable, and it runs your JS task.

## Quickstart

To see what exists today, you need Rust, then:

```bash
git clone https://github.com/lymar/trakktor
cd trakktor

export OPENAI_API_KEY=...

cargo run --bin dev_run_js -- examples/text_to_speech.js
```

## Included examples

- `examples/on_this_day.js`: fetches Wikipedia, converts HTML→Markdown, asks the
	model to extract the “On this day” section into JSON.
- `examples/structured_output.js`: minimal example of requesting JSON output.
- `examples/text_to_speech.js`: generates audio via `textToSpeech`, then opens an interactive TUI `preview`.
- `examples/audio_preview.js`: reads a local `audio.mp3` via `readFile`, then opens an interactive TUI `preview`.

## License

MIT
