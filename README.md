# Trakktor

Trakktor is an experimental Rust utility for running AI tasks. The tasks are
written in JavaScript and executed by a single binary.

Instead of relying on Node.js, Trakktor embeds the
[Boa](https://github.com/boa-dev/boa) JavaScript engine and provides a small
host API (HTTP, HTML→Markdown, AI chat) to your JS scripts.

The goal is a single, self-contained executable that can run AI automations
without external language runtimes.

## Status

Early development.

It works, but **please don’t use it for anything important yet**. The main
purpose right now is to explore the approach and iterate on the runtime API.

## Key ideas

- **Rust binary, JS tasks**: tasks are regular `.js` files.
- **Embedded JS engine**: runs on Boa (no Node.js).
- **Minimal (for now) host API** exposed to JS: `chat`, `httpGetText`,
	`htmlToMarkdown` (3 functions today; more will be added as the project
	evolves).

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

cargo run -p trakktor_core --bin dev_run_js -- examples/on_this_day.js
```

## Included examples

- `examples/on_this_day.js`: fetches Wikipedia, converts HTML→Markdown, asks the
	model to extract the “On this day” section into JSON.
- `examples/structured_output.js`: minimal example of requesting JSON output.

## License

MIT
