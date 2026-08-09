# `skill` — generate the agent skill

> Part of [trakktor](../../../README.md); the global flags and the
> output/exit-code contract are described in
> [Output and exit codes](../../../README.md#output-and-exit-codes).

trakktor can describe itself to a coding agent as an Agent Skill. The content is
generated from the live `clap` definition, so it always matches the installed
binary — there is no hand-written reference to drift out of date.

## Show the skill

```sh
trakktor skill show          # narrative guide (JSON { "content": … } by default)
trakktor skill show --full   # plus the full command/flag/value reference
trakktor skill show --text   # the raw Markdown
```

By default the Markdown is wrapped as `{ "content": "…" }`; `--text` prints the
raw Markdown.

## Install the stub

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
untouched unless `--force` is given. By default the result is a single JSON
object `{ "path": "…", "status": "written" | "skipped" }`; `--text` prints one
status line instead — `installed: <path>`, or
`skipped (exists, use --force): <path>`.
