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

## Install the stub

```sh
trakktor skill install claude            # ./.claude/skills/trakktor/SKILL.md
trakktor skill install agents            # ./.agents/skills/trakktor/SKILL.md (preliminary)
trakktor skill install claude --global   # ~/.claude/skills/trakktor/SKILL.md
trakktor skill install claude --force    # overwrite an existing stub
```

Writes a thin discovery stub to `<dir>/skills/trakktor/SKILL.md`, where `<dir>`
is `.claude` or `.agents` in the project, or `~/.claude` with `--global`. The
destination is explicit: `claude` or `agents` selects the agent layout in the
current project, and `--global` (only valid with `claude`) targets `~/.claude`
in your home directory. `claude` is the confirmed layout; `agents` is
preliminary — the `.agents` skills format is not yet verified. A project
install is relative to the current working directory (run it from the project
root) and creates the whole path; a global install requires `~/.claude` to
already exist — it is never created, and the command fails if it is missing,
so create it first (`mkdir -p ~/.claude`). An existing `SKILL.md` is left
untouched unless `--force` is given.

## The stub

The installed `SKILL.md` is a few lines of frontmatter and a pointer back at
the binary (the full trigger `description` — by which an agent decides whether
to load the skill — is elided here):

```markdown
---
name: trakktor
description: "Use trakktor, a predictable, automation-friendly CLI toolbox
  for coding agents. …"
allowed-tools: Bash(trakktor:*)
---

# trakktor

This file is a discovery stub, not the usage guide. It stays valid across
`trakktor` releases because it only points back at the binary.

Before running any `trakktor` command, load the current guide from the binary
itself:

- `trakktor skill show` — what `trakktor` does, when to use it, and the
  typical workflows.
- `trakktor skill show --full` — the above plus the complete reference of
  every command, flag, allowed value, and default, generated from this exact
  version.

The binary is the source of truth for its own interface; run those first.
```

The `allowed-tools: Bash(trakktor:*)` line grants the agent permission to run
`trakktor` commands while the skill is active, without a per-command prompt.

## Output and errors

`skill show` prints a single JSON object, `{ "content": "<markdown>" }`;
`--text` prints the raw Markdown instead. `skill install` prints
`{ "path": "…", "status": "written" | "skipped" }`; with `--text`, one status
line — `installed: <path>`, or `skipped (exists, use --force): <path>`.

Errors, all exit 1: `agent_dir_missing` (a global install found no `~/.claude`
— create it first), `no_home_dir` (the home directory cannot be determined),
`io_error`. `agents --global` is a usage error like any other invalid flag
combination: reported in the CLI's own style, exit code 2.

## Typical agent workflow

Install the stub once, per project or globally. The agent discovers `SKILL.md`,
and its `description` tells the agent when to load the skill. The loaded stub
points back at the binary: `trakktor skill show` for the narrative guide,
`trakktor skill show --full` for the complete reference — both generated on
the spot, so what the agent reads always matches the installed binary.
