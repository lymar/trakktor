# `feed` — RSS / Atom / JSON Feed

> Part of [trakktor](../../../README.md); the global flags and the
> output/exit-code contract are described in
> [Output and exit codes](../../../README.md#output-and-exit-codes).

`trakktor feed` works with web feeds — RSS, Atom, and JSON Feed — end to end:
find the feeds a page offers, read their entries, and remember which ones were
already seen, so a repeated run returns only what is new.

## Discover feeds on a page

```sh
trakktor feed discover https://example.com
```

Returns the feeds declared on the page (`url`, `type`, `title`). When the page
declares none, the typical paths — `/feed`, `/feed.xml`, `/rss`, `/rss.xml`,
`/atom.xml`, `/feed.json` — are probed with extra requests before giving up.
An empty result is success.

## Read a feed

```sh
trakktor feed read https://example.com/feed.xml             # JSON (default)
trakktor feed read https://example.com/feed.xml --all --fields all --text
```

Accepts a feed URL or a regular page (autodiscovery applies, reading the first
feed found). Each publication carries a stable `uid`; whether it was read
shows as an `is_read` field when `--fields` asks for it (`is_read` or `all`).

- By default only **unread** publications are returned; `--all` includes read
  ones.
- `uid` is each publication's **primary key** — the id you pass to `mark-read` —
  so it is **always** included, independent of `--fields`.
- `--fields <list>` selects the *additional* fields: a comma-separated list of
  `is_read,title,link,published,updated,summary,content,authors`, or the special
  values `minimal` (default, `title,link`) and `all`.
- With `--text`, `uid` is the first column and a `mark-read` hint is printed to
  stderr.

The `uid` is `hex(BLAKE3(feed_key ‖ 0x00 ‖ tag ‖ 0x00 ‖ item_key))` — the
canonicalized feed URL, a tag naming which kind of entry key was available,
and the entry's own id (falling back to its link, then to title + date) — and
is stable across runs for the same feed + entry.

## Mark publications read

```sh
trakktor feed mark-read <uid> [<uid>...]
```

Idempotent; the reply is `{"marked": n, "already_read": n}`. Read state is
stored as plain files under `<work-dir>/feed/` (`./.trakktor` by default —
`--work-dir` or `TRAKKTOR_DIR` move it), sharded by uid; nothing else is
needed (no database).

## Typical agent workflow

```sh
trakktor feed discover https://example.com             # find a feed
trakktor feed read https://example.com/feed.xml        # read unread items (JSON)
# … take each item's uid …
trakktor feed mark-read <uid1> <uid2>                  # mark them handled
```

On the next `read`, marked publications are no longer returned.
