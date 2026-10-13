---
name: docs
description: Keep speech.cpp's README.md, its Japanese README.ja.md and docs/ current, short and plain, for the people who use speech.cpp. Covers which page describes what, when a change must update a page, what a page holds and leaves out, and what to check before committing. Use when a change alters what a user does or sees (a command, an option, an endpoint, a model, a number), when writing or rewriting a page, or when asked to update, fix, shorten or check the documentation (ドキュメントを更新、README を直す、docs に書く、説明を短く). Not for decision records (docs/adr/), code comments, or what only a developer needs, which goes into the code, AGENTS.md or a skill.
---

# The documentation

The documentation is for the people who use speech.cpp: from the command line, over HTTP, as a worker or through the
C API. README.md is the shortest path from install to speech, about 200 lines, and its Documentation table is the
index. `docs/` has one page per topic, and `docs/models/` one page per family. A page that the README does not link
does not exist for a reader.

README.ja.md is README.md in Japanese, section for section, with the same links, tables, commands and numbers; the
pages of `docs/` are English alone. Every change to README.md is made to README.ja.md in the same pull request.

What a developer needs lives elsewhere, where it changes with the work: in the code (each check, converter and dump
script prints its usage), AGENTS.md, and the skills (add-model, release, windows-check, pull-request). The reasons for a
choice are in docs/adr/.

## Which page

| A change to | Updates |
|---|---|
| a subcommand or its flags | docs/cli.md; README's table if the subcommand is new |
| a request option, a voice, a model's limits | the family's page in docs/models/; docs/models.md for voices and languages |
| the catalog (a model, a name, a type) | docs/models.md, the family's page and README's model table |
| `speech serve`, its page or the HTTP API | docs/server.md; README's screenshots when the page looks different, taken with the release skill's script |
| the worker protocol | docs/worker.md |
| `speech.h` | docs/c-api.md |
| the installers, the release archives, building | docs/install.md and README's Install |
| a headline accuracy or speed | the family's page; README's table if it lists the model |

The pull request that changes the behaviour updates its pages. Nothing checks the docs against the code by test, and
none may be added: a test of wording fails on a deliberate rewording and points to no defect. The docs stay current
because they are short, hold few facts that can go stale, and the change's author reads the pages above and edits them.

## What a page holds

- What a user does and needs to know: what a command or model does, how to run it, its options and what they change,
  the limits a user meets, the errors they see, and the headline accuracy and speed.
- The current release alone. No page says what earlier releases did, or that something is new; a page reads as if
  speech.cpp had always been as it is.
- Not how it is implemented, which names of the official code it follows, the rule behind each refusal, how a guard
  works inside, how it was checked step by step, or why it was chosen. Those are the code's, the skills' and the ADRs'.
- No links to docs/adr/: a decision record is rewritten as decisions change, and a page must not depend on it.
- A value the program already prints (`speech info` for options and ranges, `speech models` for the catalog,
  `--help` for flags) is written only where the reader needs it before running anything.

## Writing

- English, plain and concrete: short sentences, common words, commands a reader can copy and run.
- Each page opens with one sentence saying what it is for.
- A small table or a code block beats a paragraph. A paragraph that only introduces the next one is cut.
- A fact may appear in the README and a page both; brevity matters more than one place per fact.
- Numbers carry their conditions: the model, the type and the device.
- No marketing words ("blazing", "seamless", "robust", "powerful") and no filler ("Let's dive in", "In this page we
  will").
- README.ja.md is written in plain Japanese, in full sentences: commands, code, option names and model names stay as
  in README.md, and a term Japanese readers already use in English (GGUF, prompt, Realtime) is left in English. A
  paragraph is one line, because GitHub shows a line break inside a paragraph as a space between Japanese words.

## Before committing

1. Every relative link resolves, anchors included. Check them with a short script you run and do not commit.
2. Tables render: every row has as many cells as its header (`gh api /markdown` renders GitHub's way).
3. README.md stays near 200 lines; if a change pushes it past, move detail to a page.
4. Commands in the README's quick start run as written against the current catalog.
5. README.ja.md has the same sections, links, tables, commands and numbers as README.md.
