# speech draws colors, marks and a spinner only on a terminal, with escapes of its own, and writes the same lines plain elsewhere

## Context

`speech` is used by people at a terminal and by programs: the smoke scripts, CI, speech-bench, ASIST and scripts that
read its stderr or keep `speech serve`'s log in a file. A person waits for steps that print nothing while they run: a
model's load, which takes seconds for a large model or the first run of a build that compiles Metal's kernels (10 s for
Qwen3-ASR 0.6B on an Apple M5, 2026-10-17), and `speech serve`, which loads up to three models before it says anything.
speech.cpp depends on ggml and two vendored headers alone, and a program reading stderr reads it a line at a time.

## Decision

- **speech draws only where stderr is a terminal that erases a line**: a tty, or on Windows a console with its virtual
  terminal processing on, and `TERM` not `dumb`. There a step under way is a line that a spinner (an equalizer's bars)
  and its seconds redraw ten times a second, which a mark replaces when it ends: a check for a step done, a note for
  audio made. A banner begins with a waveform. Elsewhere the same lines go out without color, mark or spinner, a line
  each, and a step under way writes nothing until it ends; their words are the same on both, so that what a script
  reads is what a person reads.
- **Colors are a few of xterm's 256, in mid tones** (mint, pink, lavender, sky, peach, red and gray), which read on a
  dark background and on a light one, such as the default profile of macOS's Terminal. `NO_COLOR` set to anything but
  an empty value turns them off, as no-color.org asks, and leaves the marks and the spinner.
- **A listing on stdout that a person reads, `speech models`, takes the terminal's shape** where stdout is a terminal,
  as `ls` lays out its columns: the models grouped by task, a check for a fetched file, and a model's languages once,
  cut to the width. Through a pipe it keeps one row per file with every column, and `--json` gives every pin.
- **The escapes are speech's own**, a few dozen lines in the tools, with no library and no terminal database.
- **On Windows, `speech` sets the console's output code page to UTF-8** where stdout or stderr is a console, and puts
  back the one it found as it ends, Ctrl-C included, so that the marks, the waveform and the text of a transcript in
  any language are drawn as the UTF-8 they are written in.
- **`speech serve` writes one line a request**, after the time of day: its method, its path, the status it was
  answered with or how it ended (`gone` for a client that went away, `failed`), what it did, and its client where the
  server listens for other machines. The HTTP logger writes the line with the status sent and, for a request a model
  ran, what the run made; a run that ends after its answer has begun, a stream's or one whose client went away, writes
  its request's line itself as it ends.

The alternatives were turned down:

- A library for terminal output (indicators, fmt's colors, ncurses). Each is a dependency of the tools for what a few
  escapes do, and ncurses would take the screen from the log `speech serve` writes below it.
- Colors and marks on every stream. A log file and a script would read escape codes.
- A format of its own for programs off a terminal, such as JSON on stderr. The subcommands that programs read already
  have `--json` on stdout or the worker's protocol, and a second wording would drift from the first.
- The pale colors of the 256 that a dark theme favors. On a light background they cannot be read.
- The sixteen colors a theme defines. Their tones differ from theme to theme, and the bright ones are among the pale.
- Writing a console's text through WriteConsoleW in its UTF-16 instead of changing the code page. Every write of every
  subcommand would have to go through one function.
- Two lines a request in `speech serve`'s log, the HTTP logger's and the run's. The logger writes a stream's line
  before the stream, so the two came in either order and a reader matched them by hand.
- The run's line alone for a request a model runs. The run does not know the status sent, which a `Range` header makes
  206 or 416 after it.

## Consequences

A script that reads speech's stderr reads lines without escapes when stderr is a pipe or a file, whatever `TERM` is;
`script` and other pseudo-terminals get the drawing. `speech serve`'s log lines begin with the time of day.
