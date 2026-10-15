# speech says on a terminal when a newer release is out, and speech update runs the installer

## Context

The installers put `speech` under the user's home and update it when run again, but nothing tells a user that a release
is out. Releases come every few days, and a user who installed once keeps running the version they installed. Other
programs start `speech` too: ASIST starts the worker, scripts pipe audio and transcripts through it, and CI and the
smoke scripts run it with pipes; ASIST pins each release by SHA-256 and chooses when to move to the next. GitHub
redirects `https://github.com/nyosegawa/speech.cpp/releases/latest` to the latest release's tag, which is what
`install.sh` reads, and that redirect counts against no rate limit.

## Decision

- **`speech update` asks github.com for the latest release, and runs the installer when this `speech` is the one the
  installer put in place**: `~/.local/share/speech.cpp/<version>/speech`, or
  `%LOCALAPPDATA%\Programs\speech.cpp\<version>\speech.exe` on Windows, after every link and junction is resolved. It
  runs the installer as docs/install.md gives it, with `--no-modify-path` (`-NoModifyPath`), through `/bin/sh` and
  System32's PowerShell by their full paths, shows its messages and exits with its status. A `speech` built from source
  or copied elsewhere is left alone, with a failure that says where it is and how to update it. `speech update --check`
  only says whether a later release is out.
- **The subcommands for people say so in one line on stderr, when stderr is a terminal**: `tts`, `asr`, `vad`, `voice`,
  `info`, `devices`, `models`, `pull`, `rm` and `quantize` once they end, so that the line never falls among progress
  lines, and `serve` as it starts. The worker never does, nor `--version`, `--help` or a command line that cannot run.
  `SPEECH_NO_UPDATE_CHECK` set to anything but an empty value turns it off.
- **The check never costs a subcommand time, and never reaches the network where nobody reads the line**: it runs only
  when stderr is a terminal and it is not turned off, at most once a day. It runs the system's curl as a fetch does,
  beside the subcommand, and a subcommand that ends first stops it. The time of every attempt and the latest release
  found are kept in `release.json` in speech.cpp's cache folder, beside the models' folder and not in
  `SPEECH_MODEL_DIR`, so that a machine without the network tries once a day and the notice shows a release the run
  after it is found. A check that fails says nothing and changes no exit status.

The alternatives were turned down:

- Downloading the release and replacing the executable from inside `speech`. The installer already checks the archive
  against `SHA256SUMS`, picks the Linux build by the Vulkan loader and leaves a version a running `speech.exe` holds on
  Windows; two implementations of the same steps would drift apart.
- Printing the notice on stdout. stdout carries audio, transcripts and JSON, which a line would corrupt.
- Checking from the library or the worker. The library never reaches the network, and programs such as ASIST pin a
  release by SHA-256 and choose when to update; a request from every worker they start would be theirs to explain.
- Asking GitHub's REST API for the latest release. Without a token it allows 60 requests an hour from an address, which
  a shared address or CI exhausts; the `/releases/latest` redirect has no such limit.
- Checking on every run. It is a request for every command, and once a day is enough to learn of a release.
- The notice off by default. Most users would never turn it on and never learn of a release; one variable turns it off.

## Consequences

`speech` reaches github.com only on a terminal, once a day, and when asked by `speech update`. A release's tag keeps the
form `vX.Y.Z`, which the check reads, and the installers keep their place in the main branch and their
`--no-modify-path` option, which `speech update` runs. A user who builds from source learns of a release from the
notice and updates by hand.
