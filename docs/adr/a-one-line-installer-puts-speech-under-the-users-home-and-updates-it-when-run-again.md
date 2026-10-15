# A one-line installer puts speech under the user's home and updates it when run again

## Context

A release has four archives, and on Linux the right one depends on the machine's GPU stack. rustup, uv and NVIDIA's
NeMo-Speech.cpp install with one line that a user pastes into a shell. The Linux Vulkan build links `libvulkan.so.1`: the
dynamic linker refuses it before `main` where the loader is missing. With the loader and no GPU driver it runs: ggml's
`ggml_backend_vk_reg()` catches the failure of `vkCreateInstance()` and registers no Vulkan backend, and an instance that
finds no physical device gives the backend no device, so `speech` runs on the CPU.

## Decision

- **`install.sh` for macOS and Linux and `install.ps1` for Windows**, at the root of the repository and fetched from the
  main branch: `curl -fsSL https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.sh | sh` and
  `irm https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.ps1 | iex`. The executable keeps its name,
  `speech`. The installers install no model; a command fetches one by name.
- **They pick the archive for the system and GPU**: `macos-arm64-metal` on Apple silicon, also from a shell under
  Rosetta; `windows-x64-vulkan` on Windows x64 where `vulkan-1.dll` is installed, which every GPU driver of NVIDIA, AMD
  and Intel installs, and a refusal saying so elsewhere; on Linux x86-64 with glibc 2.34 or later and AVX2, FMA and F16C,
  `linux-x64-vulkan` where the dynamic linker finds `libvulkan.so.1` (its cache or `LD_LIBRARY_PATH`) and
  `linux-x64-cpu` elsewhere, saying how to get the GPU build. An Intel Mac, another architecture or musl is refused with
  a pointer to building from source.
- **They check the archive against the release's `SHA256SUMS`, and run `speech --version` from it, before anything under
  the home folder changes.** The archive is unpacked into `~/.local/share/speech.cpp/<version>/` and `~/.local/bin/speech`
  is linked to it, the link replaced by a rename so that `speech` is never missing; on Windows into
  `%LOCALAPPDATA%\Programs\speech.cpp\<version>\`, with the junction `current` beside it, which needs no administrator,
  unlike a symbolic link. A `speech` there that the installer did not make is not replaced.
- **Running one again updates**: to the latest release, which GitHub's `/releases/latest` names, or to `--version X.Y.Z`
  (`-Version`). `speech update` runs it, with `--no-modify-path` (`-NoModifyPath`), for a `speech` it put in place. The installed version is kept as it is when it is already that one from the archive this system takes
  now, which `install.sh` records beside it (`.archive`), so that a Linux CPU build is replaced by the Vulkan build once
  the loader is installed, as its message tells, and the other way round; Windows has one archive, so a version alone
  tells it. The other versions in the folder are removed once the new one is linked; on Windows a version a running
  `speech.exe` holds is left with a warning.
- **PATH is changed only where it lacks the folder, and the installer says what it changed**: one line,
  `export PATH="$HOME/.local/bin:$PATH"`, in the startup file of the shell in `$SHELL` (`~/.zshrc`; `~/.bashrc` on Linux
  and `~/.bash_profile` on macOS, whose Terminal starts login shells; a file in fish's `conf.d`; `~/.profile`
  otherwise), not added twice; on Windows the folder at the start of the user's PATH in the registry, written as it is
  stored so that entries such as `%USERPROFILE%\bin` stay unexpanded. `--no-modify-path` (`-NoModifyPath`), as rustup
  has it, changes nothing and says what to add. docs/install.md says how to remove everything.

The alternatives were turned down:

- Choosing the Linux archive by running the Vulkan build's `speech`. It downloads 32 MB that may be thrown away, and the
  dynamic linker's own search, which the check reads, says the same.
- Installing into `/usr/local` with `sudo`. A script piped from the network should not ask for an administrator, and an
  update would ask again.
- Package managers (Homebrew, winget, apt). Each is a separate release process; the installer serves every system, and
  a package can come later.
- Copying `speech.exe` into a folder on PATH on Windows. A running `speech.exe` cannot be overwritten, so an update
  would fail while a server or worker runs; a new version's folder and a junction need no file that is in use.

## Consequences

The README's first commands are the installer and a model name. A release's archives keep their names, since the
installers build them from the version and the platform, and a change to the archives' names or contents changes the
installers in the same pull request. CI runs `install.sh` against the published releases on Linux and macOS;
`install.ps1` is checked by hand on Windows.
