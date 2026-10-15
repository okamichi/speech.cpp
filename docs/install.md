# Install

This page installs the `speech` command, updates it and removes it, and builds it from source.

## One line

macOS on Apple silicon and Linux on x86-64:

```sh
curl -fsSL https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.sh | sh
```

Windows x64, in PowerShell:

```powershell
irm https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.ps1 | iex
```

Then open a new terminal and run `speech --help`. Models are not installed: `speech` fetches a model the first time a
command names it ([models.md](models.md)).

## What the installer does

1. It downloads the latest release's archive for your system from GitHub.
2. It checks the archive against the release's `SHA256SUMS`.
3. It unpacks the archive and checks that `speech` starts.
4. It links the command and, when its folder is not on `PATH`, adds it there and says where.

| | macOS and Linux | Windows |
|---|---|---|
| Program | `~/.local/share/speech.cpp/<version>/` | `%LOCALAPPDATA%\Programs\speech.cpp\<version>\` |
| Command | the link `~/.local/bin/speech` | the junction `%LOCALAPPDATA%\Programs\speech.cpp\current`, which points to the program |
| `PATH` | a line in the startup file of the shell in `$SHELL` | the start of the user's `PATH` |

The startup file is `~/.zshrc` for zsh, `~/.bashrc` for bash on Linux and `~/.bash_profile` on macOS,
`~/.config/fish/conf.d/speech.cpp.fish` for fish, and `~/.profile` for any other shell.

On Linux the installer picks the Vulkan build when the Vulkan loader, `libvulkan.so.1`, is installed, and the CPU build
otherwise, and says why. On Windows it needs the Vulkan loader, `vulkan-1.dll`, which the GPU driver of NVIDIA, AMD or
Intel installs; without it the installer stops and says so.

## Options

| `install.sh` | `install.ps1` | Meaning |
|---|---|---|
| `--version X.Y.Z` | `-Version X.Y.Z` | install this release instead of the latest |
| `--no-modify-path` | `-NoModifyPath` | change no startup file and no `PATH`; the installer says what to add |

Options go after `sh -s --`, and in PowerShell to the script as a script block:

```sh
curl -fsSL https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.sh | sh -s -- --no-modify-path
```

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.ps1))) -NoModifyPath
```

## Update

```sh
speech update
```

`speech update` runs the installer when a later release is out, and `speech update --check` only says whether one is.
Running the installer again does the same: it installs the latest release and removes the version it replaces. On Linux
it also switches between the CPU and the Vulkan build when the Vulkan loader has been installed or removed.

On a terminal, `speech` says once a day when a later release is out ([cli.md](cli.md#rules-for-every-subcommand));
`SPEECH_NO_UPDATE_CHECK=1` turns that off.

## Uninstall

On macOS and Linux, remove the program, its link and the models it fetched:

```sh
rm -rf ~/.local/share/speech.cpp ~/.local/bin/speech
rm -rf ~/Library/Caches/speech.cpp     # the models on macOS; ~/.cache/speech.cpp on Linux
```

If the installer said it added `export PATH="$HOME/.local/bin:$PATH"` to a startup file, remove that line too.

On Windows, remove the program and the models, and take `%LOCALAPPDATA%\Programs\speech.cpp\current` out of the user's
`PATH` (Settings, System, About, Advanced system settings, Environment Variables):

```powershell
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Programs\speech.cpp", "$env:LOCALAPPDATA\speech.cpp"
```

## Requirements

- **macOS**: Apple silicon. On an Intel Mac, [build from source](#build-from-source).
- **Windows**: x64 and the Vulkan loader of a GPU driver.
- **Linux**: x86-64 with AVX2, FMA and F16C (Intel Haswell, AMD Excavator or later), and glibc 2.34 or later: Ubuntu
  22.04, Debian 12, Fedora 35, RHEL 9 and the releases after them.
  - The Vulkan build also needs the Vulkan loader (`libvulkan1` on Debian and Ubuntu, `vulkan-loader` on Fedora), without
    which it does not start, and the GPU's Vulkan driver (Mesa for AMD and Intel, NVIDIA's own for NVIDIA). With the
    loader but no GPU driver it runs on the CPU.
  - The CPU build needs neither, for a machine without a GPU.

On the first run with a GPU, the driver compiles the shaders or kernels, which takes seconds (16 s for Irodori-TTS on an
Apple M5) and is kept for the runs after.

## Release archives

Each [release](https://github.com/nyosegawa/speech.cpp/releases) has one archive per platform,
`speech-<version>-<platform>.zip`, and their SHA-256 sums in `SHA256SUMS`. The installer uses them, and a program that
links the library takes them by hand.

| Platform | Runs on |
|---|---|
| `macos-arm64-metal` | macOS on Apple silicon, with Metal |
| `windows-x64-vulkan` | Windows x64, with Vulkan |
| `linux-x64-vulkan` | Linux x86-64, with Vulkan |
| `linux-x64-cpu` | Linux x86-64, on the CPU alone |

| File in the archive | What it is |
|---|---|
| `speech` or `speech.exe` | the command, with the library and ggml linked in |
| `libspeech.3.dylib` and the link `libspeech.dylib`, `libspeech.so.3` and the link `libspeech.so`, or `speech.dll` and its import library `speech.lib` | the shared library, which exports the C API and nothing else; 3 is the C API's major version |
| `speech.h` | the C API ([c-api.md](c-api.md)) |

## Build from source

The build needs CMake 3.20 or later and a C++17 compiler.

```sh
git clone --recurse-submodules https://github.com/nyosegawa/speech.cpp.git
cd speech.cpp
cmake -B build                  # Metal on macOS, the CPU elsewhere
cmake --build build --config Release -j
```

It makes `build/speech` and the shared library beside it. A build from source is tuned to the CPU it is built on.

| Backend | Configure with | Notes |
|---|---|---|
| Metal | `cmake -B build` on macOS | checked and released |
| CPU | `cmake -B build` elsewhere | checked and released for Linux |
| Vulkan | `cmake -B build -DGGML_VULKAN=ON` | checked on NVIDIA and released for Windows and Linux; needs the Vulkan SDK to build |
| CUDA | `cmake -B build -DGGML_CUDA=ON` | builds, but is not checked or released |

On Linux, the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#linux) brings everything the Vulkan build needs in one
archive:

```sh
curl -LO https://sdk.lunarg.com/sdk/download/1.4.357.0/linux/vulkansdk-linux-x86_64-1.4.357.0.tar.xz
tar -xJf vulkansdk-linux-x86_64-1.4.357.0.tar.xz
source 1.4.357.0/setup-env.sh
cmake -B build -DGGML_VULKAN=ON
cmake --build build -j
```
