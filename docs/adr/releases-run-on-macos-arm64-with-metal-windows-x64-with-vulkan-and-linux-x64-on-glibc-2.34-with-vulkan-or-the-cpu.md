# Releases run on macOS arm64 with Metal, Windows x64 with Vulkan, and Linux x64 on glibc 2.34 with Vulkan or the CPU

## Context

A release is one archive per platform, `speech-<VERSION>-<platform>.zip`, holding `speech`, `libspeech` and `speech.h`,
which CI builds and checks on every run. Vulkan runs the GPUs of NVIDIA, AMD and Intel, NVIDIA's at the speeds
docs/models/ lists for an RTX 2080, and on Windows every GPU driver of the three installs the Vulkan loader,
`vulkan-1.dll`.

Unlike macOS and Windows, a Linux binary runs only where the C library is at least as new as the one it was linked
against, and the GPU stack differs from machine to machine: some have a Vulkan driver, some have NVIDIA's CUDA, and
servers and containers often have no GPU and no Vulkan loader at all. Measured on Ubuntu 22.04 (glibc 2.35, GCC 11.4) on
2026-10-05: the binaries need glibc symbols up to `GLIBC_2.34` (`__libc_start_main`, `pthread_create`, `dlopen` and the
other functions glibc 2.34 moved into libc), and nothing from 2.35. The Linux Vulkan SDK 1.4.357.0's `glslc` and loader
need glibc 2.34 and GCC 11's libstdc++ (`GLIBCXX_3.4.29`). GitHub retired its Ubuntu 20.04 runners in April 2025;
`ubuntu-22.04` is the oldest it hosts.

## Decision

- **macOS arm64 with Metal** (`macos-arm64-metal`) is the one macOS build. An Intel Mac builds from source.
- **Windows x64 with Vulkan** (`windows-x64-vulkan`) is the one Windows build. It needs `vulkan-1.dll`, and the
  installer refuses a machine without it.
- **Linux x64 has two builds, with Vulkan** (`linux-x64-vulkan`), for a GPU of any vendor whose driver speaks Vulkan,
  **and for the CPU alone** (`linux-x64-cpu`), which needs no Vulkan loader, for machines without a GPU or its driver.
- The x86-64 builds are made for AVX2, FMA and F16C (`-DGGML_NATIVE=OFF`).
- **The Linux jobs run on `ubuntu-22.04`, and the binaries run on glibc 2.34 or later**: Ubuntu 22.04, Debian 12,
  Fedora 35, RHEL 9 and the releases after them. CI reads the newest glibc symbol each file needs and fails when it is
  newer than 2.34.
- **On Linux the C++ runtime is linked into each file** (`-static-libstdc++ -static-libgcc`) and ggml's OpenMP is off,
  so that a file needs no shared library but glibc's and, in the Vulkan build, `libvulkan.so.1`. CI fails a file that
  needs another one. `libspeech.so` carries its own libstdc++ hidden behind `src/api/speech.map`, which is safe because
  no C++ type or exception crosses the C API.
- **On Windows the C runtime is linked into each file** (`CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`) and ggml's OpenMP
  is off, so that a file needs no DLL but Windows' own and `vulkan-1.dll`; CI reads each file's DLLs and fails one
  that needs another. MSVC's OpenMP runtime, `vcomp140.dll`, comes only with the Visual C++ Redistributable, which
  Windows does not ship.
- **No CUDA build is released.**

The alternatives were turned down:

- Keeping ggml's OpenMP on Windows and shipping `vcomp140.dll` beside `speech.exe` and `speech.dll`. OpenMP made a model
  on the CPU 3 to 16% faster on an i9-9900K (reazonspeech-nemo-v2 by beam search 10.6 against 8.9 RTFx, parakeet-tdt
  0.6b-v3 10.9 against 9.8, Qwen3-TTS 0.6B 0.65 against 0.63, 2026-10-14) and changed nothing beyond noise on the GPU,
  where the Windows build is meant to run; it would put a file of Microsoft's in every archive and thread the CPU
  backend on Windows unlike on every other platform.

- Building Linux in a container with an older glibc, such as manylinux_2_28 (glibc 2.28, which would add RHEL 8, Debian
  11 and Ubuntu 20.04). The Vulkan SDK's `glslc` does not run there, so ggml's shaders would need a shader compiler
  built from source in a second stage, and a pinned toolchain of the container's own. Ubuntu 20.04 and Debian 11 are
  past their regular support; RHEL 8 is the one supported system left out, and it can build from source.
- Building on `ubuntu-latest` (24.04, glibc 2.39). The floor would follow whatever 2.39's headers pick and leave out
  Ubuntu 22.04 and Debian 12, the most common systems with 2.34 to 2.38.
- Linking glibc statically. A static executable cannot load the Vulkan loader and the drivers it loads, and
  `libspeech.so` cannot be static.
- One Vulkan build alone on Linux. It does not start where `libvulkan.so.1` is missing, `--device cpu` included, since
  the dynamic linker refuses it before `main`.
- Loading ggml's backends at run time (`GGML_BACKEND_DL`), so that one build finds Vulkan when it is there. A release
  would hold a library per backend beside `speech` and `libspeech`, and every platform would link differently.
- A CUDA build. It would carry cuBLAS, several hundred megabytes, or require the user's CUDA toolkit to match,
  would tie the binary to a range of NVIDIA drivers, and would put the CUDA toolkit and its kernel compilation into CI.
  It stays a build from source (`-DGGML_CUDA=ON`).

## Consequences

A CPU without AVX2 is not supported by any release, and a Mac with an Intel CPU, a system of another architecture and a
Linux system with musl build from source. The installers pick the archive from these rules. When GitHub retires
`ubuntu-22.04`, the Linux jobs move to the next image, and the glibc check shows at once whether the floor moves; the
floor in the workflow, the installer, README.md, docs/install.md, AGENTS.md and this record then change together.
