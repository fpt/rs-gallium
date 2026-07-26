# Development Notes

## Building on Windows

The tricky part of a Windows build is the **`local` feature** (the in-process
llama.cpp backend, on by default): it compiles llama.cpp from source through
cmake, and that only works cleanly against the **MSVC** toolchain. The pure-Rust
`gallium` (native candle) backend has none of these problems.

`make build` handles all of this automatically — the `Makefile` has an
`ifeq ($(OS),Windows_NT)` block that sets everything below. This section explains
*what* it sets and *why*, so the setup is debuggable when something drifts.

### Prerequisites

| Tool | Notes |
|------|-------|
| **rustup, MSVC toolchain** | `x86_64-pc-windows-msvc` (rustup's Windows default). Verify with `rustc -vV` → `host: x86_64-pc-windows-msvc`. |
| **Visual Studio Build Tools** | Provides `cl.exe` and the Windows SDK. cmake/`cc` find it via the registry — no "Developer Prompt" needed. |
| **CMake** | 3.15+ (4.x fine). On PATH. |
| **Ninja** | On PATH. `choco install ninja` or `winget install Ninja-build.Ninja`. Required — see below. |
| **Git Bash** | The build is driven from Git Bash (`make`, POSIX shell). |

### What the Windows build needs (and why)

Three settings are required. All three are no-ops on macOS/Linux, and the
`Makefile` applies them for you; they're listed here for when you build by hand
or need to debug.

1. **The MSVC `cargo`, not a stray GNU one.**
   If a second Rust — e.g. a Chocolatey-installed `x86_64-pc-windows-gnu` Rust —
   sits earlier on `PATH`, a bare `cargo` builds for the GNU target and then
   can't link the MSVC-ABI `.lib` files llama.cpp produces. The `Makefile` calls
   `$(HOME)/.cargo/bin/cargo` (rustup's proxy, which honors the MSVC default
   toolchain). Check which one you're getting:

   ```bash
   which -a cargo          # is a non-rustup cargo first?
   cargo -vV | grep host   # want: x86_64-pc-windows-msvc
   ```

2. **`CMAKE_GENERATOR=Ninja`.**
   Under Git Bash, cmake otherwise picks the **MSYS Makefiles** generator, and
   MSYS path-conversion mangles MSVC-style linker flags (e.g. `/pdb:...` becomes
   a garbage path). Ninja passes arguments through verbatim, so it must be
   installed and selected.

3. **`CFLAGS=-MD CXXFLAGS=-MD` (dynamic CRT everywhere).**
   `esaxx-rs` (a transitive C++ dependency of `tokenizers`) hardcodes the
   *static* CRT (`/MT`), while Rust `std` and llama.cpp use the *dynamic* CRT
   (`/MD`). Mixing them fails the final link with `LNK2038`. `cc-rs` appends
   `CFLAGS`/`CXXFLAGS` after its own `-MT`, and `cl` honors the last `/M` flag,
   so `-MD` wins and everything ends up `/MD`.

   > Going `/MT` everywhere is **not** an option: cmake-rs 0.1.58 can't force
   > llama.cpp's cmake onto the static CRT. It writes `/MT` into
   > `CMAKE_CXX_FLAGS` but never sets `CMAKE_MSVC_RUNTIME_LIBRARY`, so under
   > CMake policy CMP0091 (NEW) the default `/MD` is appended afterward and wins.

### Building by hand

If you're not going through `make`, the equivalent is:

```bash
export PATH="$HOME/.cargo/bin:$PATH"     # MSVC cargo first
export CMAKE_GENERATOR=Ninja
export CFLAGS=-MD CXXFLAGS=-MD
cargo build --release
```

### Skipping the llama.cpp / cmake build

If you only need the native candle backend, drop the `local` feature. That
removes the llama.cpp cmake build entirely — no cmake and no Ninja, so
`CMAKE_GENERATOR` is irrelevant:

```bash
cargo build --release --no-default-features --features gallium
```

This is **not** a pure-Rust build, though: the `gallium` feature still pulls in
`tokenizers`, whose `esaxx-rs` dependency compiles a C++ source file. So you
still need a C++ compiler (`cl`), and still need `CFLAGS=-MD CXXFLAGS=-MD` —
esaxx-rs's `/MT` vs Rust std's `/MD` triggers the same `LNK2038` on its own,
even without llama.cpp in the link.

### Troubleshooting

| Symptom (in the build output) | Cause | Fix |
|---|---|---|
| `lld-link: could not open 'C:\Program Files\Git\pdb;...'` | MSYS Makefiles generator mangled `/pdb:` | `CMAKE_GENERATOR=Ninja` |
| `make: Makefile: No such file` after "CMake project was already configured" | Stale build dir configured with a different generator | `rm -rf target/release/build/llama-cpp-sys-2-*` and rebuild |
| `assert_ne!(llama_libs.len(), 0)` panic in `llama-cpp-sys-2` build.rs, and stdout shows `x86_64-pc-windows-gnu` | Built with a GNU cargo; can't find the MSVC `.lib` files | Use the MSVC `cargo` (see #1 above) |
| clang++ errors in `esaxx-rs` like *"deduced return types are a C++14 extension"* against MSVC STL headers | `CXX=clang++` compiling modern MSVC STL under `-std=c++11` | Don't force clang; let native `cl.exe` compile (unset `CC`/`CXX`) |
| `LNK2038: mismatch for 'RuntimeLibrary': MT_StaticRelease vs MD_DynamicRelease` | esaxx `/MT` vs llama/std `/MD` | `CFLAGS=-MD CXXFLAGS=-MD` |
| `LNK2019: unresolved external symbol __imp_*` (isprint, fopen, expm1f, …) | llama built `/MD` but Rust linked `/MT` | Don't set `+crt-static`/`LLAMA_STATIC_CRT`; go `/MD` everywhere |

## Building on macOS

There is nothing to configure. `cargo build --release` works as-is: cmake and a
C++ compiler come with the Xcode command line tools, and llama.cpp's Metal
backend needs no flags because it is a default feature for macOS targets (see
the `[target.'cfg(target_os = "macos")'.dependencies]` block in
`crates/gallium-agent/Cargo.toml`). This is the platform the project is
developed on, so it is also the least surprising one.

### Prerequisites

| Tool | Notes |
|------|-------|
| **rustup, stable toolchain** | The default `aarch64-apple-darwin` (or `x86_64-apple-darwin` on Intel). |
| **Xcode command line tools** | `xcode-select --install`. Provides `clang++`, and the `metal` compiler llama.cpp's Metal backend builds its shaders with. |
| **CMake** | 3.15+, for the llama.cpp build. `brew install cmake`. |

### Metal, and when to turn it down

Metal offload is automatic. A model whose weights exceed the GPU's working set
fails to decode (`llama_decode` returns `-3`) rather than falling back, so cap
the offload when that happens:

```bash
GALLIUM_GPU_LAYERS=0 gallium --config configs/gemma4-26b.toml   # CPU only
```

`recommendedMaxWorkingSetSize` in the startup log is the number to compare a
GGUF's size against.

### Skipping the llama.cpp / cmake build

Same escape hatch as on Windows — drop the `local` feature to build only the
native candle backend, which removes cmake from the picture entirely:

```bash
cargo build --release --no-default-features --features gallium
```

### The release artifact

`.github/workflows/build-macos.yml` builds `gallium` on Apple Silicon and
uploads a tarball of the binary plus `configs/`. It runs on demand and on `v*`
tags, mirroring the Windows job.

Two things about a downloaded artifact:

- It is a **tarball inside** the artifact zip, because `upload-artifact` re-zips
  its input and drops the executable bit doing so. Tar carries the mode through.
- It is **unsigned**, so Gatekeeper quarantines it. Clear that once with
  `xattr -d com.apple.quarantine gallium`, or build locally.
