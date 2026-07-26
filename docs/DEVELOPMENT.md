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
| **CUDA Toolkit** (optional) | For the default GPU build. Must support your GPU's arch — CUDA 13.x dropped Pascal, so GTX 10-series needs CUDA 12.x. See [CUDA (GPU offload)](#cuda-gpu-offload). |

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

### CUDA (GPU offload)

On Windows `make build` defaults to the `cuda` feature, so the llama.cpp backend
offloads to the GPU (the runtime asks for all layers by default; cap it for a
small card with `GALLIUM_GPU_LAYERS=<n>`). Build CPU-only with
`make build CARGO_FEATURES=`.

The CUDA build has two extra constraints, both handled by
[`scripts/build-windows-cuda.sh`](../scripts/build-windows-cuda.sh) — just run it:

```bash
bash scripts/build-windows-cuda.sh                 # defaults: CUDA 12.9, sm_61
CUDA_VER=12.6 CUDAARCHS=75 bash scripts/build-windows-cuda.sh
```

Why the wrapper is needed:

1. **nvcc must support your GPU's arch.** CUDA **13.x dropped Pascal** (sm_61,
   the GTX 10-series) — its nvcc only targets sm_75+. Pascal cards need a **CUDA
   12.x** toolkit. Check with `nvcc --list-gpu-arch`. The script points
   `CUDACXX`/`CUDA_PATH` at the chosen toolkit and sets `CUDAARCHS`.
2. **nvcc calls `vcvars64.bat` itself**, and the usual (huge) Windows PATH
   overflows cmd's 8191-char limit → `nvcc fatal: ... The input line is too
   long`. The script builds from a **slim PATH** (cl dir, cmake, ninja, nvcc,
   rustup cargo, MSYS utils) so vcvars has room; nvcc then finds cl and sets up
   VS on its own. It also passes `-allow-unsupported-compiler` since a current
   `cl` is usually newer than the 12.x toolkit officially lists.

At runtime the binary needs `cudart64_12.dll` / `cublas64_12.dll` (from the CUDA
`bin` dir, normally on PATH) and `nvcuda.dll` (from the NVIDIA driver). Verify a
build is CUDA-enabled with `dumpbin /dependents target/release/gallium.exe` (it
should list those DLLs), or run it against a GGUF and look for
`ggml_cuda_init: found N CUDA devices` on stderr.

### Troubleshooting

| Symptom (in the build output) | Cause | Fix |
|---|---|---|
| `lld-link: could not open 'C:\Program Files\Git\pdb;...'` | MSYS Makefiles generator mangled `/pdb:` | `CMAKE_GENERATOR=Ninja` |
| `nvcc fatal: Cannot find compiler 'cl.exe' in PATH` | MSVC not visible to nvcc | Put the MSVC `HostX64/x64` dir on PATH (the CUDA script does this) |
| `nvcc fatal: ... The input line is too long` (vcvars) | PATH too long for nvcc's vcvars call | Build from a slim PATH — use `scripts/build-windows-cuda.sh` |
| `nvcc fatal: Unsupported gpu architecture 'compute_61'` | CUDA 13.x can't target Pascal | Use a CUDA 12.x toolkit (`CUDA_VER=12.9`) |
| `make: Makefile: No such file` after "CMake project was already configured" | Stale build dir configured with a different generator | `rm -rf target/release/build/llama-cpp-sys-2-*` and rebuild |
| `assert_ne!(llama_libs.len(), 0)` panic in `llama-cpp-sys-2` build.rs, and stdout shows `x86_64-pc-windows-gnu` | Built with a GNU cargo; can't find the MSVC `.lib` files | Use the MSVC `cargo` (see #1 above) |
| clang++ errors in `esaxx-rs` like *"deduced return types are a C++14 extension"* against MSVC STL headers | `CXX=clang++` compiling modern MSVC STL under `-std=c++11` | Don't force clang; let native `cl.exe` compile (unset `CC`/`CXX`) |
| `LNK2038: mismatch for 'RuntimeLibrary': MT_StaticRelease vs MD_DynamicRelease` | esaxx `/MT` vs llama/std `/MD` | `CFLAGS=-MD CXXFLAGS=-MD` |
| `LNK2019: unresolved external symbol __imp_*` (isprint, fopen, expm1f, …) | llama built `/MD` but Rust linked `/MT` | Don't set `+crt-static`/`LLAMA_STATIC_CRT`; go `/MD` everywhere |
