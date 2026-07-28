#!/usr/bin/env bash
# Build the Windows `gallium` binary with CUDA (GPU offload) enabled.
#
# `make build` already defaults to the `cuda` feature on Windows, but the CUDA
# build has two Windows-specific toolchain constraints that don't fit in the
# Makefile, so this wrapper sets them up:
#
#   * nvcc must support your GPU's compute arch. CUDA 13.x dropped Pascal
#     (sm_61 / GTX 10-series), so those need CUDA 12.x. Select the toolkit with
#     CUDA_VER and the arch with CUDAARCHS.
#   * nvcc invokes vcvars64.bat itself to set up MSVC. The usual (very long)
#     Windows PATH overflows cmd's 8191-char limit ("The input line is too
#     long"), so we build from a SLIM PATH with only the tools the build needs;
#     nvcc then finds cl and sets VS up on its own.
#
# Run from Git Bash. Override via env:
#   CUDA_VER   CUDA toolkit version         (default 12.9)
#   CUDAARCHS  target GPU arch, no dots     (default 61, i.e. sm_61 / GTX 10xx)
#   CL_DIR     MSVC HostX64/x64 bin dir     (default: newest found)
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_VER="${CUDA_VER:-12.9}"
export CUDAARCHS="${CUDAARCHS:-61}"
CUDA_ROOT="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA_VER}"
[ -x "$CUDA_ROOT/bin/nvcc.exe" ] || { echo "nvcc not found: $CUDA_ROOT/bin (set CUDA_VER)"; exit 1; }

# Newest MSVC x64 cl.exe unless CL_DIR is given (glob covers Hostx64 / HostX64).
if [ -z "${CL_DIR:-}" ]; then
  CL_DIR="$(ls -d "/c/Program Files"*"/Microsoft Visual Studio/"*/BuildTools/VC/Tools/MSVC/*/bin/Host*/x64 2>/dev/null | sort -V | tail -1 || true)"
fi
[ -n "$CL_DIR" ] && [ -x "$CL_DIR/cl.exe" ] || { echo "cl.exe not found (set CL_DIR)"; exit 1; }

export PATH="$HOME/.cargo/bin:$CL_DIR:/c/Program Files/CMake/bin:/c/ProgramData/chocolatey/bin:$CUDA_ROOT/bin:/usr/bin:/bin:/c/Windows/System32:/c/Windows"
export CUDA_PATH="$(cygpath -w "$CUDA_ROOT")"
export CUDACXX="$(cygpath -w "$CUDA_ROOT/bin/nvcc.exe")"
export CUDAFLAGS="${CUDAFLAGS:--allow-unsupported-compiler}"   # newer cl than the toolkit officially lists
export GALLIUM_CUDA_WRAPPER=1   # tell the Makefile to skip its bare-`make build` heads-up

echo "cl:   $CL_DIR"
echo "cuda: $CUDA_ROOT  (sm_$CUDAARCHS)"

exec make build CARGO_FEATURES=cuda
