#!/usr/bin/env bash
# Vendored build of SuiteSparse:GraphBLAS v9.4.5 (Apache-2.0) + LAGraph v1.2.1
# (BSD) for the rustyred-thg-graphblas crate, following the FalkorDB vendoring
# pattern: a script builds + installs the native libraries, build.rs links them.
#
# Usage: graphblas.sh <install_prefix>
#   Builds compact (-DGRAPHBLAS_COMPACT=1) + JIT, OpenMP via Homebrew libomp,
#   and installs lib/ + include/suitesparse/ under <install_prefix>.
#
# Source selection:
#   - $RUSTYRED_GRAPHBLAS_VENDOR/{GraphBLAS,LAGraph} if set (offline / pinned), else
#   - shallow clone of the pinned tags from GitHub.
#
# IMPORTANT: <install_prefix> must contain NO spaces. LAGraph's CMake passes its
# source include path unquoted, so a space (e.g. an external volume named
# "SSD Samsung") splits the clang argument and the build fails. build.rs targets
# a space-free per-user cache for exactly this reason.
set -euo pipefail

PREFIX="${1:?usage: graphblas.sh <install_prefix>}"
case "$PREFIX" in
  *" "*) echo "graphblas.sh: refusing to build into a path with spaces: $PREFIX" >&2; exit 2;;
esac

GB_TAG="v9.4.5"
LG_TAG="v1.2.1"
WORK="$(dirname "$PREFIX")/src"
NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

OMP_ARGS=()
if [[ "$(uname -s)" == "Darwin" ]]; then
  OMP_PREFIX="${RUSTYRED_LIBOMP_PREFIX:-/opt/homebrew/opt/libomp}"
  OMP_FLAGS="-Xclang -fopenmp -I${OMP_PREFIX}/include"
  OMP_LIB="${OMP_PREFIX}/lib/libomp.dylib"
  OMP_ARGS=(
    -DOpenMP_C_FLAGS="$OMP_FLAGS"
    -DOpenMP_C_LIB_NAMES=omp
    -DOpenMP_omp_LIBRARY="$OMP_LIB"
  )
fi

# macOS: bake the SDK sysroot into the build so GraphBLAS captures a valid
# -isysroot into its runtime JIT compiler flags. Without it, compact-mode JIT
# kernels (e.g. min-plus FP64) fail to find system headers (GxB_JIT_ERROR).
SDK="$(xcrun --show-sdk-path 2>/dev/null || true)"
SYSROOT_ARGS=()
[ -n "$SDK" ] && SYSROOT_ARGS=(-DCMAKE_OSX_SYSROOT="$SDK")

mkdir -p "$WORK"
if [ -n "${RUSTYRED_GRAPHBLAS_VENDOR:-}" ] && [ -d "${RUSTYRED_GRAPHBLAS_VENDOR}/GraphBLAS" ]; then
  echo "graphblas.sh: using vendored source ${RUSTYRED_GRAPHBLAS_VENDOR}"
  rsync -a --delete --exclude build --exclude .git "${RUSTYRED_GRAPHBLAS_VENDOR}/GraphBLAS/" "$WORK/GraphBLAS/"
  rsync -a --delete --exclude build --exclude .git "${RUSTYRED_GRAPHBLAS_VENDOR}/LAGraph/"   "$WORK/LAGraph/"
else
  echo "graphblas.sh: cloning ${GB_TAG} / ${LG_TAG}"
  [ -d "$WORK/GraphBLAS/.git" ] || { rm -rf "$WORK/GraphBLAS"; git clone --depth 1 --branch "$GB_TAG" https://github.com/DrTimothyAldenDavis/GraphBLAS.git "$WORK/GraphBLAS"; }
  [ -d "$WORK/LAGraph/.git" ]   || { rm -rf "$WORK/LAGraph";   git clone --depth 1 --branch "$LG_TAG" https://github.com/GraphBLAS/LAGraph.git "$WORK/LAGraph"; }
fi

echo "graphblas.sh: building GraphBLAS (compact, JIT) -> $PREFIX"
cmake -S "$WORK/GraphBLAS" -B "$WORK/GraphBLAS/build" -DCMAKE_BUILD_TYPE=Release -DGRAPHBLAS_COMPACT=1 \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" "${SYSROOT_ARGS[@]}" \
  "${OMP_ARGS[@]}"
cmake --build "$WORK/GraphBLAS/build" --config Release -j"$NCPU"
cmake --install "$WORK/GraphBLAS/build"

echo "graphblas.sh: building LAGraph (shared) -> $PREFIX"
cmake -S "$WORK/LAGraph" -B "$WORK/LAGraph/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC_LIBS=OFF \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" "${SYSROOT_ARGS[@]}" \
  "${OMP_ARGS[@]}"
cmake --build "$WORK/LAGraph/build" --config Release -j"$NCPU"
cmake --install "$WORK/LAGraph/build"

echo "graphblas.sh: installed to $PREFIX"
