#!/usr/bin/env bash
# Source this file:
#   source ./prepare-build-021.sh

CUDA_HOME="/usr/local/cuda-13.0"
GCC_BIN="/tmp/gcc15-bin"

echo "=== Preparing vLLM 0.2.1 build environment ==="

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "ERROR: CUDA 13.0 nvcc not found: $CUDA_HOME/bin/nvcc"
    return 1
fi

if [[ ! -x "$GCC_BIN/gcc" || ! -x "$GCC_BIN/g++" ]]; then
    echo "ERROR: GCC 15 wrappers not found in: $GCC_BIN"
    return 1
fi

export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
export PATH="$GCC_BIN:$CUDA_HOME/bin:$PATH"

export CC="$GCC_BIN/gcc"
export CXX="$GCC_BIN/g++"
export CUDAHOSTCXX="$GCC_BIN/g++"

export CMAKE_C_COMPILER="$GCC_BIN/gcc"
export CMAKE_CXX_COMPILER="$GCC_BIN/g++"

echo
echo "=== Toolchain ready ==="
echo "CUDA_HOME: $CUDA_HOME"
echo "CC:        $CC"
echo "CXX:       $CXX"

echo
echo "--- gcc ---"
command -v gcc
gcc --version | head -n1

echo
echo "--- g++ ---"
command -v g++
g++ --version | head -n1

echo
echo "--- nvcc ---"
command -v nvcc
nvcc --version | grep -E "release|Cuda compilation tools" || true

echo
echo "Environment is active in this shell only."