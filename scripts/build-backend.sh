#!/usr/bin/env bash
# Build the llama.cpp inference backend (llama-server + llama-cli) from source.
# llmon uses these when present on PATH for REAL local inference;
# without them it runs in stub mode (API-correct, demo text only).
#
# Usage: ./scripts/build-backend.sh [--cuda] [CMAKE_CUDA_ARCHITECTURES]
#   --cuda : also build CUDA support (requires nvcc + CUDA toolkit).
#   Set CMAKE_CUDA_ARCHITECTURES for your GPU (75=T4, 86=Ampere, 89=Ada,
#   90=Hopper, 100/120=Blackwell); auto-detected when nvcc can query it.
set -euo pipefail

CUDA=OFF
[ "${1:-}" = "--cuda" ] && CUDA=ON

SRC="$(cd "$(dirname "$0")/../knowledge" 2>/dev/null && pwd || echo "")"
# Standalone mode: clone upstream when no sibling checkout with llama.cpp exists.
if [ ! -d "${SRC}/llama.cpp" ]; then
  SRC=/tmp/llmon-backend
  [ -d "$SRC/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC/llama.cpp"
fi

cd "$SRC/llama.cpp"
ARCH_ARGS=()
if [ "$CUDA" = "ON" ]; then
  ARCH="${CMAKE_CUDA_ARCHITECTURES:-}"
  if [ -z "$ARCH" ] && command -v nvidia-smi >/dev/null; then
    # Map GPU name -> architecture number when nvcc can't (best effort).
    case "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)" in
      *T4*) ARCH=75 ;;
      *A10*|*A30*|*A40*|*3090*|*3080*) ARCH=86 ;;
      *A100*) ARCH=80 ;;
      *H100*|*H200*) ARCH=90 ;;
      *4090*|*4080*|*4070*|*L40*) ARCH=89 ;;
    esac
  fi
  [ -n "$ARCH" ] && ARCH_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=$ARCH")
fi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
  -DGGML_CUDA=$CUDA "${ARCH_ARGS[@]}" \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server llama-cli -j"$(nproc)"

mkdir -p "$HOME/.local/bin" "$HOME/.local/lib"
cp -P build/bin/llama-server build/bin/llama-cli "$HOME/.local/bin/"
cp -P build/bin/*.so* "$HOME/.local/lib/" 2>/dev/null || true

export PATH="$HOME/.local/bin:$PATH" LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
llama-cli --version | head -n 1
echo "backend installed to ~/.local/bin (libs in ~/.local/lib)"
