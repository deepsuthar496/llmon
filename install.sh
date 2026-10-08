#!/usr/bin/env bash
# llmon one-click installer — Linux & macOS.
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/deepsuthar496/llmon/main/install.sh | bash
#   # or from a local checkout:
#   ./install.sh
#
# What it does:
#   1. Installs Rust (rustup) if `cargo` is missing.
#   2. Tries a prebuilt binary from GitHub releases (fast path, ~7 MB).
#   3. Falls back to `cargo build --release` from source (always works).
#   4. Installs the binary as `llmon` to ~/.local/bin (or /usr/local/bin with sudo).
#
# Rebrand note: the installed binary name comes from Cargo `[[bin]] name`
# plus src/branding.rs. To rename the product, change those two and re-run.

set -euo pipefail

APP_BIN="${APP_BIN:-llmon}"
REPO="${LLMON_REPO:-}"   # e.g. "yourname/llmon" — enables prebuilt-binary fast path
VERSION="${LLMON_VERSION:-0.1.0}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

info() { printf '\033[1;32m[llmon]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[llmon]\033[0m %s\n' "$*"; }

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ARCH="x86_64" ;;
  arm64|aarch64) ARCH="aarch64" ;;
  *) warn "unsupported arch $ARCH — source build will be attempted" ;; 
esac
info "detected: $OS/$ARCH"

# 1. Rust toolchain if needed (source-build fallback).
if ! command -v cargo >/dev/null 2>&1; then
  info "cargo not found — installing Rust via rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

mkdir -p "$INSTALL_DIR"
BIN="$INSTALL_DIR/$APP_BIN"

# 2. Prebuilt fast path (only when REPO is set).
if [ -n "$REPO" ]; then
  case "$OS" in
    linux)  TARGET="${ARCH}-unknown-linux-musl" ;;
    darwin) TARGET="${ARCH}-apple-darwin" ;;
    *) TARGET="" ;;
  esac
  if [ -n "${TARGET:-}" ]; then
    URL="https://github.com/${REPO}/releases/download/v${VERSION}/${APP_BIN}-${VERSION}-${TARGET}.tar.gz"
    info "trying prebuilt binary: $URL"
    if curl -fsSL "$URL" -o /tmp/llmon.tgz; then
      tar -xzf /tmp/llmon.tgz -C /tmp
      install -m 755 "/tmp/${APP_BIN}" "$BIN"
      info "installed prebuilt $BIN"
      "$BIN" --help >/dev/null && info "OK: $("$BIN" --version 2>/dev/null || echo installed)"
      exit 0
    else
      warn "no prebuilt binary — falling back to source build"
    fi
  fi
fi

# 3. Source build (works everywhere with Rust).
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$SRC_DIR/Cargo.toml" ]; then
  if [ -n "$REPO" ]; then
    info "cloning source..."
    rm -rf /tmp/llmon-src && git clone --depth 1 "https://github.com/${REPO}" /tmp/llmon-src
    SRC_DIR=/tmp/llmon-src
  else
    echo "error: Cargo.toml not found and LLMON_REPO is unset." >&2
    echo "Run this script from the llmon checkout, or set LLMON_REPO=user/llmon." >&2
    exit 1
  fi
fi
info "building from source (release, ~1-2 min)..."
(cd "$SRC_DIR" && cargo build --release --locked)
install -m 755 "$SRC_DIR/target/release/$APP_BIN" "$BIN"

case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *) warn "add to PATH: export PATH=\"$INSTALL_DIR:\$PATH\"" ;;
esac
info "installed $BIN"
"$BIN" bench --tokens 32
info "done. Start with: $APP_BIN serve   |   $APP_BIN run demo \"hello\""
