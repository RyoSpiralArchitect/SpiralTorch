#!/usr/bin/env bash
set -euo pipefail

if command -v go >/dev/null 2>&1; then
  echo "Go is already available: $(command -v go)"
  exit 0
fi

OS_NAME=$(uname -s)
case "$OS_NAME" in
  Linux)
    if command -v apt-get >/dev/null 2>&1; then
      echo "Installing Go via apt-get (requires sudo)..."
      sudo apt-get update
      sudo apt-get install -y golang
      exit 0
    fi
    ;;
  Darwin)
    if command -v brew >/dev/null 2>&1; then
      echo "Installing Go via Homebrew..."
      brew install go
      exit 0
    fi
    ;;
  MINGW*|MSYS*|CYGWIN*)
    echo "Please install Go manually via https://go.dev/dl/ or winget."
    exit 1
    ;;
  *)
    ;;
endcase

echo "Automatic Go installation is not supported on this platform." >&2
echo "Download from https://go.dev/dl/ and follow the official instructions." >&2
exit 1
