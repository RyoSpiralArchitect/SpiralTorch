#!/usr/bin/env bash
set -euo pipefail

if command -v julia >/dev/null 2>&1; then
  echo "Julia is already available: $(command -v julia)"
  exit 0
fi

OS_NAME=$(uname -s)
case "$OS_NAME" in
  Linux)
    if command -v apt-get >/dev/null 2>&1; then
      echo "Installing Julia via apt-get (requires sudo)..."
      sudo apt-get update
      sudo apt-get install -y julia
      exit 0
    fi
    ;;
  Darwin)
    if command -v brew >/dev/null 2>&1; then
      echo "Installing Julia via Homebrew..."
      brew install julia
      exit 0
    fi
    ;;
  MINGW*|MSYS*|CYGWIN*)
    echo "Please install Julia manually from https://julialang.org/downloads/ or via winget."
    exit 1
    ;;
  *)
    ;;
endcase

echo "Automatic Julia installation is not supported on this platform." >&2
echo "You can install juliaup instead: curl -fsSL https://install.julialang.org | sh" >&2
exit 1
