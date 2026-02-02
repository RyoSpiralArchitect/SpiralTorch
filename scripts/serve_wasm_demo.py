#!/usr/bin/env python3
"""
Serve SpiralTorch WASM demos with correct MIME types.

Fixes the common browser error:
  "Unexpected response MIME type. Expected 'application/wasm'"

Example:
  python scripts/serve_wasm_demo.py bindings/st-wasm/examples/mellin-log-grid/dist --port 4174
"""

from __future__ import annotations

import argparse
import mimetypes
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final

DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 4174


def _install_mime_overrides() -> None:
    mimetypes.add_type("application/wasm", ".wasm")
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("application/javascript", ".mjs")


class WasmHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def guess_type(self, path: str) -> str:
        if path.endswith(".wasm"):
            return "application/wasm"
        return super().guess_type(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a directory with WASM-friendly headers.")
    parser.add_argument(
        "directory",
        nargs="?",
        default="bindings/st-wasm/examples/mellin-log-grid/dist",
        help="Directory to serve (default: Mellin log-grid dist).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    _install_mime_overrides()

    root = Path(args.directory).resolve()
    if not root.exists():
        raise SystemExit(f"directory not found: {root}")
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    os.chdir(root)

    server = ThreadingHTTPServer((args.host, args.port), lambda *h_args, **h_kwargs: WasmHandler(*h_args, directory=str(root), **h_kwargs))
    url = f"http://{args.host}:{args.port}/"
    print(f"[SpiralTorch] serving {root}")
    print(f"[SpiralTorch] open: {url}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SpiralTorch] shutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

