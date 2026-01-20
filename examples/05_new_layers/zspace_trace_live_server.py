#!/usr/bin/env -S python3 -s
# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect

"""Live Z-space trace stream (SSE + Canvas viewer).

Run from a source checkout:
`python3 -s examples/05_new_layers/zspace_trace_live_server.py`

If you see `AttributeError: ... install_trace_recorder`, rebuild the Python extension
(e.g. via `maturin build` / `maturin develop`) so the latest Rust bindings are picked up.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

try:
    import spiraltorch as st
except ModuleNotFoundError:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import spiraltorch as st

from spiraltorch.nn import ZSpaceCoherenceSequencer

print("🌀 SpiralTorch Z-space live trace server\n")

model = ZSpaceCoherenceSequencer(dim=768, num_heads=12, curvature=-1.0)
model.install_trace_recorder(capacity=8192, max_vector_len=64, publish_plugin_events=True)

trace_path = Path(tempfile.gettempdir()) / "spiraltorch_zspace_trace_live.jsonl"
server = st.serve_zspace_trace(record_jsonl=str(trace_path), open_browser=True, background=True)
print(f"viewer_url={server.url}")
print(f"record_jsonl={server.record_jsonl}")

try:
    for step in range(16):
        x = st.Tensor.rand(2, 768, seed=10 + step)
        model.forward_with_diagnostics(x)
        time.sleep(0.15)
    print("streaming... (Ctrl+C to stop)")
    while True:
        x = st.Tensor.rand(2, 768, seed=int(time.time()) & 0xFFFF)
        model.forward_with_diagnostics(x)
        time.sleep(0.35)
except KeyboardInterrupt:
    print("\nshutting down...")
    server.close()

