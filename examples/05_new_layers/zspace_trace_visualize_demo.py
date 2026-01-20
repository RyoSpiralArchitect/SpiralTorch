#!/usr/bin/env -S python3 -s
# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect

"""Z-space trace + HTML visualization demo.

Run from a source checkout:
`python3 -s examples/05_new_layers/zspace_trace_visualize_demo.py`

If you see `AttributeError: ... install_trace_recorder`, rebuild the Python extension
(e.g. via `maturin build` / `maturin develop`) so the latest Rust bindings are picked up.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

try:
    import spiraltorch as st
except ModuleNotFoundError:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import spiraltorch as st

from spiraltorch.nn import ZSpaceCoherenceSequencer

print("🌀 SpiralTorch Z-space trace + visualization demo\n")

model = ZSpaceCoherenceSequencer(dim=768, num_heads=12, curvature=-1.0)
model.install_trace_recorder(capacity=2048, max_vector_len=64, publish_plugin_events=True)

x = st.Tensor.rand(2, 768, seed=7)

trace_dir = Path(tempfile.gettempdir())
trace_jsonl = trace_dir / "spiraltorch_zspace_trace_plugin.jsonl"
trace_html = trace_dir / "spiraltorch_zspace_trace.html"

with st.plugin.record(trace_jsonl, ["ZSpaceTrace"], mode="w"):
    y, coherence, diagnostics = model.forward_with_diagnostics(x)

html_path = st.write_zspace_trace_html(trace_jsonl, trace_html, title="SpiralTorch Z-Space Trace")

print(f"y_shape={y.shape()}")
print(f"coherence_channels={len(coherence)} label={diagnostics.observation.label}")
print(f"trace_jsonl={trace_jsonl}")
print(f"trace_html={html_path}")
