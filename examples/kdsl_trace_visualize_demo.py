#!/usr/bin/env -S python3 -s
# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect

"""SpiralK (KDSL) trace + HTML viewer demo.

Run from a source checkout:
`python3 -s examples/kdsl_trace_visualize_demo.py`
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

try:
    import spiraltorch as st
except ModuleNotFoundError:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import spiraltorch as st

print("🌀 SpiralTorch SpiralK (KDSL) trace + visualization demo\n")

plan = st.plan_topk(rows=1024, cols=16384, k=32, backend="cpu")
ctx = plan.spiralk_context()

script = r"""
let base = r / 4;
wg: base;
soft(wg, base, 0.5, true);
soft(radix, base, 1.0, false);

for i in 0..4 {
    tile_cols: i + 1;
}
"""

out, trace = ctx.eval_with_trace(script, max_events=256)
print("out:", out)

tmp = Path(tempfile.gettempdir())
jsonl_path = tmp / "spiraltorch_kdsl_trace.jsonl"
html_path = tmp / "spiraltorch_kdsl_trace.html"

st.write_kdsl_trace_jsonl(trace, jsonl_path)
st.write_kdsl_trace_html(trace, html_path, title="SpiralTorch SpiralK Trace")

print(f"\ntrace_jsonl={jsonl_path}")
print(f"viewer_html={html_path}")

