#!/usr/bin/env -S python3 -S -s
# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2025 Ryo ∴ SpiralArchitect

"""TrainerStep plugin trace + HTML viewer demo.

Run from a source checkout:
`python3 -S -s examples/trainer_trace_visualize_demo.py`
"""

from __future__ import annotations

import importlib
import importlib.util
import random
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    import spiraltorch as st
except ModuleNotFoundError:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    importlib.invalidate_caches()
    sys.modules.pop("spiraltorch", None)
    import spiraltorch as st


def _load_trainer_trace_writer():
    writer = getattr(st, "write_trainer_trace_html", None)
    if callable(writer):
        return writer

    candidate = (
        _REPO_ROOT / "bindings" / "st-py" / "spiraltorch" / "trainer_trace.py"
    )
    if candidate.is_file():
        spec = importlib.util.spec_from_file_location(
            "_spiraltorch_trainer_trace_viewer", candidate
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load trainer trace viewer: {candidate}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        writer = getattr(module, "write_trainer_trace_html", None)
        if callable(writer):
            return writer

    raise AttributeError(
        "spiraltorch.write_trainer_trace_html is unavailable. "
        "Upgrade spiraltorch or use the source checkout helper "
        f"at {candidate}."
    )


write_trainer_trace_html = _load_trainer_trace_writer()

print("SpiralTorch TrainerStep trace + visualization demo\n")

trainer = st.nn.ModuleTrainer(
    backend="cpu",
    curvature=-1.0,
    hyper_learning_rate=0.05,
    fallback_learning_rate=0.01,
)

model = st.nn.Sequential()
model.add(st.nn.Linear("lin1", 4, 8))
model.add(st.nn.Relu())
model.add(st.nn.Linear("lin2", 8, 1))
model.attach_hypergrad(curvature=-1.0, learning_rate=0.05)

loss = st.nn.MeanSquaredError()
schedule = trainer.roundtable(1, 1)

rng = random.Random(0)
dataset = []
for _ in range(48):
    x_data = [rng.uniform(-1.0, 1.0) for _ in range(4)]
    y = sum(x_data) / 4.0
    x = st.Tensor(1, 4, x_data)
    t = st.Tensor(1, 1, [y])
    dataset.append((x, t))

tmp = Path(tempfile.gettempdir())
trace_jsonl = tmp / "spiraltorch_trainer_trace.jsonl"
html_path = tmp / "spiraltorch_trainer_trace.html"

with st.plugin.record(
    trace_jsonl,
    ["TrainerStep", "TrainerPhase", "EpochStart", "EpochEnd"],
    mode="w",
):
    stats = trainer.train_epoch(model, loss, dataset, schedule)

write_trainer_trace_html(trace_jsonl, html_path, title="SpiralTorch Trainer Trace")

print(f"average_loss={stats.average_loss:.6f}")
print(f"\ntrace_jsonl={trace_jsonl}")
print(f"viewer_html={html_path}")
