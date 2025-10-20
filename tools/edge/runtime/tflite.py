from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List


class TFLiteEmulator:
    """Minimal TensorFlow Lite compatible runtime stub."""

    def __init__(self, artefact_path: Path | str):
        path = Path(artefact_path)
        data = json.loads(path.read_text())
        self.weights: List[float] = data.get("weights", [])
        self.metadata = data

    def run(self, input_vector: Iterable[float]) -> float:
        return sum(w * float(x) for w, x in zip(self.weights, input_vector))
