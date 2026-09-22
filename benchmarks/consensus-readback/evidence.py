"""Reuse the append-only archive engine; raw arrays/binaries remain local."""
from functools import partial
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import consensus_protocol as protocol

spec = importlib.util.spec_from_file_location(
    "consensus_archive_engine", Path(__file__).resolve().parents[1] / "nerf-single-submit/archive.py")
shared = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = shared
spec.loader.exec_module(shared)
STAGES = ["format", "admission", "archive-tests", "archive-compatibility", "native-clippy", "wasm-clippy", "tensor-clippy",
          "tensor-wasm-clippy", "backend-tests", "tensor-tests", "native-build", "freeze-native", "wasm-build", "bindgen"]
STAGES += [f"round-{r}-{family}" for r in range(3) for family in ("native", "browser", "torch")]
PROTOCOL = shared.ArchiveProtocol(protocol.analyze, protocol.validate_summary, STAGES,
    "Preserve the original per-statistic reduction tree and snapshot ownership. Distinguish batched readback "
    "from coupled barriers using all four factors; count=2 is a reduction placebo. Keep every slower/noisy "
    "condition. Public Tensor still blends consensus on CPU; raw GPU consensus is a separate tested boundary. "
    "Initial browser export crashed after all cells started; screen2 uses bounded per-case export. "
    "Earlier failed and incomplete runs remain separate and are never overwritten. "
    "Final accepted-balanced rounds also fix the legacy reversal/rotation position imbalance; "
    "screen2 remains a separate legacy-order diagnostic, not pooled with final data.",
    screening_prefix="screen2", accepted_directory="accepted-balanced")
publish = partial(shared.publish, protocol=PROTOCOL)
verify = partial(shared.verify, protocol=PROTOCOL)

if __name__ == "__main__":
    shared.main(PROTOCOL)
