"""Use the existing append-only evidence engine with a softmax-only protocol."""
from functools import partial
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import softmax_protocol as protocol

spec = importlib.util.spec_from_file_location(
    "softmax_archive_engine", Path(__file__).resolve().parents[1] / "nerf-single-submit/archive.py")
shared = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = shared
spec.loader.exec_module(shared)
STAGES = ["format", "admission", "archive-tests", "native-clippy", "wasm-clippy",
          "backend-tests", "tensor-tests", "native-build", "freeze-native", "wasm-build", "bindgen"]
STAGES += [f"round-{r}-{family}" for r in range(3) for family in ("native", "browser", "torch")]
PROTOCOL = shared.ArchiveProtocol(
    protocol.analyze, protocol.validate_summary, STAGES,
    "Repair finite-domain normalization and row-alias indexing; retain required reduction barriers. "
    "Embedded construction makes the canonical ABI filesystem-independent. Removing two consecutive "
    "barriers does not justify a portable speedup claim; retain every slower/noisy condition.")
publish = partial(shared.publish, protocol=PROTOCOL)
verify = partial(shared.verify, protocol=PROTOCOL)

if __name__ == "__main__":
    shared.main(PROTOCOL)
