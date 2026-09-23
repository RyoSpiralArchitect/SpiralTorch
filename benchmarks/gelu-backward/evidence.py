"""Reuse the append-only archive engine without storing raw arrays in Git."""
from functools import partial
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol_gelu as protocol

spec = importlib.util.spec_from_file_location("gelu_archive_engine",
    Path(__file__).resolve().parents[1]/"nerf-single-submit/archive.py")
shared = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = shared
spec.loader.exec_module(shared)
STAGES = ["format","admission","archive-tests","native-clippy","wasm-clippy","tensor-wasm-clippy",
          "backend-tests","tensor-tests","nn-tests","core-tests","native-build","freeze-native","wasm-build","bindgen"]
STAGES += [f"round-{r}-{family}" for r in range(3) for family in ("native","browser","torch")]
PROTOCOL = shared.ArchiveProtocol(protocol.analyze,protocol.validate_summary,STAGES,
    "Remove discarded work from the ordinary Tensor GELU derivative while retaining the full fused contract. "
    "Compare old fused/all reads, the same fused/selected read, and plain/selected read; "
    "separately compare shared derivative and batched observation for the full contract. "
    "Preserve the original loader compile failure and the syntax-only-repaired legacy numerical diagnostic. "
    "Use the same canonical derivative for native/WASM, Tensor and resident execution, "
    "without changing CPU finite guards or redefining residual accumulation. "
    "All slower cases and failed attempts remain visible; prepared timings are not whole-model speedups.")
publish = partial(shared.publish,protocol=PROTOCOL)
verify = partial(shared.verify,protocol=PROTOCOL)

if __name__ == "__main__":
    shared.main(PROTOCOL)
