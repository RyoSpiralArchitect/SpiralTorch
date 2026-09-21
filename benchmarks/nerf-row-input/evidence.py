"""Row-input protocol, shared append-only publication/fixity implementation."""
from functools import partial
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare

# Multiple historical benchmarks have an archive.py; bind this shared engine
# by its actual path rather than whichever dependency prepended sys.path last.
spec = importlib.util.spec_from_file_location(
    "nerf_submit_archive", Path(__file__).resolve().parents[1] / "nerf-single-submit/archive.py"
)
shared = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = shared
spec.loader.exec_module(shared)

STAGES = [
    "format", "admission", "archive-tests", "legacy-protocol", "legacy-archive-tests",
    "legacy-archive-check", "contracts", "contract-clippy", "native-clippy", "wasm-clippy",
    "backend-tests", "upper-tests", "native-build", "freeze-native", "wasm-build", "bindgen",
    "legacy-native", "legacy-browser", "submission-native", "submission-browser",
] + [f"round-{r}-{family}" for r in range(3) for family in ("native", "browser", "torch")]

PROTOCOL = shared.ArchiveProtocol(
    compare.analyze, compare.validate_summary, STAGES,
    "Use eligible affine-row addressing in first-linear forwards; retain the same-submission "
    "forced-packing control and automatic fallback. This removes an allocation/dispatch, "
    "not a submission; no universal speed claim or training-path change.",
)
publish = partial(shared.publish, protocol=PROTOCOL)
verify = partial(shared.verify, protocol=PROTOCOL)

if __name__ == "__main__":
    shared.main(PROTOCOL)
