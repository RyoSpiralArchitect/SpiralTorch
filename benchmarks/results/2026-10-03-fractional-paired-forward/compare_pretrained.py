"""Bitwise comparison of two isolated pretrained probe processes."""

import hashlib
import json
import math
import struct
from pathlib import Path

import torch

root = Path(__file__).resolve().parent
output = root / "pretrained-parity.json"
assert not output.exists()


def exact(left, right, path="root"):
    assert type(left) is type(right), path
    if isinstance(left, torch.Tensor):
        assert left.shape == right.shape and left.dtype == right.dtype, path
        assert bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()), path
        assert torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                           right.contiguous().reshape(-1).view(torch.uint8)), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            exact(left[key], right[key], path + "/" + str(key))
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            exact(a, b, path + "/" + str(index))
    elif isinstance(left, float):
        assert math.isfinite(left) and math.isfinite(right), path
        assert struct.pack("!d", left) == struct.pack("!d", right), path
    else:
        assert left == right, path


left, right = (torch.load(root / f"pretrained-{name}.pt", weights_only=True, map_location="cpu")
               for name in ("baseline", "paired"))
exact(left, right)
before, after = (json.loads((root / f"pretrained-{name}.json").read_bytes())
                 for name in ("baseline", "paired"))
assert before["native_sha256"] != after["native_sha256"]
assert before["probe_sha256"] == after["probe_sha256"]
exact(before["runs"], after["runs"])
assert len(before["runs"]) == 6
assert all(len(run["records"]) == 2 for run in before["runs"])
report = {"schema": "spiraltorch.fractional_forward_pretrained_parity.v1", "status": "passed",
          "baseline_native_sha256": before["native_sha256"],
          "paired_native_sha256": after["native_sha256"], "probe_sha256": before["probe_sha256"],
          "comparator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "local_tensor_payload_sha256": {name: hashlib.sha256((root / f"pretrained-{name}.pt").read_bytes()).hexdigest()
                                          for name in ("baseline", "paired")},
          "adapters": 6, "updates_per_binary": 12, "total_auxiliary_updates": 24,
          "all_loss_gradient_parameter_adam_bits_equal": True,
          "frozen_base_unchanged": before["frozen_base_unchanged"] and after["frozen_base_unchanged"],
          "original_studies_unchanged": before["original_study_files_unchanged"] and after["original_study_files_unchanged"],
          "heldout_losses_computed": False, "runs": before["runs"],
          "scope": "Separate native binaries, identical copied checkpoints and two training updates per adapter. No new primary training, endpoint scoring, quality or speed claim."}
with output.open("x") as handle:
    json.dump(report, handle, indent=2, allow_nan=False)
    handle.write("\n")
print(json.dumps({"status": "passed", "adapters": 6, "auxiliary_updates": 24}), flush=True)
