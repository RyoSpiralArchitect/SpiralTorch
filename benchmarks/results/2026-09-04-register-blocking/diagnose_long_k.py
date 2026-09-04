import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import spiraltorch as st
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--helper", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
spec = importlib.util.spec_from_file_location("fixture_helpers", args.helper)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
native = Path(st._rs.__file__).resolve()
report = {"schema": "spiraltorch.long_k_diagnostic.v1", "native": str(native),
          "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
          "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "helper_sha256": hashlib.sha256(args.helper.read_bytes()).hexdigest(),
          "shape_m_k_n": [128, 3072, 768], "cases": []}


def errors(actual, reference):
    actual = actual.double()
    delta = (actual - reference).abs()
    allowed = 1e-5 + 1e-4 * reference.abs()
    ratio = delta / allowed
    index = int(ratio.reshape(-1).argmax())
    row, col = divmod(index, reference.shape[1])
    return {"max_abs_error": float(delta.max()), "failed_cells": int((ratio > 1).sum()),
            "worst_gate_ratio": float(ratio.max()), "worst_gate_cell": [row, col],
            "worst_actual": float(actual[row, col]), "worst_reference": float(reference[row, col])}


for seed in (17, 29, 43):
    a, ha = helpers.fixture(128 * 3072, seed)
    b, hb = helpers.fixture(3072 * 768, seed + 1)
    ta, tb = torch.tensor(a).reshape(128, 3072), torch.tensor(b).reshape(3072, 768)
    reference = ta.double() @ tb.double()
    sequential = torch.zeros((128, 768))
    for k in range(3072):
        sequential += ta[:, k:k+1] * tb[k:k+1, :]
    row = {"seed": seed, "input_sha256": [ha, hb],
           "cpu_sequential_f32": errors(sequential, reference),
           "torch_cuda": errors((ta.cuda() @ tb.cuda()).cpu(), reference), "variants": {}}
    for tile in ((8, 8, 16), (16, 16, 16)):
        kernels = ("scalar", "register_2x2") if hasattr(st.WgpuMatmul, "kernel") else (None,)
        for kernel in kernels:
            options = {"tile_mnk": tile}
            if kernel is not None:
                options["kernel"] = kernel
            workspace = st.WgpuMatmul(128, 3072, 768, **options)
            workspace.upload(st.Tensor(128, 3072, a), st.Tensor(3072, 768, b))
            workspace.dispatch()
            actual = torch.tensor(workspace.readback().tolist())
            row["variants"][str(tile) + ":" + str(kernel)] = {
                **errors(actual, reference), "adapter": workspace.adapter_info(),
                "max_abs_vs_cpu_sequential": float((actual - sequential).abs().max()),
                "equal_cpu_sequential": bool(torch.equal(actual, sequential))}
    report["cases"].append(row)
    print(json.dumps(row), flush=True)
with args.output.open("x") as out:
    json.dump(report, out, indent=2, allow_nan=False)
    out.write("\n")
