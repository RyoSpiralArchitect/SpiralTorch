"""Serial-use throughput check, not a concurrent-race reproducer."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def invoke_and_discard(function):
    # Do not let the timing helper retain an output alias into the next call.
    function()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--native-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[key] = "1"
    import spiraltorch as st
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location("helpers", args.helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    native = Path(st._rs.__file__).resolve()
    if not native.is_relative_to(args.native_prefix.resolve()):
        raise RuntimeError("native extension outside the requested private environment")
    report = {"schema": "spiraltorch.tensor_out_borrow_bench.v1", "native": str(native),
              "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "helper_sha256": hashlib.sha256(args.helper.read_bytes()).hexdigest(),
              "torch": torch.__version__, "device": "cpu", "dtype": "float32", "threads": 1,
              "boundary": "one serial host Tensor out= call; preallocated outputs and prepacked RHS; no training or race evidence",
              "comparison": "paired variants within a run; baseline/candidate wheel runs are unpaired diagnostics",
              "cases": []}
    with args.output.open("x") as stream:
        try:
            for size in (128, 256):
                for seed in (17, 29, 43):
                    av, ah = helper.fixture(size * size, seed)
                    bv, bh = helper.fixture(size * size, seed + 1)
                    case = {"size": size, "seed": seed, "input_sha256": [ah, bh], "correctness": {}}
                    report["cases"].append(case)
                    a, b = st.Tensor(size, size, av), st.Tensor(size, size, bv)
                    packed = st.cpu_simd_prepack_rhs(b)
                    out = st.Tensor(size, size, [0.0] * (size * size))
                    packed_out = st.Tensor(size, size, [0.0] * (size * size))
                    ta = torch.tensor(av, dtype=torch.float32).reshape(size, size)
                    tb = torch.tensor(bv, dtype=torch.float32).reshape(size, size)
                    tout = torch.empty_like(ta)
                    reference = ta.double() @ tb.double()
                    functions = {
                        "st_faer_out": lambda: a.matmul(b, backend="faer", out=out),
                        "st_packed_out": lambda: a.matmul_simd_prepacked(packed, out=packed_out),
                        "torch_cpu_out": lambda: torch.mm(ta, tb, out=tout),
                    }
                    for name, function in functions.items():
                        result = function()
                        actual = result if name.startswith("torch") else torch.tensor(result.tolist(), dtype=torch.float64)
                        case["correctness"][name] = helper.correctness(actual, reference, torch, 1e-4, 1e-5)
                        del result
                    dispatches = {name: lambda fn=fn: invoke_and_discard(fn)
                                  for name, fn in functions.items()}
                    case["timings"] = helper.paired_timings(dispatches, 5, 20, lambda: None, seed)
            report["status"] = "passed"
        except Exception as error:
            report.update(status="error", error=f"{type(error).__name__}: {error}")
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "cases": len(report["cases"])}))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
