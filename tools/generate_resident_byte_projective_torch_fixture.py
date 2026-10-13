"""Freeze six synthetic learning cases; no corpus-quality or speed claim."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "geometry", Path(__file__).with_name("generate_resident_byte_geometry_torch_fixture.py"))
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)
    cases = []
    for mode in ("projective", "four_way_shared", "four_way_query"):
        for blocks, external, isolated, seed in [(1, False, True, 1761), (2, True, False, 1863)]:
            case = geometry.case(blocks, external, isolated, seed,
                                 projective=mode == "projective",
                                 projective_mixture=mode != "projective",
                                 query_mixture=mode == "four_way_query")
            case["name"] = mode + "_" + case["name"]
            cases.append(case)
    payload = {
        "schema": "spiraltorch.resident_byte_geometry_projective.torch_fixture.v1",
        "torch": torch.__version__, "device": "cpu", "dtype": "float32",
        "tolerance": {"atol": 3e-6, "rtol": 5e-5, "geometry_relative_l2": .002},
        "scope": "QKV byte LM correctness, full gradients, 16 SGD updates, not language quality",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(cases), "steps_per_case": 16,
                      "schema": payload["schema"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
