#!/usr/bin/env python3
"""Replay installed Python-wheel receipts inside an actual Node WASM runtime."""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
from pathlib import Path

import spiraltorch as st


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wasm-module", type=Path, required=True)
    parser.add_argument("--random-cases", type=int, default=64)
    args = parser.parse_args()
    base = {
        "forward_request": {
            "input": [0.2, -0.8, 0.5, 0.3, 0.7, -0.4],
            "potential": [0.3, -0.6, 0.2],
            "standard_normal": [0.3, -0.5, 0.7, -0.2, 0.1, 0.4],
            "rows": 2, "features": 3,
            "config": {"time_step": 0.3, "hopping_rate": -0.8, "loss_rate": 0.07, "noise_scale": 0.2},
        },
        "input_imaginary": [0.5, 0.1, -0.2, 0.6, -0.9, 0.2],
        "cotangent": {"real": [0.6, -0.2, 0.8, -0.3, 0.5, 0.1], "imaginary": [-0.4, 0.7, -0.6, 0.2, 0.8, -0.5]},
    }
    requests = [base]
    for change in [{"time_step": 0.0}, {"loss_rate": 0.0, "noise_scale": 0.0}, {"hopping_rate": 0.0}, {"loss_rate": 10.0, "noise_scale": 3.0}]:
        req = copy.deepcopy(base)
        req["forward_request"]["config"].update(change)
        requests.append(req)
    rng = random.Random(173)
    for _ in range(args.random_cases):
        req = copy.deepcopy(base)
        req["forward_request"].update({
            "input": [rng.uniform(-10, 10) for _ in range(6)],
            "potential": [rng.uniform(-20, 20) for _ in range(3)],
            "standard_normal": [rng.gauss(0, 1) for _ in range(6)],
            "config": {"time_step": rng.uniform(0.01, 1), "loss_rate": rng.uniform(0.01, 1), "hopping_rate": rng.uniform(-3, 3), "noise_scale": rng.uniform(0, 2)},
        })
        req["input_imaginary"] = [rng.uniform(-10, 10) for _ in range(6)]
        requests.append(req)
    cases = [{"request": req, "python_receipt": st.zspace_stochastic_schrodinger_complex_step(req)} for req in requests]
    result = subprocess.run([
        "node", str(Path(__file__).with_name("verify_zspace_complex_clients.cjs")), str(args.wasm_module.resolve()),
    ], input=json.dumps(cases), text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr)
    wasm_receipts = json.loads(result.stdout)
    assert len(wasm_receipts) == len(cases)
    for case, receipt in zip(cases, wasm_receipts):
        assert receipt == case["python_receipt"]
        assert st.validate_zspace_stochastic_schrodinger_complex(receipt) == receipt
    print(json.dumps({"status": "ok", "cross_client_cases": len(cases), "python_version": st.__version__, "wasm_module": str(args.wasm_module.resolve()), "forward_and_gradient_equal": True, "bidirectional_replay": True}, sort_keys=True))


if __name__ == "__main__":
    main()
