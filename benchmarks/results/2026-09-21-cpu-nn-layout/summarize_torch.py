"""Compute all matched high-level ratios, without combining packing with model timing."""
import json
import math
from pathlib import Path
import statistics
import sys

def summarize(root):
    torch = json.loads((root / "torch.json").read_text())
    assert len(torch["cases"]) == 12 and all(case["valid"] for case in torch["cases"])
    torch_cases = {(case["operation"], case["rows"], case["inner"], case["cols"]): case for case in torch["cases"]}
    assert len(torch_cases) == 12
    results = []
    for round_id in ["a", "b"]:
        native = json.loads((root / "native" / ("candidate-" + round_id + ".json")).read_text())
        for case in native["cases"]:
            if case["operation"] not in ["linear", "mlp"]:
                continue
            reference = torch_cases[tuple(case[key] for key in ["operation", "rows", "inner", "cols"])]
            results.append({"operation": case["operation"], "shape": [case[key] for key in ["rows", "inner", "cols"]],
                            "backend": case["backend"], "invalidate": case["invalidate"], "round": round_id,
                            "torch_over_spiral_ratio": statistics.median(reference["elapsed_ns"]) / statistics.median(case["elapsed_ns"])})
    groups = {}
    for result in results:
        group = f"{result['operation']}/{result['backend']}/{result['invalidate']}"
        groups.setdefault(group, []).append(result["torch_over_spiral_ratio"])
    return {"results": results, "groups": {group: {"geomean": math.exp(statistics.mean(math.log(value) for value in values)),
            "minimum": min(values), "maximum": max(values), "spiral_favorable": sum(value > 1 for value in values), "count": len(values)} for group, values in sorted(groups.items())},
            "boundary": "one Torch run reused for both native rounds and cache states; invalidation has no Torch equivalent; Python versus Rust eager entry overhead is included, not isolated kernel speed"}

if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
