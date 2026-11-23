"""Reproduce the pinned GPT-2 margin-VJP result and split size from direction. No model."""
import json
import math
from pathlib import Path
import runpy
import sys
from statistics import median

study = runpy.run_path(str(Path(__file__).with_name("study.py")))
PINS = {
    "plan.json": "cd0c2131b579db2f1ab2b509c2d1c7fd8afc249f7e8f7fe6c2b102fd2a7117dd",
    "attempt/rows.jsonl": "386def6c7b639879be6aa53476c88e286880bbe4475814c0a00560825c4ec6f6",
    "attempt/result.json": "aaf38a614d3189511926fdb961a9dd637afb2074f12ab2ab73458591c1762238",
    "attempt/receipt.json": "b96233e4efa6fe3b6d8146aa422707bf768c018b31396b8cf83598cba1e68f0c",
}


def split(arms):
    norm, dist = study["norm"], study["dist"]
    affine, saturation, projection, both = (arms[k] for k in study["ARMS"])
    if not len(affine) or any(len(v) != len(affine) for v in arms.values()):
        raise ValueError("incompatible derivative widths")
    sizes = {k: norm(v) for k, v in arms.items()}
    cos = {}
    for k in ("saturation", "projection"):
        denom = sizes[k] * sizes["both"]
        cos[k] = math.fsum(x*y for x, y in zip(arms[k], both))/denom if denom else None
    return dict(
        ratios={k: v/sizes["affine"] if sizes["affine"] else None for k, v in sizes.items()},
        cosine_to_full=cos,
        projection_closer=dist(projection, both) < dist(saturation, both),
        saturation_more_aligned=(cos["saturation"] > cos["projection"]
                                 if all(v is not None for v in cos.values()) else None),
    )


def summarize(directory):
    files = {}
    for name, pin in PINS.items():
        data = (directory/name).read_bytes()
        if study["digest"](data) != pin:
            raise ValueError("input pin mismatch: " + name)
        files[name] = data
    plan = json.loads(files["plan.json"])
    result = study["replay"](plan, directory)
    if study["encoded"](result) != files["attempt/result.json"]:
        raise ValueError("archived result did not reproduce")
    output = {}
    def med(values):
        measured = [v for v in values if v is not None]
        return median(measured) if measured else None
    for host in study["HOSTS"]:
        rows = [r["receipt"] for r in map(json.loads, files["attempt/rows.jsonl"].splitlines())
                if r["host"] == host]
        output[host] = {}
        for blocks in (("gate",), ("bias",), ("gate", "bias")):
            measurements = []
            for row in rows:
                arms = {arm: [x for b in blocks for x in study["vector"](maps[b], plan["shape"][1])]
                        for arm, maps in row["arms"].items()}
                measurements.append(split(arms))
            output[host]["+".join(blocks)] = dict(
                n=len(rows),
                median_norm_ratio={k: med(m["ratios"][k] for m in measurements) for k in study["ARMS"]},
                median_cosine_to_full={k: med(m["cosine_to_full"][k] for m in measurements)
                                       for k in ("saturation", "projection")},
                projection_closer=sum(m["projection_closer"] for m in measurements),
                saturation_more_aligned=sum(m["saturation_more_aligned"] is True for m in measurements),
                undefined_alignment=sum(m["saturation_more_aligned"] is None for m in measurements),
            )
    return dict(pins=PINS, pretrained_calls_this_readback=0, original_calls=result["calls"],
                original_local_maps=result["maps"], original_gate=result["gate"], descriptive=output)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: block_readback.py ARCHIVED_GPT2_DIRECTORY")
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2, sort_keys=True, allow_nan=False))
