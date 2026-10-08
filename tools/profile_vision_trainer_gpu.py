#!/usr/bin/env python3
"""Profile existing VJP passes on retained real-data timing cases; no download."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--timing-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64])
    parser.add_argument("--modes", nargs="+", choices=["plain", "feedback"], default=["plain", "feedback"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats <= 0 or any(not x or len(x) != len(set(x)) for x in (args.seeds, args.batches, args.modes)):
        raise ValueError("nonempty unique cases and positive repeats required")
    module_path = Path(__file__).with_name("run_vision_matched_learning.py")
    spec = importlib.util.spec_from_file_location("matched_vision", module_path)
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    data, evidence = shared.dataset(argparse.Namespace(data_root=args.data_root, download=False,
        train_per_class=128, test_per_class=32, batch_size=1))
    pixels = data[0][0].tobytes()
    assert sha(pixels) == evidence["train_pixels_sha256"]
    args.output.mkdir(parents=True, exist_ok=False)
    pixels_file = args.output.resolve() / "pixels.u8"
    with pixels_file.open("xb") as stream:
        stream.write(pixels)
    binary = args.binary.resolve()
    binary_sha = sha(binary.read_bytes())
    records = []
    for seed in args.seeds:
        for batch in args.batches:
            for mode in args.modes:
                retained = args.timing_root / f"seed-{seed}-batch-{batch}-{mode}-0-candidate"
                old = json.loads((retained / "result.json").read_text())
                assert old["status"] == "passed" and old["contract"]["data"] == evidence
                initial, final = retained / "initial.json", retained / "spiraltorch-final.json"
                assert sha(initial.read_bytes()) == old["contract"]["initial_checkpoint_sha256"]
                assert sha(final.read_bytes()) == old["final_checkpoint"]["sha256"]
                for repeat in range(args.repeats):
                    name = f"seed-{seed}-batch-{batch}-{mode}-{repeat}"
                    recipe = dict(dataset=dict(pixels_file=str(pixels_file), pixels_sha256=sha(pixels),
                        labels=data[0][1].tolist(), ids=evidence["train_indices"]),
                        dataset_id=old["contract"]["dataset_sha256"], initial_file=str(initial.resolve()),
                        initial_sha256=sha(initial.read_bytes()), expected_final_file=str(final.resolve()),
                        expected_final_sha256=sha(final.read_bytes()), seed=seed, steps=old["recipe"]["steps"],
                        warmup=old["recipe"]["warmup"], profile_first=bool(repeat % 2))
                    recipe_file = args.output / f"{name}.json"
                    save(recipe_file, recipe)
                    with (args.output / f"{name}.log").open("x") as log:
                        subprocess.run([str(binary), str(recipe_file), str(args.output / name)],
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                    result = json.loads((args.output / name / "result.json").read_text())
                    records.append(dict(case=name, seed=seed, batch_size=batch, mode=mode, repeat=repeat, result=result))
                    print(json.dumps(dict(case=name, passed=result["passed"])), flush=True)
    assert sha(binary.read_bytes()) == binary_sha
    save(args.output / "summary.json", dict(schema="spiraltorch.vision.gpu_profile_sweep.v1", passed=True,
        requested=dict(seeds=args.seeds, batches=args.batches, modes=args.modes, repeats=args.repeats),
        binary_sha256=binary_sha, launcher_sha256=sha(Path(__file__).read_bytes()), records=records))


if __name__ == "__main__":
    main()
