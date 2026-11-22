"""Matched end-to-end Python calls on one synthetic native decoder, without pretrained inputs."""
import argparse
import json
from pathlib import Path
import runpy
import statistics
import sys
import time


def run():
    def native_only(event, args):
        if event == "import" and args[0].split(".")[0] in {"numpy", "torch", "transformers", "safetensors", "accelerate"}:
            raise AssertionError("unexpected ML import: " + args[0])
    sys.addaudithook(native_only)
    tests = Path(__file__).resolve().parents[5] / "bindings/st-py/tests"
    helper = runpy.run_path(str(tests / "test_native_causal_lm.py"))
    st = helper["st"]
    width, layers = 48, 4
    config, weights = helper["fixture"](width=width, heads=4, layers=layers, positions=72, vocabulary=512)
    base = st.NativeCausalLm.from_gpt2(config, weights)
    state = json.loads(helper["artifact"](base, trainable=True).state_json())
    placement = state["placements"][0]
    placement["recipe"] = {"schema": "spiraltorch.wave_gate_adapter.v1", "features": width, "strength": .25,
                           "kernel": {"curvature": -1., "saturation": 1., "porosity": .05, "max_values": 32768}}
    for tensor in placement["tensors"]:
        tensor.update(shape=[width], values=[((i % 7)-3)*.03 for i in range(width)])
    model = base.with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(state)))
    ids = [(i*31+7) % 512 for i in range(56)]
    prefix, results = ids[:24], []
    for length in (1, 8, 32):
        suffix = ids[24:24+length]
        reference = model.forward(prefix + suffix, [1, 24+length], last_only=True).values()
        def measure(branch):
            ctx = st.NativeCausalLmContext(model, prefix, max_cache_values=1 << 20)
            start = time.perf_counter_ns()
            if branch:
                child = ctx.with_suffix(suffix)
                values = child.logits().values()
            else:
                for token in suffix:
                    row = ctx.append(token)
                values = row.values()
            elapsed = (time.perf_counter_ns()-start)/1e6
            if len(values) != len(reference):
                raise ValueError("missing vocabulary rows")
            error = max(abs(a-b) for a, b in zip(values, reference))
            if not error <= 3e-5:
                raise ValueError("numerical check failed: " + str(error))
            return elapsed, error
        measure(False)
        measure(True)
        pairs, errors = [], []
        for i in range(8):
            times = {}
            for branch in ((False, True) if i % 2 == 0 else (True, False)):
                elapsed, error = measure(branch)
                times["chunk" if branch else "serial"] = elapsed
                errors.append(error)
            pairs.append(times)
        results.append(dict(known_tokens=length, pairs_ms=pairs, max_abs_full=max(errors),
                            median_serial_ms=statistics.median(x["serial"] for x in pairs),
                            median_chunk_ms=statistics.median(x["chunk"] for x in pairs),
                            median_paired_ratio=statistics.median(x["serial"]/x["chunk"] for x in pairs)))
    return dict(schema="spiraltorch.cached_suffix_artificial_bench.v1", prefix=24, width=width,
                layers=layers, heads=4, vocab=512, warmups_per_arm=1, results=results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    with args.output.open("x") as output:
        json.dump(run(), output, allow_nan=False, sort_keys=True, indent=2)
        output.write("\n")
