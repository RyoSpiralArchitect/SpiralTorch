"""Offline CPU A/B for absolute set likelihood or signed menu ranking. Artificial weights only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
from statistics import median
import subprocess
import sys
import time


FIXTURE = Path(__file__).resolve().parents[1] / "bindings/st-py/tests/test_native_causal_lm.py"
CASES = {
    "chain": [list(range(10, 10 + n)) for n in range(1, 13)],
    "fork": [[10, 11, 12, 13, 14, 15, i] for i in range(64, 80)],
    "single": [[10, 11, 12, 13, 14, 15, 64]],
}
SHAPE = dict(width=4, heads=2, positions=24, layers=2, vocabulary=8192)
PROMPT, STOP, PAIRS = [2, 4, 6], 0, 6
MODES = ("forward", "forward_backward")
THREADS = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def check(value, why):
    if not value:
        raise ValueError(why)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def sizes(lines):
    prefixes = {tuple(line[:n]) for line in lines for n in range(len(line) + 1)}
    leaves = sum(not any(len(other) > len(line) and other[:len(line)] == line for other in lines) for line in lines)
    return dict(targets=sum(len(line)+1 for line in lines), prefixes=len(prefixes), leaves=leaves)


def recipes(objective):
    if objective == "set":
        return {case:(lines, {}) for case, lines in CASES.items()}
    check(objective == "ranking", "unknown objective")
    inputs = {"chain": CASES["chain"], "fork": CASES["fork"], "pair": [[64], [65]]}
    return {case+"_"+kind:(lines,dict(gold_index=len(lines)//2,mix=mix))
            for case, lines in inputs.items() for kind,mix in (("ce",0.), ("mixed",.5), ("rank",1.))}


def worker(pin, reverse, objective):
    class NativeOnly:
        def find_spec(self, name, path=None, target=None):
            if name.partition(".")[0] in {"torch", "numpy", "transformers", "safetensors", "accelerate", "mlx"}:
                raise AssertionError("unexpected foreign ML import: " + name)
    sys.meta_path.insert(0, NativeOnly())
    fixture = runpy.run_path(str(FIXTURE))
    st = fixture["st"]
    check(sha(st._rs.__file__) == pin, "wrong installed native library")
    config, weights = fixture["fixture"](**SHAPE)
    base = st.NativeCausalLm.from_gpt2(config, weights)
    state = json.loads(fixture["artifact"](base, path="transformer.h.1.attn", trainable=True).state_json())
    site = state["placements"][0]
    site["recipe"] = dict(schema="spiraltorch.fractional_memory_adapter.v1", features=4, strength=.5,
                          kernel=dict(kernel_len=4, step=.8, max_values=4096, max_products=32768))
    site["tensors"] = [t for t in site["tensors"] if t["name"] == "gate"] + [
        dict(name="log_alpha", values=[-.2], shape=[], role="parameter", requires_grad=True)]
    lora = json.loads(base.with_lora(1, 2, 3.).lora_to_json())
    lora.update(a=[(i-4)/32 for i in range(8)], b=[(3-i)/64 for i in range(8)])
    model = base.with_lora_json(json.dumps(lora)).with_geometry(st.GeometryAdapterArtifact.create_json(json.dumps(state)))
    model = model.with_readout_accumulator("f32_wide_backward")
    owners = [model.base_model_identity, model.lora_artifact_id, model.geometry_artifact_id]
    cases = recipes(objective)
    jobs = [(case, mode) for case in cases for mode in MODES]
    if reverse:
        jobs.reverse()
    for warmup in (True, False):
        for case, mode in jobs:
            lines, options = cases[case]
            if objective == "ranking":
                fn = model.line_menu_loss if mode == "forward" else model.geometry_line_menu_gradients
            else:
                fn = model.line_set_loss if mode == "forward" else model.geometry_line_set_gradients
            cpu, wall = time.process_time_ns(), time.perf_counter_ns()
            result = fn(PROMPT, lines, stop_id=STOP, **options)
            wall, cpu = time.perf_counter_ns()-wall, time.process_time_ns()-cpu
            report = result if mode == "forward" else (result.line_menu_report() if options else result.line_set_report())
            extra = {} if mode == "forward" else dict(gradient=result.parameter_gradients(), targets=result.target_tokens,
                                                      shape=result.window_shape)
            print(json.dumps(dict(warmup=warmup, case=case, mode=mode, cpu_ns=cpu, wall_ns=wall,
                                  owners=owners, report=report, **extra), allow_nan=False), flush=True)
            del result


def numerical(a, b, cases):
    for key in ("owners", "case", "mode", "warmup"):
        check(a[key] == b[key], "different "+key)
    lines, options = cases[a["case"]]
    work = sizes(lines)
    for row in (a,b):
        r = row["report"]
        check(r["target_counts"] == [len(line)+1 for line in lines] and r["stop_id"] == STOP,
              "wrong target domain")
        scalars = ("loss", "menu_nll", "gold_mean_nll", "rival_mass") if options else ("loss", "set_nll")
        for key in scalars:
            check(abs(r[key] - a["report"][key]) <= 2e-6, "different "+key)
        if options:
            check(all(r[k] == v for k,v in options.items()), "wrong gold index or mix")
            for key in ("winner_index", "shared_content_tokens"):
                check(r[key] == a["report"][key], "different "+key)
            for key in ("line_nlls", "suffix_nlls", "menu_probabilities"):
                check(len(r[key]) == len(lines) and all(abs(x-y)<=2e-6 for x,y in zip(r[key],a["report"][key])), "different "+key)
        else:
            check(r["prefix_nodes"] == work["prefixes"], "wrong set prefix count")
        check(row["wall_ns"] > 0 and row["cpu_ns"] > 0, "missing clock")
    check((b["report"]["decoder_lanes"], b["report"]["prefix_nodes"]) == (work["leaves"],work["prefixes"]),
          "wrong physical work")
    diff = 0.
    if a["mode"] == "forward_backward":
        for row in (a,b):
            check(row["targets"] == work["targets"] and row["shape"] == [len(lines), len(PROMPT)+max(map(len, lines))+1],
                  "logical gradient domain changed")
        x, y = a["gradient"], b["gradient"]
        check(x.keys() == y.keys() == {"transformer.h.1.attn"}, "wrong gradient owner")
        for path in x:
            check(x[path].keys() == y[path].keys() == {"gate","log_alpha"}, "wrong gradient parameters")
            check(all(any(v != 0. for v in param) for param in x[path].values()), "unexercised reference gradient")
            for k, left in x[path].items():
                right = y[path][k]
                check(len(left) == len(right) == (4 if k == "gate" else 1), "wrong gradient width")
                for u, v in zip(left, right):
                    diff = max(diff, abs(u-v))
                    check(math.isfinite(u) and math.isfinite(v) and abs(u-v) <= 4e-7+4e-5*abs(u), "different parameter gradient")
    return diff


def run(args):
    out = args.output.resolve()
    os.umask(0o077)
    out.mkdir(mode=0o700, exist_ok=False)
    order = [["reference", "candidate"] if i%2 == 0 else ["candidate", "reference"] for i in range(PAIRS)]
    cases = recipes(args.objective)
    calls = 2 * PAIRS * 2 * len(cases) * len(MODES)
    nll = "menu_nll" if args.objective == "ranking" else "set_nll"
    pins = dict(reference=args.reference_pin, candidate=args.candidate_pin)
    pythons = dict(reference=args.reference_python, candidate=args.candidate_python)
    check(pins["reference"] != pins["candidate"] and all(len(v)==64 for v in pins.values()), "need distinct native hashes")
    save(out/"plan.json",dict(cases={k:v for k,(v,_) in cases.items()}, options={k:v for k,(_,v) in cases.items()},
         objective=args.objective, prompt=PROMPT, stop=STOP, model=SHAPE, native=pins, order=order,
         fixture_sha256=sha(FIXTURE), harness_sha256=sha(__file__), work={k:sizes(v) for k,(v,_) in cases.items()},
         slot="fresh process, all cases forward/backward once warm then once timed; reverse job order on odd pairs",
         precision="f32_wide_backward", geometry="fractional memory", loss_atol=2e-6, grad_atol=4e-7, grad_rtol=4e-5,
         wall_limit_seconds=180, calls=calls, updates=0, threads=1))
    times, stable, worst, max_nll = {}, {}, 0., 0.
    deadline = time.monotonic() + 180
    for i, roles in enumerate(order):
        pair = {}
        for role in roles:
            env = dict(os.environ, **{k:"1" for k in THREADS}, SPIRALTORCH_BACKEND="cpu",
                       SPIRALTORCH_AUTOTUNE_STORE=str(out/f"{i}-{role}.tune.json"))
            dest = out/f"{i}-{role}.jsonl"
            with dest.open("x") as stdout, (out/f"{i}-{role}.stderr").open("x") as stderr:
                subprocess.run([pythons[role], "-I", str(Path(__file__).resolve()), "worker", pins[role], str(i%2),args.objective],
                               stdout=stdout, stderr=stderr, check=True, env=env, timeout=min(25., max(.01,deadline-time.monotonic())))
            rows = [json.loads(line) for line in dest.read_text().splitlines()]
            pair[role] = {(r["warmup"],r["case"],r["mode"]):r for r in rows}
            expected = {(warm,case,mode) for warm in (True,False) for case in cases for mode in MODES}
            check(len(rows)==len(expected) and pair[role].keys()==expected, "missing or duplicate call")
            for row in rows:
                key = (role, row["case"], row["mode"])
                signature = {k:v for k,v in row.items() if k not in {"warmup","wall_ns","cpu_ns"}}
                check(stable.setdefault(key, signature)==signature,"same wheel changed across observations")
        for key, a in pair["reference"].items():
            b = pair["candidate"][key]
            worst = max(worst,numerical(a,b,cases))
            max_nll = max(max_nll,abs(a["report"][nll]-b["report"][nll]))
            if not key[0]:
                times.setdefault((key[1],key[2]), []).append((a["wall_ns"], b["wall_ns"], a["cpu_ns"], b["cpu_ns"]))
    summary = {"/".join(k):dict(pairs=len(v), reference_ms=median(r[0] for r in v)/1e6,
                candidate_ms=median(r[1] for r in v)/1e6, paired_speedup=median(r[0]/r[1] for r in v),
                paired_cpu_speedup=median(r[2]/r[3] for r in v)) for k,v in times.items()}
    save(out/"results.json",dict(plan_sha256=sha(out/"plan.json"), calls=calls, updates=0,
         max_abs_gradient_diff=worst,**{"max_abs_"+nll+"_diff":max_nll},timings=summary,
         raw_sha256={p.name:sha(p) for p in sorted(out.glob("*.jsonl"))}))


if __name__ == "__main__":
    if sys.argv[1:2] == ["worker"]:
        worker(sys.argv[2], sys.argv[3]=="1", sys.argv[4])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--objective", choices=("set","ranking"), default="set")
        for role in ("reference", "candidate"):
            parser.add_argument("--"+role+"-python", required=True)
            parser.add_argument("--"+role+"-pin", required=True, help="SHA256 of the installed native extension")
        parser.add_argument("--output", type=Path, required=True)
        run(parser.parse_args())
