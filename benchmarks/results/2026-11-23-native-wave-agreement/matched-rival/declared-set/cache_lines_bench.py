"""Matched full/cold-cache/reused-cache line scoring on deterministic WaveGate weights."""
import itertools
import json
from pathlib import Path
import runpy
import statistics
import sys
import time


def run():
    st, model = runpy.run_path(str(Path(__file__).with_name("cache_suffix_bench.py")))["artificial_model"]()
    prompt = [(i*31+7) % 512 for i in range(32)]
    def prefill():
        return st.NativeCausalLmContext(model, prompt, max_cache_values=1 << 20)
    root = prefill()
    arms = ("padded", "inc_prefill", "reused")
    records = []
    for name, lines in (("shared", [[1,2,3,4,5,i] for i in range(6,10)]),
                        ("disjoint", [[10*i+j for j in range(1,7)] for i in range(4)])):
        ref = model.score_lines(prompt, lines, stop_id=0)
        counts = root.score_lines(lines, stop_id=0)
        counts.pop("scores")
        def observe(arm):
            start = time.perf_counter_ns()
            if arm == "padded":
                s = model.score_lines(prompt, lines, stop_id=0)
            else:
                s = (prefill() if arm == "inc_prefill" else root).score_lines(lines, stop_id=0)["scores"]
            elapsed_ms = (time.perf_counter_ns()-start)/1e6
            for k in ("target_counts","stop_id","winner_index","decoder_lanes","shared_content_tokens","prefix_nodes"):
                assert s[k] == ref[k], (arm, name, k)
            def errs(a,b):
                if isinstance(a,list):
                    assert len(a) == len(b)
                    return [e for x,y in zip(a,b) for e in errs(x,y)]
                d = abs(a-b)
                assert d < 3e-5, (arm, name, d)
                return [d]
            error = max(max(errs(s[k], ref[k])) for k in ("token_nlls","line_nlls","conditional_nlls","menu_set_nll"))
            return elapsed_ms, error
        for arm in arms:
            observe(arm)
        trials = []
        worst = 0.
        for order in itertools.permutations(arms):
            times = {}
            for arm in order:
                times[arm], error = observe(arm)
                worst = max(worst,error)
            trials.append(dict(order=list(order), ms=times))
        records.append(dict(menu=name, work=counts, trials=trials, max_abs_error=worst,
                            median_ms={a:statistics.median(t["ms"][a] for t in trials) for a in arms},
                            median_paired_speedup={a:statistics.median(t["ms"]["padded"]/t["ms"][a] for t in trials)
                                                   for a in arms[1:]}))
    return dict(schema="spiraltorch.artificial_cached_lines_timing.v1", model="WaveGate, CPU, 4 layers x 48, 4 heads, vocab 512",
                prompt=32, lines=4, content_per_line=6, results=records)


if __name__ == "__main__":
    with Path(sys.argv[1]).open("x") as f:
        json.dump(run(), f, allow_nan=False, indent=2)
        f.write("\n")
