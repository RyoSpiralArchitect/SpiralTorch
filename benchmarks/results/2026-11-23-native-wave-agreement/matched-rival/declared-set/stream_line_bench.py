"""Compare dense cached and tiled cached terminated scoring in fresh CPU processes."""
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import runpy
import statistics
import subprocess
import sys
import time

VOCAB = 50_257
PROMPT = [7, 13, 18, 33, 29, 50, 73, 90]
LINES = [[1+arm*257+j for j in range(32)] for arm in range(4)]
ARMS = ("dense_cache", "stream", "one_row")


def no_foreign(event, args):
    if event == "import" and args[0].partition(".")[0] in {"torch","numpy","safetensors","accelerate","transformers"}:
        raise RuntimeError("unexpected external ML import")


def save(path, data):
    with path.open("x") as f:
        json.dump(data, f, allow_nan=False, indent=2)
        f.write("\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(directory):
    sys.addaudithook(no_foreign)
    directory.mkdir(mode=0o700)
    helpers = runpy.run_path(str(Path(__file__).resolve().parents[5] / "bindings/st-py/tests/test_native_causal_lm.py"))
    st = helpers["st"]
    config, weights = helpers["fixture"](width=8, heads=2, layers=1, positions=48, vocabulary=VOCAB, raw_weights=True)
    with (directory/"synthetic.safetensors").open("xb") as f:
        f.write(weights)
    base = st.NativeCausalLm.from_gpt2(config, st.SafeTensorArchive.from_bytes(weights))
    state = json.loads(helpers["artifact"](base, trainable=True).state_json())
    placement = state["placements"][0]
    placement["path"] = "transformer.h.0.attn"
    placement["recipe"] = {"schema":"spiraltorch.wave_gate_adapter.v1","features":8,"strength":0.25,
                           "kernel":{"curvature":-1., "saturation":1., "porosity":0.05,"max_values":32768}}
    for t in placement["tensors"]:
        t.update(shape=[8], values=[((i%7)-3)*0.03 for i in range(8)])
    art = st.GeometryAdapterArtifact.create_json(json.dumps(state))
    save(directory/"fixture.json",dict(config=config, weights_sha256=digest(weights), base_id=base.base_model_identity,
                                      geometry=art.to_json(), geometry_id=art.artifact_id))


def peak_bytes():
    if sys.platform not in ("darwin", "linux"):
        raise ValueError("ru_maxrss units unspecified on this platform")
    n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return n if sys.platform == "darwin" else n*1024


def worker(directory, arm, output):
    sys.addaudithook(no_foreign)
    import spiraltorch as st
    f = json.loads((directory/"fixture.json").read_bytes())
    base = st.NativeCausalLm.from_gpt2(f["config"],st.SafeTensorArchive(str(directory/"synthetic.safetensors")))
    assert base.base_model_identity == f["base_id"]
    art = st.GeometryAdapterArtifact.from_json(f["geometry"], expected_id=f["geometry_id"],
                                                expected_base_model_identity=f["base_id"])
    m = base.with_geometry(art)
    ctx = None if arm == "padded" else st.NativeCausalLmContext(m, PROMPT, max_cache_values=1<<20)
    kw = {"max_tile_values": VOCAB} if arm == "one_row" else {}
    before = peak_bytes()
    start = time.perf_counter_ns()
    if arm == "padded":
        scores, receipt = m.score_lines(PROMPT, LINES, stop_id=0), {}
    else:
        receipt = ctx.score_lines(LINES, stop_id=0, **kw)
        scores = receipt.pop("scores")
    elapsed = time.perf_counter_ns()-start
    after = peak_bytes()
    assert ctx is None or (ctx.token_ids == PROMPT and ctx.geometry_artifact_id == f["geometry_id"])
    save(output, dict(arm=arm,base_id=f["base_id"],geometry_id=f["geometry_id"],ns=elapsed,
                      highwater_before_bytes=before,highwater_after_bytes=after,
                      highwater_increase_bytes=after-before,receipt=receipt,scores=scores))


def compare_scores(a,b):
    assert a.keys() == b.keys()
    for k in ("target_counts","stop_id","winner_index","prefix_nodes","decoder_lanes","shared_content_tokens"):
        assert a[k] == b[k], k
    assert a["target_counts"] == [33]*4 and a["prefix_nodes"] == 129
    def compare(x,y):
        if isinstance(x,list):
            assert len(x) == len(y)
            return max(compare(u,v) for u,v in zip(x,y))
        d = abs(x-y)
        if not math.isfinite(d) or d > 3e-6:
            raise ValueError("score mismatch: %r" % d)
        return d
    return max(compare(a[k],b[k]) for k in ("menu_set_nll","line_nlls","conditional_nlls","token_nlls"))


def compare(legacy, candidate, directory, out):
    out.mkdir(mode=0o700)
    fixture = json.loads((directory/"fixture.json").read_bytes())
    assert digest((directory/"synthetic.safetensors").read_bytes()) == fixture["weights_sha256"]
    env = dict(os.environ, RAYON_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    def once(python,arm,name):
        dest=out/(name+".json")
        subprocess.run([python,"-IB",str(Path(__file__).resolve()),"worker",str(directory),arm,str(dest)],
                       env=env,check=True,timeout=45)
        r=json.loads(dest.read_bytes())
        assert (r["arm"],r["base_id"],r["geometry_id"]) == (arm,fixture["base_id"],fixture["geometry_id"])
        return r
    oracle = once(legacy,"padded","padded")
    trials = []
    for i,order in enumerate(itertools.permutations(ARMS)):
        row = {}
        for arm in order:
            r = once(legacy if arm=="dense_cache" else candidate,arm,"%02d-%s"%(i,arm))
            scores = r.pop("scores")
            error = compare_scores(scores,oracle["scores"])
            work = r["receipt"]
            assert (work["decoded_content_tokens"],work["decoder_chunks"]) == (128,4)
            if arm == "dense_cache":
                assert "readout_batches" not in work, "not the baseline wheel"
            else:
                assert (work["scanned_logit_values"],work["retained_edge_logits"]) == (129*VOCAB,132)
                q,batches = (5,28) if arm=="stream" else (1,128)
                assert (work["peak_projected_logit_values"],work["readout_batches"]) == (q*VOCAB,batches)
            row[arm] = {k:r[k] for k in ("ns","highwater_before_bytes","highwater_after_bytes","highwater_increase_bytes")}
            row[arm].update(max_abs_vs_padded=error,score_sha256=digest(json.dumps(scores,sort_keys=True).encode()))
        trials.append(dict(order=order,observations=row))
    medians={arm:{k:statistics.median(t["observations"][arm][k] for t in trials) for k in
                  ("ns","highwater_before_bytes","highwater_after_bytes","highwater_increase_bytes")} for arm in ARMS}
    save(out/"summary.json",dict(schema="spiraltorch.artificial_streamed_line_memory.v1",
         platform=platform.system(),release=platform.release(),machine=platform.machine(),model="artificial WaveGate; 1 layer, width 8, 2 heads",
         vocabulary=VOCAB,geometry_id=fixture["geometry_id"],weights_sha256=fixture["weights_sha256"],
         prompts=1,prompt_length=8,lines=4,content_length=32,reference="one separate old padded process",
         threads_env=1,median=medians,trials=trials))


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "prepare" and len(sys.argv)==3:
        prepare(Path(sys.argv[2]))
    elif command == "worker" and len(sys.argv)==5 and sys.argv[3] in ("padded",*ARMS):
        worker(Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]))
    elif command == "compare" and len(sys.argv)==6:
        compare(sys.argv[2],sys.argv[3],Path(sys.argv[4]),Path(sys.argv[5]))
    else:
        sys.exit("prepare DIR | compare OLD_PY NEW_PY FIXTURE_DIR NEW_OUTPUT_DIR")
