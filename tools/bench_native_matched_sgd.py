"""Offline, synthetic CPU benchmark of three independent SGD arms vs the same lockstep updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
from statistics import median
import sys
import time

FIXTURE = Path(__file__).resolve().parents[1] / "bindings/st-py/tests/test_native_causal_lm.py"
THREADS = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
SHAPE = dict(width=64, heads=4, positions=32, layers=6, vocabulary=97)
ARMS = ["lora_only", "frozen_wave_lora", "joint_wave_lora"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, obj):
    with path.open("x", encoding="utf8") as stream:
        json.dump(obj, stream, allow_nan=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory for protocol and every timed pair")
    out = parser.parse_args().output
    os.umask(0o077)
    out.mkdir(mode=0o700, exist_ok=False)
    for name in THREADS:
        os.environ[name] = "1"
    os.environ["SPIRALTORCH_AUTOTUNE_STORE"] = str(out.resolve() / "autotune.json")
    forbidden = {"numpy", "torch", "transformers", "accelerate", "safetensors", "tokenizers"}
    class NativeOnly:
        def find_spec(self, name, path=None, target=None):
            if name.partition(".")[0] in forbidden:
                raise ModuleNotFoundError("optional ML package disabled", name=name)
    sys.meta_path.insert(0, NativeOnly())
    import spiraltorch as st

    order = [["independent","matched"] if i % 2 == 0 else ["matched","independent"] for i in range(6)]
    protocol = dict(shape=SHAPE, arms=ARMS, layer=5, rank=2, alpha=4., context=25, batch_windows=2,
                    learning_rate=.02, epochs=1, updates=6, target_tokens_per_update=8, order=order,
                    warmup=["independent","matched"], thread_env={n:os.environ[n] for n in THREADS},
                    native_sha256=sha(st._rs.__file__), script_sha256=sha(__file__), fixture_sha256=sha(FIXTURE),
                    platform=platform.platform(), python=platform.python_version(),
                    scope="sequential warm CPU updates through Python; no setup, scoring, serialization or disk in timed loop")
    save(out/"protocol.json",protocol)
    fixture = runpy.run_path(str(FIXTURE))["fixture"]
    cfg, weights = fixture(**SHAPE)
    base = st.NativeCausalLm.from_gpt2(cfg, weights)
    require(base.execution_backend == "rust_f32_cpu_prepacked", "benchmark requires the native CPU route")
    lora = base.with_lora(protocol["layer"],protocol["rank"],protocol["alpha"])
    def parameter(name, values, scalar=False):
        return dict(name=name, role="parameter", requires_grad=True, values=values, shape=[] if scalar else [len(values)])
    width = SHAPE["width"]
    wave = st.GeometryAdapterArtifact.create_json(json.dumps({
        "schema":"spiraltorch.geometry_adapter_artifact.v1","base_model_identity":base.base_model_identity,
        "policy":"full_prefix_f32_no_cache.v1","training":True,"placements":[{
            "path":f"transformer.h.{protocol['layer']}.attn","training":True,
            "recipe":{"schema":"spiraltorch.wave_gate_adapter.v2","features":width,"strength":.25,
                      "learnable_radius":True,"kernel":{"curvature":-1.,"saturation":1.,"porosity":.05,"max_values":8192}},
            "tensors":[parameter("gate",[(j%3-1)*.01 for j in range(width)]),
                       parameter("bias",[(1 if j%2 else -1)*.02 for j in range(width)]),
                       parameter("log_radius",[0.],True)]}]}))
    both = lora.with_geometry(wave)
    docs = [[(7*d+13*t+t*t) % SHAPE["vocabulary"] for t in range(protocol["context"])] for d in range(12)]
    labels = [[-100]*(len(row)-4) + row[-4:] for row in docs]
    session = st.NativeCausalLmTrainingSession
    template = session(lora, docs, protocol["context"], labels=labels, window_policy="covering_targets",
                       mode="lora", learning_rate=protocol["learning_rate"], batch_windows=2)
    require(template.total_updates == protocol["updates"], "unexpected data schedule")

    def run(engine, verify_scores=False):
        arms = [template.fork_variant(m,mode) for m,mode in ((lora,"lora"),(both,"lora"),(both,"joint"))]
        updates = template.total_updates
        rows = []
        start = time.perf_counter_ns()
        if engine == "matched":
            for _ in range(updates):
                rows.append(session.step_matched(arms))
        else:
            for _ in range(updates):
                rows.append((None,[arm.step() for arm in arms]))
        ms = (time.perf_counter_ns()-start)/1e6
        trace = hashlib.sha256()
        for i,(depth,steps) in enumerate(rows,1):
            require(depth == (protocol["layer"] if engine == "matched" else None), "incorrect shared prefix")
            require(len(steps) == len(ARMS), "incorrect arm count")
            for s in steps:
                require(s.update == i and s.target_tokens == protocol["target_tokens_per_update"], "incorrect update/denominator")
                trace.update(s.to_json().encode()+b"\n")
        require(all(s.exhausted for s in arms) and arms[1].model.geometry_artifact_id == wave.artifact_id
                and arms[2].model.geometry_artifact_id != wave.artifact_id, "incorrect terminal owners")
        result = dict(ms=ms,trace_sha256=trace.hexdigest(),checkpoints=[s.checkpoint_id for s in arms])
        if verify_scores:
            score = st.NativeCausalLm.evaluate_variants([s.model for s in arms], docs, protocol["context"],
                                                       microbatch_windows=2, labels=labels)
            require(all(v.target_tokens == 48 for v in score.evaluations), "post-update score denominator differs")
            result["training_probe_nll"] = [v.nll for v in score.evaluations]
        return result

    reference = None
    ratios = []
    with (out/"observations.jsonl").open("x",encoding="utf8") as observations:
        for index, engines in enumerate([protocol["warmup"], *order]):
            pair = {}
            for engine in engines:
                result = run(engine,verify_scores=index == 0)
                observations.write(json.dumps(dict(pair=index,engine=engine,**result), allow_nan=False)+"\n")
                observations.flush()
                signature = (result["trace_sha256"],result["checkpoints"])
                if reference is None:
                    reference = signature
                require(signature == reference, "report or final checkpoint differs; stopping without retry")
                pair[engine] = result
            if index == 0:
                require(pair["matched"]["training_probe_nll"] == pair["independent"]["training_probe_nll"],
                        "post-update masked training probes differ")
            else:
                ratios.append(pair["independent"]["ms"]/pair["matched"]["ms"])
    require(not (forbidden & {k.partition(".")[0] for k in sys.modules}), "foreign ML module loaded")
    summary = dict(schema="spiraltorch.synthetic_matched_sgd_bench.v1", base=base.base_model_identity,
                   data_id=template.data_id, paired_speed_ratios=ratios, median_paired_speed_ratio=median(ratios),
                   trace_sha256=reference[0], checkpoint_ids=reference[1], protocol_sha256=sha(out/"protocol.json"),
                   observations_sha256=sha(out/"observations.jsonl"),
                   limits="one synthetic host/shape/roster; not GPU, real GPT-2, PyTorch, or a quality comparison")
    save(out/"result.json",summary)
    print(json.dumps(summary,allow_nan=False))


if __name__ == "__main__":
    main()
