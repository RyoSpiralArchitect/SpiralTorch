"""One CPU pass over 9 pinned frozen endpoints x 96 training windows. Never calls an optimizer."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as m  # noqa: E402

sys.path.insert(0, str(m.ROOT / "tools"))
import byte_release_artifacts as wheels  # noqa: E402
from owned_process import run  # noqa: E402

CORE = "03bb3eda3138a0fb747f84b860fbbc970f198e79"
WHEEL = "f2b12c91971fe270344f826ee07531e2f4d20260d42adcfa193f442e0bdb40d2"
NATIVE = "2a44caf73fcd9135f91081bd62f7b98224323beacb93b4ddeef587a0a8d618f4"
FOREIGN = {"torch", "numpy", "transformers", "accelerate", "peft", "safetensors", "tokenizers"}
LIMIT = 1800


def source(commit):
    files = (Path(__file__).resolve(), m.HERE / "metrics.py", m.ROOT / "tools/owned_process.py",
             m.ROOT / "tools/byte_release_artifacts.py")
    pins = {str(f.relative_to(m.ROOT)): m.sha(f) for f in files}
    for path, pin in pins.items():
        data = subprocess.check_output(["git", "-C", str(m.ROOT), "show", f"{commit}:{path}"])
        m.check(m.hashlib.sha256(data).hexdigest() == pin, "uncommitted observation source")
    return pins


def allow_train_only(root, paths):
    # A Python-side guard for the old private run, not an OS sandbox. Rust only gets
    # explicit base paths, checkpoint bytes and the one current training window.
    root = root.resolve()
    permitted = {str(path.resolve(strict=True)) for path in paths}
    m.check(all(not f.is_symlink() and f.is_file() and f.resolve().is_relative_to(root) for f in paths),
            "expected regular private input files")
    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.abspath(os.fsdecode(args[0]))
            if path == str(root) or path.startswith(str(root) + os.sep):
                m.check(path in permitted and not args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC),
                        "forbidden previous-run access")
    sys.addaudithook(audit)


def train_plan(data):
    plan = []
    for key, ids, mask in zip(data["keys"], data["token_documents"], data["labels"], strict=True):
        first = next(i for i, t in enumerate(mask) if t != -100)
        m.check(0 < first < len(ids) <= 64 and mask == [-100] * first + ids[first:], "not answer-only supervision")
        plan.append([key, len(ids), len(ids) - first])
    m.check(len(plan) == m.N and len({r[0] for r in plan}) == m.N and sum(r[2] for r in plan) == m.TARGETS
            and sum(r[1] - 1 for r in plan) == m.ROWS, "unmatched training corpus")
    return plan


def worker(a):
    root = a.output
    started = m.read(root / "attempt.json")
    remaining = min(LIMIT, started["started_epoch"] + LIMIT - time.time()) - 3
    m.check(remaining > 1, "budget exhausted before worker")
    resource.setrlimit(resource.RLIMIT_CPU, (math.floor(remaining), math.floor(remaining)))
    m.write(root / "worker.json", dict(started_epoch=time.time(), limit_cpu=math.floor(remaining), retry=False))
    paths = [a.originals / "train.json", *(a.originals / o / (arm + ".ck") for o in m.ORDERS for arm in m.ARMS)]
    allow_train_only(a.originals, paths)
    entries = m.boundaries()
    m.check(m.sha(paths[0]) == m.TRAIN and all("sha256:" + m.sha(a.originals / order / (arm + ".ck")) == e["checkpoint_id"]
            for (order, arm), e in entries.items()), "changed original checkpoint or train-only dataset")
    data = m.read(paths[0])
    plan = train_plan(data)
    for k in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", SPIRALTORCH_AUTOTUNE_STORE=str(root / "autotune.json"))
    class Guard:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.partition(".")[0] in FOREIGN:
                raise ModuleNotFoundError("external ML disabled in read-only diagnostic", name=fullname)
    sys.meta_path.insert(0, Guard())
    m.check(not FOREIGN & {x.partition(".")[0] for x in sys.modules}, "foreign runtime already loaded")
    inventory = wheels.wheel_archive_inventory(a.wheel)
    m.check(inventory["sha256"] == WHEEL and inventory["native"]["sha256"] == NATIVE, "wrong instrument wheel")
    identity = wheels.installed_identity(inventory)
    import spiraltorch as st
    m.check(m.sha(Path(st._rs.__file__)) == NATIVE and all(m.sha(a.model / n) == h for n,h in m.MODEL.items()),
            "loaded extension or base bytes differ")
    pins = source(a.commit)
    m.write(root / "manifest.json", dict(schema="spiraltorch.native_endpoint_pullback_manifest.v1", core_commit=CORE,
            study_commit=a.commit, source_sha256=pins, train_sha256=m.TRAIN, plan=plan, base_sha256=m.MODEL,
            installed=identity, boundary_sha256=list(m.BOUNDARIES), original_files=len(paths), updates=0))
    base = st.NativeCausalLm.from_gpt2((a.model / "config.json").read_text(), st.SafeTensorArchive(a.model / "model.safetensors"))
    m.check(base.base_model_identity == m.BASE and base.execution_backend == "rust_f32_cpu_prepacked"
            and base.readout_accumulator == "f32", "wrong backend, model or accumulator")
    calls = 0
    with (root / "observations.jsonl").open("x") as stream:
        for (order, arm), e in entries.items():
            ck = (a.originals / order / (arm + ".ck")).read_bytes()
            model = st.NativeCausalLm.from_session_checkpoint_at(base, ck, e["checkpoint_id"], update=96,
                    recorded_data_id=e["data_id"], post_lora_id=e["owners"][0], post_geometry_id=e["owners"][1])
            for i, (ids, mask) in enumerate(zip(data["token_documents"], data["labels"], strict=True)):
                m.check(time.time() < started["started_epoch"] + LIMIT, "wall deadline")
                gradient = model.adaptation_loss_gradients(ids, [1,len(ids)], mode="joint", labels=mask, observe_geometry=True)
                obs = json.loads(gradient.observation_json())
                row = dict(order=order, arm=arm, index=i, key=plan[i][0], observation=obs)
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                calls += 1
                m.validate_row(row, order, arm, i, e, plan[i])
            m.check([model.lora_artifact_id,model.geometry_artifact_id] == e["owners"], "a frozen endpoint changed")
            print(f"{order}/{arm}: {calls}/864 windows; zero updates", flush=True)
    m.check(calls == 864 and not FOREIGN & {x.partition(".")[0] for x in sys.modules}, "wrong call count or foreign runtime")
    with (root / "observations.jsonl").open("rb") as stream:
        summary = m.summarize(iter(stream), plan)
    m.write(root / "result.json", dict(schema="spiraltorch.native_endpoint_pullback.v1", observations=calls, updates=0,
            unique_train_windows=96, eval_windows=0, finished_epoch=time.time(), cpu_seconds=time.process_time(),
            manifest_sha256=m.sha(root / "manifest.json"), journal_sha256=m.sha(root / "observations.jsonl"), orders=summary))


if __name__ == "__main__":
    os.umask(0o077)
    m.check(sys.flags.isolated and sys.dont_write_bytecode, "use python -I -B")
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("originals", "model", "wheel", "output"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    m.check(subprocess.check_output(["git","-C",str(m.ROOT),"rev-parse","HEAD"],text=True).strip() == a.commit, "wrong checkout")
    if a.worker:
        try:
            worker(a)
        except Exception as exc:
            m.write(a.output / "first-error.json", dict(type=type(exc).__name__, message=str(exc), at=time.time()))
            raise
    else:
        m.check(not a.output.resolve().is_relative_to(a.originals.resolve()), "refuse to write into source run")
        a.output.mkdir(mode=0o700, exist_ok=False)
        start = time.time()
        m.write(a.output / "attempt.json", dict(started_epoch=start, limit_seconds=LIMIT, commit=a.commit,
                core_commit=CORE, checkpoint_sets=list(m.BOUNDARIES), train_sha256=m.TRAIN, retries=0))
        with (a.output / "process.log").open("x") as log:
            outcome = run([sys.executable,"-I","-B",__file__,*sys.argv[1:],"--worker"],
                          timeout=LIMIT - (time.time()-start), stdout=log)
        m.write(a.output / "exit.json", dict(status=outcome.returncode, finished_epoch=time.time()))
        sys.exit(outcome.returncode)
