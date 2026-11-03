"""One bounded offline trial. Prepare fixes the encoded inputs; fit consumes their external pin once."""
import argparse
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as p  # noqa: E402

old = p.old
FROZEN_SHA = "7ba48e87df11570ac065e1e5251cfe23e8f91a33af6430e002b0d01c7685a7e2"
require, save, read, sha = old.require, old.save, old.read, old.sha


def source(commit):
    paths = [Path(__file__).resolve(), p.HERE / "protocol.py", p.PILOT,
             old.ROOT / "tools/byte_release_artifacts.py", old.ROOT / "tools/owned_process.py"]
    pins = {}
    for path in paths:
        name = str(path.relative_to(old.ROOT))
        committed = subprocess.check_output(["git", "-C", str(old.ROOT), "show", f"{commit}:{name}"])
        pins[name] = sha(path)
        require(pins[name] == old.digest(committed), "uncommitted execution source: " + name)
    return pins


def prepare(a):
    old.admit_time(a.deadline)
    st, identity = old.package(a)
    require(identity["native_sha256"] == p.NATIVE_SHA, "incorrect native binary")
    rows, frozen = p.freeze()
    require(p.pin(frozen) == FROZEN_SHA, "record/condition freeze changed")
    require(sha(a.model_dir / "tokenizer.json") == old.FILES["tokenizer.json"], "unexpected tokenizer bytes")
    spec = importlib.machinery.PathFinder.find_spec("tokenizers", [str(a.tokenizers_site)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tokenizers"] = mod
    spec.loader.exec_module(mod)
    require(mod.__version__ == "0.22.2", "unexpected tokenizer package")
    tokenizer = mod.Tokenizer.from_file(str(a.model_dir / "tokenizer.json"))
    files, stats, seen = {}, {}, set()
    for split, records in rows.items():
        docs, labels = [], []
        for r in records:
            prompt, answer = r["prompt"], r["answer"]
            text = prompt + answer
            ids, mask = st.encode_supervised_document(tokenizer, text, [(len(prompt), len(text))])
            boundary = next(i for i, v in enumerate(mask) if v != -100)
            require(0 < boundary < len(ids) <= p.CONTEXT and mask == [-100] * boundary + ids[boundary:]
                    and tokenizer.decode(ids[:boundary], skip_special_tokens=False) == prompt
                    and tokenizer.decode(ids[boundary:], skip_special_tokens=False) == answer,
                    "answer mask does not round-trip")
            require(tuple(ids) not in seen, "duplicate encoded record")
            seen.add(tuple(ids))
            docs.append(ids)
            labels.append(mask)
        payload = dict(keys=[r["key"] for r in records], token_documents=docs, labels=labels)
        save(a.output / (split + ".json"), payload)
        files[split] = sha(a.output / (split + ".json"))
        stats[split] = dict(documents=len(docs), target_tokens=sum(x != -100 for row in labels for x in row),
                            tokens=sum(map(len, docs)), longest=max(map(len, docs)))
    require(not old.FORBIDDEN & {n.partition(".")[0] for n in sys.modules}, "foreign ML runtime loaded")
    save(a.output / "manifest.json", dict(schema="spiraltorch.field_control_inputs.v1", commit=a.commit,
         source_sha256=source(a.commit), frozen=frozen, files=files, stats=stats, installed=identity,
         tokenizer_version=mod.__version__, attempt_sha256=sha(a.output / "prepare-attempt.json"),
         deadline_epoch=a.deadline, prep_cpu_seconds=time.process_time()))
    print(json.dumps(dict(manifest_sha256=sha(a.output / "manifest.json"), stats=stats)), flush=True)


def models_for(st, base, width, layer):
    lora = base.with_lora(layer, p.RANK, p.ALPHA)
    tensors = json.loads(lora.lora_to_json())
    a, b = tensors["a"], tensors["b"]
    require(len(a) == len(b) == width * p.RANK and any(a) and not any(b), "incorrect LoRA origin")
    states = [p.geometry(st, base.base_model_identity, arm, width=width, layer=layer) for arm in p.ARMS]
    counts = [p.parameter_state(s)[1] + len(a) + len(b) for s in states]
    require(counts == [6 * width, 6 * width, 6 * width + 1], "incorrect trainable capacity")
    return [lora.with_geometry(s) for s in states], counts


def score(st, models, data, context, layer):
    n = len(data["labels"])
    r = st.NativeCausalLm.evaluate_variants_by_answer(models, data["token_documents"], context,
                                                       labels=data["labels"], microbatch_windows=2)
    targets = sum(v != -100 for row in data["labels"] for v in row)
    require(r.shared_prefix_layers == layer and (len(r), r.scored_documents, r.document_count) == (n, n, n)
            and all(v.target_tokens == targets and v.window_count == n for v in r.evaluations), "scoring coverage")
    scores = json.loads(r.to_json())
    for i, (row, mask) in enumerate(zip(scores["answers"], data["labels"], strict=True)):
        require(row["document"] == i and row["end"] == len(mask) and mask[:row["start"]] == [-100] * row["start"]
                and -100 not in mask[row["start"]:] and row["start"] < row["end"], "scored different answer span")
    return dict(scores=scores, contrasts={k: r.contrast(*pair) for k, pair in p.PAIRS.items()})


def one_order(st, base, origin, train, indices, location, deadline, context, layer):
    location.mkdir(mode=0o700, exist_ok=False)
    n = len(train["labels"])
    require(sorted(indices) == list(range(n)), "training order is not a permutation")
    docs, labels = ([train[k][i] for i in indices] for k in ("token_documents", "labels"))
    Session = st.NativeCausalLmTrainingSession
    first = Session(origin[0], docs, context, labels=labels, mode="joint", window_policy="covering_targets",
                    batch_windows=1, epochs=1, learning_rate=p.LR)
    cohort = [first] + [first.fork_variant(m, "joint") for m in origin[1:]]
    require(all(s.completed_updates == 0 and s.total_updates == n and s.data_id == first.data_id
                and owners(s.model) == owners(m) for s, m in zip(cohort, origin, strict=True)), "different start")
    save(location / "order.json", dict(keys=[train["keys"][i] for i in indices], data_id=first.data_id))
    previous = [owners(m) for m in origin]
    with (location / "steps.jsonl").open("x", encoding="utf8") as journal:
        for i, mask in enumerate(labels, 1):
            require(time.time() < deadline, "expired before training step")
            depth, reports = Session.step_matched(cohort)
            raw = [json.loads(r.to_json()) for r in reports]
            journal.write(json.dumps(dict(depth=depth, reports=raw), allow_nan=False, separators=(",", ":")) + "\n")
            journal.flush()
            require(depth == layer and len(raw) == 3, "unmatched frozen prefix")
            for j, r in enumerate(raw):
                require(r["epoch"] == 0 and r["mode"] == "joint" and r["update"] == i
                        and [r["window_start"], r["window_end"]] == [i - 1, i]
                        and r["target_tokens"] == sum(x != -100 for x in mask) and math.isfinite(r["loss"])
                        and [r["pre_lora_artifact_id"], r["pre_geometry_artifact_id"]] == previous[j],
                        "step, target or lineage mismatch")
                previous[j] = [r["post_lora_artifact_id"], r["post_geometry_artifact_id"]]
            if i == 1:
                require(len({r["loss"] for r in raw}) == len({o[0] for o in previous}) == 1,
                        "different initial loss or LoRA update")
                save(location / "first-step.json", p.first_step([s.model.geometry_artifact() for s in cohort]))
    restored, checkpoints = [], []
    require(all(s.exhausted for s in cohort), "missing updates")
    for arm, session, ids in zip(p.ARMS, cohort, previous, strict=True):
        require(owners(session.model) == ids, "checkpoint owner mismatch")
        file = location / (arm + ".ck")
        ck = session.save_checkpoint_new(file)
        require(ck == "sha256:" + sha(file), "checkpoint pin mismatch")
        model = st.NativeCausalLm.from_session_checkpoint_at(base, file.read_bytes(), ck, update=n,
                recorded_data_id=first.data_id, post_lora_id=ids[0], post_geometry_id=ids[1])
        require(owners(model) == ids, "restored owners differ")
        state, count = p.parameter_state(model.geometry_artifact())
        radius = state.get("log_radius", [None])[0]
        if arm == "wave_fixed":
            require(radius == 0., "frozen radius moved")
        checkpoints.append(dict(arm=arm, checkpoint_id=ck, owners=ids, update=n, data_id=first.data_id,
                                geometry_parameters=count, log_radius=radius))
        restored.append(model)
    save(location / "checkpoints.json", checkpoints)
    return restored, first.data_id


def owners(model):
    return [model.lora_artifact_id, model.geometry_artifact_id]


def fit(a):
    root = a.output
    require(sha(root / "manifest.json") == a.manifest_pin, "manifest not pinned")
    m = read(root / "manifest.json")
    cpu = old.admit_time(a.deadline, m["prep_cpu_seconds"])
    save(root / "fit-attempt.json", dict(manifest_sha256=a.manifest_pin, deadline_epoch=a.deadline,
                                         started_epoch=time.time(), cpu_limit_seconds=cpu, no_retry=True))
    records, frozen = p.freeze()
    require(p.pin(frozen) == FROZEN_SHA and frozen == m["frozen"] and m["commit"] == a.commit
            and m["source_sha256"] == source(a.commit) and m["deadline_epoch"] == a.deadline
            and sha(root / "prepare-attempt.json") == m["attempt_sha256"], "protocol changed")
    st, installed = old.package(a)
    require(installed == m["installed"] and installed["native_sha256"] == p.NATIVE_SHA
            and all(sha(a.model_dir / n) == h for n, h in old.FILES.items()), "installed bytes/base changed")
    data = {}
    for k in p.SPLITS:
        path = root / (k + ".json")
        require(sha(path) == m["files"][k], "encoded split changed")
        data[k] = read(path)
        require(data[k]["keys"] == [r["key"] for r in records[k]], "record-to-token mapping changed")
    base = st.NativeCausalLm.from_gpt2((a.model_dir / "config.json").read_text(),
                                       st.SafeTensorArchive(a.model_dir / "model.safetensors"))
    require(base.base_model_identity == old.BASE_ID and base.readout_accumulator == "f32"
            and base.execution_backend == "rust_f32_cpu_prepacked", "wrong base, arithmetic or backend")
    models, counts = models_for(st, base, p.WIDTH, p.LAYER)
    save(root / "initial-owners.json", dict(owners=[owners(m) for m in models], counts=counts))
    initial = score(st, models, data["eval"], p.CONTEXT, p.LAYER)
    save(root / "scores-000.json", initial)
    require(all(c["tied"] == 48 for c in initial["contrasts"].values())
            and all(row["scores"] == [row["scores"][0]] * 3 for row in initial["scores"]["answers"]),
            "unequal initial answer scores")
    eval_id = initial["scores"]["comparison"]["evaluations"][0]["data_id"]
    contrasts, ids = {}, set()
    for order in p.ORDERS:
        loc = root / order
        indices = p.train_order(records["train"], order)
        restored, data_id = one_order(st, base, models, data["train"], indices, loc, a.deadline, p.CONTEXT, p.LAYER)
        require(data_id not in ids, "reused the same order")
        ids.add(data_id)
        final = score(st, restored, data["eval"], p.CONTEXT, p.LAYER)
        save(loc / "scores-096.json", final)
        require(all(e["data_id"] == eval_id for e in final["scores"]["comparison"]["evaluations"]), "evaluation changed")
        contrasts[order] = final["contrasts"]
        print(order + ": " + json.dumps(final["contrasts"], allow_nan=False), flush=True)
    require(not (old.FORBIDDEN | {"tokenizers"}) & {n.partition(".")[0] for n in sys.modules}, "foreign runtime")
    names = ["prepare-attempt.json", "fit-attempt.json", "initial-owners.json", "scores-000.json"] + [
             f"{order}/{file}" for order in p.ORDERS
             for file in ("order.json", "steps.jsonl", "first-step.json", "checkpoints.json", "scores-096.json")]
    result = dict(manifest_sha256=a.manifest_pin, source_commit=a.commit, receipt_sha256={n: sha(root / n) for n in names},
                  arm_updates=3 * 3 * 96, contrasts=contrasts, unique_eval_answers=48, distinct_orders=3,
                  decision=p.classify([c["fixed_minus_affine"] for c in contrasts.values()]),
                  train_cpu_seconds=time.process_time(), finished_epoch=time.time())
    require(result["finished_epoch"] < a.deadline and result["train_cpu_seconds"] + m["prep_cpu_seconds"] < 3600,
            "budget exceeded")
    save(root / "result.json", result)
    print(result["decision"], flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    require(sys.flags.isolated and sys.dont_write_bytecode, "require python -I -B")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "fit"))
    for key in ("output", "wheel", "model-dir"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--deadline", type=int, required=True)
    ap.add_argument("--tokenizers-site", type=Path)
    ap.add_argument("--manifest-pin")
    ap.add_argument("--owned-worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.wheel_pin = p.WHEEL_SHA
    require(subprocess.check_output(["git", "-C", str(old.ROOT), "rev-parse", "HEAD"], text=True).strip() == a.commit,
            "checkout differs from fixed commit")
    require((a.action == "prepare" and a.tokenizers_site and not a.manifest_pin)
            or (a.action == "fit" and not a.tokenizers_site and a.manifest_pin), "incorrect phase inputs")
    if not a.owned_worker:
        from owned_process import run
        wall = a.deadline - time.time()
        require(0 < wall <= 3601, "invalid absolute deadline")
        sys.exit(run([sys.executable, "-I", "-B", __file__, *sys.argv[1:], "--owned-worker"],
                     timeout=min(wall, 3600), stdout=sys.stdout).returncode)
    if a.action == "prepare":
        a.output.mkdir(mode=0o700, exist_ok=False)
        save(a.output / "prepare-attempt.json", dict(started_epoch=time.time(), deadline_epoch=a.deadline,
                                                     source_commit=a.commit, no_retry=True))
    try:
        (prepare if a.action == "prepare" else fit)(a)
    except Exception as exc:
        save(a.output / (a.action + "-failure.json"), dict(type=type(exc).__name__, message=str(exc), epoch=time.time()))
        raise
