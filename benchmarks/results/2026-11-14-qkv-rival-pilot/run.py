"""One-attempt, offline GPT-2 QKV objective comparison; prepare before fit."""

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol
sys.path.insert(0, str(protocol.HERE.parents[2] / "tools"))
from owned_process import run as run_owned


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read(path):
    return json.loads(path.read_bytes())


def inputs(args):
    pilot, _ = protocol._predecessors()
    paths = {
        "config.json": args.model_dir / "config.json",
        "model.safetensors": args.model_dir / "model.safetensors",
        "tokenizer.json": args.model_dir / "tokenizer.json",
        "frozen_wave_lora-096.ck": args.source_dir / "frozen_wave_lora-096.ck",
        "frozen_wave_lora-096-boundary.json": args.source_dir / "frozen_wave_lora-096-boundary.json",
    }
    for name, path in paths.items():
        require(sha(path) == protocol.SOURCE_SHA256[name], "source pin mismatch: " + name)
    return pilot, paths


def package(pilot, args):
    args.wheel_pin = args.wheel_pin.lower()
    require(len(args.wheel_pin) == 64 and sha(args.wheel) == args.wheel_pin,
            "wheel pin mismatch")
    return pilot.package(args)


def prepare(args):
    args.output.mkdir(mode=0o700, exist_ok=False)
    save(args.output / "prepare-attempt.json", {
        "started_epoch": time.time(), "deadline_epoch": args.deadline, "no_retry": True,
    })
    pilot, paths = inputs(args)
    pilot.admit_time(args.deadline)
    st, installed = package(pilot, args)
    spec = importlib.machinery.PathFinder.find_spec("tokenizers", [str(args.tokenizers_site)])
    require(spec is not None, "pinned tokenizer installation missing")
    codec = importlib.util.module_from_spec(spec)
    sys.modules["tokenizers"] = codec
    spec.loader.exec_module(codec)
    require(codec.__version__ == "0.22.2", "tokenizer version mismatch")
    tokenizer = codec.Tokenizer.from_file(str(paths["tokenizer.json"]))
    splits, frozen = protocol.freeze()
    files = {}
    for name, rows in splits.items():
        docs, labels = [], []
        for row in rows:
            full = row["prompt"] + row["answer"]
            ids, mask = st.encode_supervised_document(
                tokenizer, full, [(len(row["prompt"]), len(full))]
            )
            start = next((i for i, value in enumerate(mask) if value != -100), None)
            require(start is not None and 0 < start < len(ids) <= 64
                    and mask == [-100] * start + ids[start:]
                    and tokenizer.decode(ids[:start], skip_special_tokens=False) == row["prompt"]
                    and tokenizer.decode(ids[start:], skip_special_tokens=False) == row["answer"],
                    "answer mask or tokenizer round trip changed")
            docs.append(ids)
            labels.append(mask)
        require(len({tuple(doc) for doc in docs}) == len(rows), "duplicate encoded record")
        path = args.output / (name + ".json")
        save(path, {"keys": [row["key"] for row in rows],
                    "token_documents": docs, "labels": labels})
        files[name] = sha(path)
    require(time.time() < args.deadline, "deadline expired before input manifest")
    save(args.output / "manifest.json", {
        "schema": "spiraltorch.gpt2_qkv_rival_inputs.v1",
        "protocol_sha256": protocol.pin(frozen),
        "runner_sha256": sha(__file__),
        "wheel_sha256": args.wheel_pin,
        "installed": installed,
        "files_sha256": files,
        "tokenizers_version": codec.__version__,
        "prepare_deadline_epoch": args.deadline,
        "prepare_cpu_seconds": time.process_time(),
    })


def scored(optimizer, source, data):
    result = optimizer.evaluate_against_source_by_answer(
        source, data["token_documents"], 64,
        labels=data["labels"], microbatch_windows=1,
    )
    require(result.document_count == result.scored_documents == len(data["keys"]),
            "answer coverage mismatch")
    raw = json.loads(result.to_json())
    require(len(raw["answers"]) == len(data["keys"])
            and all(row["document"] == i and len(row["scores"]) == 2
                    for i, row in enumerate(raw["answers"])),
            "answer mapping mismatch")
    return raw


def contrast(ordinary, rival):
    a, b = ordinary["answers"], rival["answers"]
    require(len(a) == len(b) == 48, "evaluation size changed")
    nll_delta = 0.0
    answer_nll_delta = 0.0
    tokens = 0
    out = {"ordinary_all_top1": 0, "rival_all_top1": 0,
           "ordinary_token_top1": 0, "rival_token_top1": 0,
           "improved_nll": 0, "tied_nll": 0, "regressed_nll": 0}
    for x, y in zip(a, b, strict=True):
        require((x["document"], x["start"], x["end"]) ==
                (y["document"], y["start"], y["end"]) and
                x["scores"][0] == y["scores"][0], "unmatched source score")
        count = x["end"] - x["start"]
        left, right = x["scores"][1], y["scores"][1]
        delta = right["nll"] - left["nll"]
        require(math.isfinite(delta), "nonfinite answer score")
        nll_delta += delta * count
        answer_nll_delta += delta
        tokens += count
        out["ordinary_all_top1"] += left["top1_correct"] == count
        out["rival_all_top1"] += right["top1_correct"] == count
        out["ordinary_token_top1"] += left["top1_correct"]
        out["rival_token_top1"] += right["top1_correct"]
        out["improved_nll" if delta < 0 else "regressed_nll" if delta > 0 else "tied_nll"] += 1
    out["all_top1_delta"] = out["rival_all_top1"] - out["ordinary_all_top1"]
    out["token_weighted_delta_nll"] = nll_delta / tokens
    out["answer_weighted_delta_nll"] = answer_nll_delta / len(a)
    return out


def fit(args):
    manifest_path = args.output / "manifest.json"
    require(sha(manifest_path) == args.manifest_pin, "manifest pin mismatch")
    manifest = read(manifest_path)
    splits, frozen = protocol.freeze()
    require(manifest["protocol_sha256"] == protocol.pin(frozen)
            and manifest["runner_sha256"] == sha(__file__)
            and manifest["wheel_sha256"] == args.wheel_pin
            and set(manifest["files_sha256"]) == {"train", "eval"}
            and manifest["prepare_deadline_epoch"] <= args.deadline,
            "frozen protocol, runner, wheel or deadline changed")
    save(args.output / "fit-attempt.json", {
        "started_epoch": time.time(), "manifest_sha256": args.manifest_pin,
        "deadline_epoch": args.deadline, "no_retry": True,
    })
    pilot, paths = inputs(args)
    pilot.admit_time(args.deadline, manifest["prepare_cpu_seconds"])
    st, installed = package(pilot, args)
    require(installed == manifest["installed"], "installed wheel changed")
    data = {}
    for split, expected in manifest["files_sha256"].items():
        path = args.output / (split + ".json")
        require(sha(path) == expected, "encoded input changed: " + split)
        data[split] = read(path)
        require(data[split]["keys"] == [r["key"] for r in splits[split]],
                "record-to-token map changed")
    base = st.NativeCausalLm.from_gpt2(
        paths["config.json"].read_text(), st.SafeTensorArchive(paths["model.safetensors"])
    ).with_readout_accumulator("f32")
    from spiraltorch.native_session_io import read_pinned_session
    payload, boundary = read_pinned_session(
        paths["frozen_wave_lora-096.ck"],
        "sha256:" + protocol.SOURCE_SHA256["frozen_wave_lora-096.ck"],
        paths["frozen_wave_lora-096-boundary.json"],
    )
    source = st.NativeCausalLm.from_session_checkpoint_at(
        base, payload, "sha256:" + protocol.SOURCE_SHA256["frozen_wave_lora-096.ck"],
        **boundary,
    )
    require(source.readout_accumulator == "f32" and
            source.execution_backend == "rust_f32_cpu_prepacked", "unexpected source backend")
    initial = scored(st.NativeCausalLmQkvAdamW(source, 11, max_gradient_norm=1.0),
                     source, data["eval"])
    require(all(row["scores"][0] == row["scores"][1] for row in initial["answers"]),
            "QKV arms do not start at source")
    save(args.output / "scores-000.json", initial)
    results = {}
    for order in protocol.ORDERS:
        indices = protocol.order_indices(splits["train"], order)
        optimizers = {
            arm: st.NativeCausalLmQkvAdamW(source, 11, max_gradient_norm=1.0)
            for arm in protocol.RECIPE["arms"]
        }
        order_dir = args.output / order
        order_dir.mkdir(mode=0o700, exist_ok=False)
        with (order_dir / "steps.jsonl").open("x", encoding="utf-8") as journal:
            for step, index in enumerate(indices, 1):
                require(time.time() < args.deadline, "deadline expired before update")
                ids = data["train"]["token_documents"][index]
                labels = data["train"]["labels"][index]
                shared = dict(labels=labels, learning_rate=0.0001, weight_decay=0.01)
                ce = optimizers["ordinary_ce"].step_windows(ids, [1, len(ids)], **shared)
                rival = optimizers["top_rival_ce"].step_rival_windows(
                    ids, [1, len(ids)], rivals=4, mix=0.5, **shared)
                target_count = sum(value != -100 for value in labels)
                require(ce["steps"] == rival["steps"] == step
                        and ce["target_tokens"] == rival["competition"]["target_tokens"] == target_count,
                        "step count or supervised coverage changed")
                journal.write(json.dumps({"step": step, "key": splits["train"][index]["key"],
                                          "ordinary_ce": ce, "top_rival_ce": rival},
                                         sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                journal.flush()
            os.fsync(journal.fileno())
        scores = {}
        for arm, optimizer in optimizers.items():
            require(optimizer.steps == 24, "incomplete QKV training")
            checkpoint = order_dir / (arm + ".ck")
            owner = optimizer.save_checkpoint_new(checkpoint)
            require(owner == "sha256:" + sha(checkpoint), "QKV checkpoint pin mismatch")
            score = scored(optimizer, source, data["eval"])
            save(order_dir / (arm + "-scores.json"), score)
            scores[arm] = score
        results[order] = contrast(scores["ordinary_ce"], scores["top_rival_ce"])
    receipt_paths = [args.output / "prepare-attempt.json", args.output / "fit-attempt.json",
                     args.output / "manifest.json", args.output / "train.json",
                     args.output / "eval.json", args.output / "scores-000.json"]
    for order in protocol.ORDERS:
        receipt_paths.extend([args.output / order / name for name in (
            "steps.jsonl", "ordinary_ce.ck", "top_rival_ce.ck",
            "ordinary_ce-scores.json", "top_rival_ce-scores.json")])
    require(time.time() < args.deadline, "deadline expired before final receipt")
    save(args.output / "result.json", {
        "schema": "spiraltorch.gpt2_qkv_rival_result.v1",
        "manifest_sha256": args.manifest_pin,
        "contrasts": results,
        "arm_updates": 144,
        "receipt_sha256": {str(path.relative_to(args.output)): sha(path) for path in receipt_paths},
        "interpretation_limit": frozen["interpretation_limit"],
        "finished_epoch": time.time(),
    })


def main():
    os.umask(0o077)
    require(sys.flags.isolated and sys.dont_write_bytecode, "require python -I -B")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "fit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--wheel-pin", required=True)
    parser.add_argument("--deadline", type=int, required=True)
    parser.add_argument("--tokenizers-site", type=Path)
    parser.add_argument("--manifest-pin")
    parser.add_argument("--owned-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require((args.action == "prepare" and args.tokenizers_site and not args.manifest_pin) or
            (args.action == "fit" and args.manifest_pin and not args.tokenizers_site),
            "phase arguments mismatch")
    if not args.owned_worker:
        remaining = args.deadline - time.time()
        require(0 < remaining <= 3601, "invalid or expired one-hour deadline")
        try:
            outcome = run_owned([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                                 *sys.argv[1:], "--owned-worker"],
                                timeout=min(remaining, 3600), stdout=sys.stdout)
        except Exception as exc:
            if args.output.is_dir():
                path = args.output / (args.action + "-failure.json")
                if not path.exists():
                    save(path, {"type": type(exc).__name__, "message": str(exc),
                                "epoch": time.time()})
            raise
        if outcome.returncode != 0 and args.output.is_dir():
            path = args.output / (args.action + "-failure.json")
            if not path.exists():
                save(path, {"type": "WorkerExit", "returncode": outcome.returncode,
                            "epoch": time.time()})
        raise SystemExit(outcome.returncode)
    try:
        (prepare if args.action == "prepare" else fit)(args)
    except Exception as exc:
        if args.output.is_dir():
            path = args.output / (args.action + "-failure.json")
            if not path.exists():
                save(path, {"type": type(exc).__name__, "message": str(exc), "epoch": time.time()})
        raise


if __name__ == "__main__":
    main()
