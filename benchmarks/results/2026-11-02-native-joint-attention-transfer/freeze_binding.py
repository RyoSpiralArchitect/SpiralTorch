"""Bind the retained primary run to the committed freeze, independently of its scorer."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from joint_protocol import BOOKS, PLAN, read, require, sha
from joint_worker import paths, wheels

FREEZE_SHA256 = "8f8b9a0c950bd3e58dc13279418fe073743be6c4f9c5c75c5656cce04281a601"


def frozen():
    path = HERE / "execution.json"
    require(
        sha(path) == FREEZE_SHA256, "not the prospectively committed execution freeze"
    )
    return read(path)


def section_identity(docs):
    # Independent wire calculation for this study's one fixed shape only.
    # Field order is Rust's serialized CausalDocumentWindows, not sorted JSON.
    require(
        len(docs) == 12
        and all(
            len(d) == 64 and all(type(v) is int and 0 <= v < 50257 for v in d)
            for d in docs
        ),
        "unexpected section shape/ID",
    )
    plan = {
        "schema": "spiraltorch.causal_document_windows.v1",
        "policy": "overlap_one_token_include_tail_per_document.v1",
        "context_length": 64,
        "document_lengths": [64] * 12,
        "windows": [{"document": i, "start": 0, "length": 64} for i in range(12)],
        "total_tokens": 768,
        "dropped_tokens": 0,
        "target_tokens": 756,
    }
    pid = (
        "sha256:"
        + hashlib.sha256(json.dumps(plan, separators=(",", ":")).encode()).hexdigest()
    )
    h = hashlib.sha256(
        b"spiraltorch.native_causal_evaluation_data.v1\0" + pid.encode() + b"\0"
    )
    for doc in docs:
        h.update(struct.pack("<65Q", 64, *doc))
    return {"data_id": "sha256:" + h.hexdigest(), "plan_id": pid}


def bind_inputs(a):
    f = frozen()
    require(
        sha(HERE / "plan.json") == f["plan_sha256"]
        and sha(a.data / "manifest.json") == f["data_manifest_sha256"]
        and sha(a.data / "windows.json") == f["held_out_windows_sha256"]
        and sha(a.data / "prompts.json") == f["prompt_ids_sha256"]
        and sha(a.old / "data" / "train.json") == PLAN["train_sha256"],
        "inputs differ from committed freeze",
    )
    inv = wheels.wheel_archive_inventory(a.wheel)
    require(
        inv["sha256"] == f["development_wheel_sha256"]
        and inv["native"]["sha256"] == f["native_sha256"],
        "engine differs from committed freeze",
    )
    groups = read(a.data / "windows.json")
    require(set(groups) == set(BOOKS), "book roster")
    ids = {}
    for book, keys in f["sections"].items():
        require(sorted(map(int, groups[book])) == keys, "chapter roster")
        ids.update(
            {f"{book}-{k}": section_identity(groups[book][str(k)]) for k in keys}
        )
    require(
        len({i["data_id"] for i in ids.values()}) == 18, "duplicate section identity"
    )
    return {
        "schema": "spiraltorch.native_lm.joint_attention_input_binding.v1",
        "execution_sha256": FREEZE_SHA256,
        "windows_sha256": f["held_out_windows_sha256"],
        "sections": ids,
    }


def match_rows(expected, rows, count):
    require(
        len(rows) == count
        and all(
            (v["plan_id"], v["data_id"]) == (expected["plan_id"], expected["data_id"])
            for v in rows
        ),
        "score belongs to another chapter or document plan",
    )


def bind_run(a, binding):
    f = frozen()
    d = read(a.out / "declaration.json")
    require(
        (d["plan_sha256"], d["data_sha256"], d["wheel_sha256"], d["native_sha256"])
        == (
            f["plan_sha256"],
            f["data_manifest_sha256"],
            f["development_wheel_sha256"],
            f["native_sha256"],
        ),
        "retained run differs from committed inputs",
    )
    scores = read(a.out / "scored.json")
    require(
        set(scores["sections"]) == set(binding["sections"]), "scored chapter roster"
    )
    for k, identity in binding["sections"].items():
        path = a.out / "scores" / (k + ".json")
        require(sha(path) == scores["sections"][k], "changed chapter receipt")
        match_rows(identity, read(path)["evaluations"], 13)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    paths(p)
    p.add_argument("action", choices=["bind", "run", "audit"])
    p.add_argument("--python", type=Path)
    a = p.parse_args()
    binding = bind_inputs(a)
    if a.action == "bind":
        print(json.dumps(binding, sort_keys=True, separators=(",", ":")))
    else:
        import joint_study

        require(
            binding == read(HERE / "input_binding.json"),
            "published section mapping differs",
        )
        if a.action == "run":
            require(a.python is not None, "a pinned study interpreter is required")
            f = frozen()
            a.plan_pin, a.data_pin, a.wheel_pin, a.native_pin = (
                f[key]
                for key in (
                    "plan_sha256",
                    "data_manifest_sha256",
                    "development_wheel_sha256",
                    "native_sha256",
                )
            )
            joint_study.run(a)
        bind_run(a, binding)
        r = joint_study.audit(a)
        require(
            r == read(a.out / "results.json")
            and sha(a.out / "results.json")
            == read(a.out / "status.json")["results_sha256"],
            "primary result mismatch",
        )
        print(
            "frozen wheel, corpus, prompts, 18 chapter identities and primary result verified"
        )
