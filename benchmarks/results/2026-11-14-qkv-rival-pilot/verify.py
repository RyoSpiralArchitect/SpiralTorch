"""Read-only verifier for the frozen QKV pilot inputs and eventual receipts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol
import run


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory, manifest_pin):
    manifest_path = directory / "manifest.json"
    require(digest(manifest_path) == manifest_pin, "manifest pin mismatch")
    manifest = json.loads(manifest_path.read_bytes())
    rows, frozen = protocol.freeze()
    require(manifest["schema"] == "spiraltorch.gpt2_qkv_rival_inputs.v1"
            and manifest["protocol_sha256"] == protocol.pin(frozen)
            and manifest["runner_sha256"] == digest(Path(__file__).with_name("run.py"))
            and set(manifest["files_sha256"]) == {"train", "eval"},
            "protocol or runner identity mismatch")
    counts = {}
    for split in ("train", "eval"):
        path = directory / (split + ".json")
        require(digest(path) == manifest["files_sha256"][split],
                "encoded input pin mismatch: " + split)
        data = json.loads(path.read_bytes())
        require(set(data) == {"keys", "token_documents", "labels"}
                and data["keys"] == [row["key"] for row in rows[split]]
                and len(data["labels"]) == len(data["keys"])
                and all(len(ids) == len(mask) and 1 < len(ids) <= 64
                        and any(value != -100 for value in mask)
                        and all(value == -100 or value == token
                                for token, value in zip(ids, mask, strict=True))
                        for ids, mask in zip(data["token_documents"], data["labels"], strict=True)),
                "encoded input shape or mask mismatch: " + split)
        counts[split] = len(data["keys"])
    result_path = directory / "result.json"
    if not result_path.exists():
        status = ("failed_or_incomplete" if (directory / "fit-attempt.json").exists()
                  or (directory / "fit-failure.json").exists() else "prepared_only")
        return {"status": status, "counts": counts, "manifest_sha256": manifest_pin}
    require(not (directory / "fit-failure.json").exists(),
            "result and fit failure marker coexist")
    result = json.loads(result_path.read_bytes())
    require(result["schema"] == "spiraltorch.gpt2_qkv_rival_result.v1"
            and result["manifest_sha256"] == manifest_pin
            and result["arm_updates"] == 144
            and set(result["contrasts"]) == set(protocol.ORDERS),
            "result identity or coverage mismatch")
    expected_receipts = {"prepare-attempt.json", "fit-attempt.json", "manifest.json",
                         "train.json", "eval.json", "scores-000.json"}
    for order in protocol.ORDERS:
        expected_receipts.update(f"{order}/{name}" for name in (
            "steps.jsonl", "ordinary_ce.ck", "top_rival_ce.ck",
            "ordinary_ce-scores.json", "top_rival_ce-scores.json"))
    require(set(result["receipt_sha256"]) == expected_receipts,
            "result receipt inventory mismatch")
    for name, expected in result["receipt_sha256"].items():
        path = directory / name
        require(path.resolve().is_relative_to(directory.resolve())
                and digest(path) == expected, "result receipt mismatch: " + name)
    for order in protocol.ORDERS:
        ordinary = json.loads((directory / order / "ordinary_ce-scores.json").read_bytes())
        rival = json.loads((directory / order / "top_rival_ce-scores.json").read_bytes())
        require(run.contrast(ordinary, rival) == result["contrasts"][order],
                "stored contrast differs from scored answers: " + order)
    return {"status": "receipts_verified", "counts": counts,
            "manifest_sha256": manifest_pin, "result_sha256": digest(result_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--manifest-pin", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.manifest_pin), sort_keys=True))
