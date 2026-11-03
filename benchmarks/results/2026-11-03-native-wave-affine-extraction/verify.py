"""Independent, file-only read-back of the matched trial. Never loads a model or rescores an answer."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as p  # noqa: E402
from run import FROZEN_SHA  # noqa: E402

check, sha, read = p.old.require, p.old.sha, p.old.read
TOP = ("prepare-attempt.json", "fit-attempt.json", "initial-owners.json", "scores-000.json")
PER = ("order.json", "steps.jsonl", "first-step.json", "checkpoints.json", "scores-096.json")
NAMES = (*TOP, *(f"{order}/{file}" for order in p.ORDERS for file in PER))


def close(x, y):
    return math.isclose(x, y, rel_tol=2e-12, abs_tol=2e-12)


def endpoint(receipt, owners, eval_stats, masks=None):
    raw = receipt["scores"]
    answers, evaluations = raw["answers"], raw["comparison"]["evaluations"]
    check(raw["scored_documents"] == raw["document_count"] == len(answers) == 48 and len(evaluations) == 3
          and raw["comparison"]["shared_prefix_layers"] == 11, "scoring roster")
    lengths = []
    for i, row in enumerate(answers):
        a, b = row["start"], row["end"]
        check(row["document"] == i and 0 < a < b <= p.CONTEXT and len(row["scores"]) == 3, "answer index/span")
        if masks is not None:
            check(b == len(masks[i]) and masks[i][:a] == [-100] * a and -100 not in masks[i][a:], "answer mask")
        for s in row["scores"]:
            check(math.isfinite(s["nll"]) and s["nll"] >= 0 and 0 <= s["top1_correct"] <= b - a, "answer score")
        lengths.append(b - a)
    targets = sum(lengths)
    check(targets == eval_stats["target_tokens"], "target denominator")
    eval_ids = set()
    for j, (ev, owner) in enumerate(zip(evaluations, owners, strict=True)):
        micro = math.fsum(n * row["scores"][j]["nll"] for n, row in zip(lengths, answers)) / targets
        check([ev["lora_artifact_id"], ev["geometry_artifact_id"]] == owner and ev["base_model_identity"] == p.old.BASE_ID
              and ev["readout_accumulator"] == "f32" and ev["target_tokens"] == targets
              and ev["config"] == dict(context_length=64, microbatch_windows=2)
              and ev["window_count"] == 48 and ev["skipped_windows"] == 0
              and math.isclose(micro, ev["nll"], rel_tol=1e-6, abs_tol=1e-7), "scored owner/plan/NLL")
        eval_ids.add((ev["data_id"], ev["plan_id"]))
    check(len(eval_ids) == 1 and set(receipt["contrasts"]) == set(p.PAIRS), "unpaired evaluation")
    for key, (a, b) in p.PAIRS.items():
        c = receipt["contrasts"][key]
        pairs = [(r["scores"][a], r["scores"][b]) for r in answers]
        ds = [y["nll"] - x["nll"] for x, y in pairs]
        check((c["reference"], c["candidate"], c["answer_count"], c["target_tokens"]) == (a, b, 48, targets)
              and close(c["answer_weighted_delta_nll"], math.fsum(ds) / 48)
              and close(c["token_weighted_delta_nll"], math.fsum(n * d for n, d in zip(lengths, ds)) / targets)
              and [c[t] for t in ("improved", "tied", "regressed")] ==
              [sum(d < 0 for d in ds), sum(d == 0 for d in ds), sum(d > 0 for d in ds)], "paired reduction")
        for k, which in enumerate(("reference", "candidate")):
            hits = [pair[k]["top1_correct"] for pair in pairs]
            check(c[which + "_top1_correct"] == sum(hits) and c[which + "_all_correct_answers"] ==
                  sum(h == n for h, n in zip(hits, lengths)), "top-1 counts")
    return eval_ids.pop()


def checkpoint(path, entry):
    ck = read(path)
    ids = [ck["lora_id"], ck["geometry_id"]]
    check(entry["checkpoint_id"] == "sha256:" + sha(path) and ids == entry["owners"]
          and entry["data_id"] == ck["data_id"] and ck["updates"] == 96 and ck["config"] ==
          dict(context_length=64, epochs=1, learning_rate=.01, mode="joint")
          and ck["data_policy"] == dict(has_labels=True, window_policy="covering_targets")
          and ck["base_model_identity"] == p.old.BASE_ID and ck["readout_accumulator"] == "f32", "checkpoint bytes")
    lo, geo = ck["lora_json"], ck["geometry_json"]
    check(ids == ["sha256:" + p.old.digest(b"spiraltorch.native_lora.v1\0" + lo.encode()),
                  "sha256:" + p.old.digest(geo.encode())], "inner adapter digest")
    lora, record = json.loads(lo), json.loads(geo)
    check(lora["base_model_identity"] == record["base_model_identity"] == p.old.BASE_ID and lora["layer"] == 11
          and lora["width"] == 768 and lora["rank"] == 2 and lora["alpha"] == 4
          and len(lora["a"]) == len(lora["b"]) == 1536 and len(record["placements"]) == 1, "adapter placement")
    site = record["placements"][0]
    arm, recipe = entry["arm"], site["recipe"]
    check(record["training"] and site["training"] and site["path"] == "transformer.h.11.attn"
          and recipe["strength"] == .25 and recipe["features"] == 768, "geometry placement or gain")
    if arm == "affine":
        check(recipe["schema"] == "spiraltorch.channel_affine_adapter.v1" and recipe["max_values"] == 65536,
              "not ordinary affine")
    else:
        check(recipe["schema"] == "spiraltorch.wave_gate_adapter.v2" and recipe["learnable_radius"] == (arm == "wave_learned")
              and recipe["kernel"] == dict(curvature=-1., saturation=1., porosity=.05, max_values=65536), "wave recipe")
    tensors = {t["name"]: t for t in site["tensors"]}
    check(set(tensors) == ({"gate", "bias"} if arm == "affine" else {"gate", "bias", "log_radius"}), "geometry roster")
    for name, t in tensors.items():
        trainable = name != "log_radius" or arm == "wave_learned"
        count = 1 if name == "log_radius" else 768
        check(t["requires_grad"] == trainable and t["role"] == ("parameter" if trainable else "buffer")
              and t["shape"] == ([] if count == 1 else [count]) and len(t["values"]) == count
              and all(map(math.isfinite, t["values"])), "geometry ownership/shape")
    radius = tensors["log_radius"]["values"][0] if arm != "affine" else None
    check(radius == entry["log_radius"] and (arm != "wave_fixed" or radius == 0.), "radius value")


def verify(root, originals=False):
    manifest, result = (read(root / n) for n in ("manifest.json", "result.json"))
    records, frozen = p.freeze()
    check(sha(root / "manifest.json") == result["manifest_sha256"] and p.pin(frozen) == FROZEN_SHA
          and frozen == manifest["frozen"] and manifest["commit"] == result["source_commit"]
          and manifest["installed"]["wheel_sha256"] == p.WHEEL_SHA and manifest["installed"]["native_sha256"] == p.NATIVE_SHA
          and manifest["installed"]["all_original_files_match"], "protocol/source/build identity")
    source_names = [p.PILOT, p.HERE / "protocol.py", p.HERE / "run.py",
                    p.old.ROOT / "tools/byte_release_artifacts.py", p.old.ROOT / "tools/owned_process.py"]
    check(manifest["source_sha256"] == {str(f.relative_to(p.old.ROOT)): sha(f) for f in source_names}, "source bytes")
    check(set(result["receipt_sha256"]) == set(NAMES)
          and all(sha(root / n) == result["receipt_sha256"][n] for n in NAMES), "receipt inventory or bytes")
    attempt, started = read(root / "fit-attempt.json"), read(root / "prepare-attempt.json")
    check(started["no_retry"] and attempt["no_retry"] and attempt["manifest_sha256"] == result["manifest_sha256"]
          and sha(root / "prepare-attempt.json") == manifest["attempt_sha256"] and started["source_commit"] == manifest["commit"]
          and started["deadline_epoch"] == attempt["deadline_epoch"] == manifest["deadline_epoch"]
          and started["started_epoch"] <= attempt["started_epoch"] <= result["finished_epoch"] <= attempt["deadline_epoch"]
          and attempt["deadline_epoch"] - started["started_epoch"] <= 3601
          and 0 <= manifest["prep_cpu_seconds"] + result["train_cpu_seconds"] <= 3600, "attempt/budget")
    data = {}
    check(set(manifest["files"]) == set(manifest["stats"]) == set(p.SPLITS), "dataset inventory")
    for k, n in p.SPLITS.items():
        stats = manifest["stats"][k]
        check(stats["documents"] == n and stats["target_tokens"] >= n and stats["longest"] <= 64, "dataset size")
        if originals:
            f = root / (k + ".json")
            check(sha(f) == manifest["files"][k], "encoded dataset pin")
            d = data[k] = read(f)
            check(d["keys"] == [r["key"] for r in records[k]] and len(d["labels"]) == len(d["token_documents"]) == n,
                  "encoded record keys")
            for doc, mask in zip(d["token_documents"], d["labels"]):
                first = next(i for i, t in enumerate(mask) if t != -100)
                check(first > 0 and len(mask) == len(doc) <= 64 and mask == [-100] * first + doc[first:], "supervision")
            check(sum(map(len, d["token_documents"])) == stats["tokens"]
                  and sum(t != -100 for mask in d["labels"] for t in mask) == stats["target_tokens"], "token counts")
    owners = read(root / "initial-owners.json")
    initial = owners["owners"]
    check(owners["counts"] == [4608, 4608, 4609] and len(initial) == 3 and len({x[0] for x in initial}) == 1
          and len({x[1] for x in initial}) == 3, "capacity/initial owners")
    masks = data.get("eval", {}).get("labels")
    start = read(root / "scores-000.json")
    ev = endpoint(start, initial, manifest["stats"]["eval"], masks)
    check(all(r["scores"] == [r["scores"][0]] * 3 for r in start["scores"]["answers"]), "initial predictions")
    ids, contrasts, steps = set(), {}, 0
    for order in p.ORDERS:
        loc = root / order
        plan = read(loc / "order.json")
        indices = p.train_order(records["train"], order)
        check(plan["keys"] == [records["train"][i]["key"] for i in indices] and p.pin(plan["keys"]) == frozen["orders"][order]
              and plan["data_id"] not in ids, "training permutation")
        ids.add(plan["data_id"])
        before = initial.copy()
        lines = (loc / "steps.jsonl").read_bytes().splitlines()
        check(len(lines) == 96, "incomplete training")
        targets = 0
        for i, line in enumerate(lines, 1):
            row = json.loads(line)
            rs = row["reports"]
            check(row["depth"] == 11 and len(rs) == 3, "matched prefix")
            n = rs[0]["target_tokens"]
            if originals:
                check(n == sum(t != -100 for t in data["train"]["labels"][indices[i-1]]), "target/order mismatch")
            for j, r in enumerate(rs):
                check([r["pre_lora_artifact_id"], r["pre_geometry_artifact_id"]] == before[j] and r["mode"] == "joint"
                      and [r["epoch"], r["update"], r["window_start"], r["window_end"]] == [0, i, i - 1, i]
                      and r["target_tokens"] == n > 0 and math.isfinite(r["loss"]), "update lineage or coverage")
                before[j] = [r["post_lora_artifact_id"], r["post_geometry_artifact_id"]]
            if i == 1:
                check(len({r["loss"] for r in rs}) == len({x[0] for x in before}) == 1, "first-step loss or LoRA")
            targets += n
            steps += len(rs)
        check(targets == manifest["stats"]["train"]["target_tokens"], "targets per arm")
        first = read(loc / "first-step.json")
        check(set(first) == {"bias", "gate"} and all(math.isfinite(v) and v >= 0 for v in first.values()), "origin report")
        entries = read(loc / "checkpoints.json")
        for j, (arm, e) in enumerate(zip(p.ARMS, entries, strict=True)):
            check(e["arm"] == arm and e["update"] == 96 and e["owners"] == before[j] and e["data_id"] == plan["data_id"]
                  and e["geometry_parameters"] == 1536 + int(arm == "wave_learned")
                  and (arm != "wave_fixed" or e["log_radius"] == 0.) and (arm != "affine" or e["log_radius"] is None),
                  "checkpoint owner/radius")
            if originals:
                checkpoint(loc / (arm + ".ck"), e)
        final = read(loc / "scores-096.json")
        check(endpoint(final, before, manifest["stats"]["eval"], masks) == ev, "changed heldout set/plan")
        contrasts[order] = final["contrasts"]
    decision = p.classify([contrasts[o]["fixed_minus_affine"] for o in p.ORDERS])
    check(steps == result["arm_updates"] == 864 and result["unique_eval_answers"] == 48 and result["distinct_orders"] == 3
          and contrasts == result["contrasts"] and result["decision"] == decision, "headline mismatch")
    return dict(decision=decision, arm_updates=steps, orders=3, unique_answers=48, scored_endpoints=4,
                checkpoint_boundaries=9, original_checkpoints_verified=9 if originals else 0, model_executions=0)


def describe(root, originals=False):
    """Descriptive read-back only: same answers, order sensitivity, and optional weight norms."""
    start = read(root / "scores-000.json")["scores"]["answers"]
    def means(rows):
        return [math.fsum(r["scores"][j]["nll"] for r in rows) / len(rows) for j in range(3)]
    lower = {arm: [] for arm in p.ARMS[:2]}
    orders = {}
    for order in p.ORDERS:
        loc = root / order
        rows = read(loc / "scores-096.json")["scores"]["answers"]
        for arm, (a, b) in zip(lower, [(0, 1), (1, 0)], strict=True):
            lower[arm].append({r["document"] for r in rows if r["scores"][a]["nll"] < r["scores"][b]["nll"]})
        entry = dict(answer_mean_nll=means(rows), changed_answer_hit_counts_since_start=[
                     sum(r["scores"][j]["top1_correct"] != s["scores"][j]["top1_correct"]
                         for r, s in zip(rows, start, strict=True)) for j in range(3)])
        if originals:
            norms = {}
            for arm in p.ARMS:
                ck = read(loc / (arm + ".ck"))
                site = json.loads(ck["geometry_json"])["placements"][0]
                norms[arm] = {t["name"]: math.hypot(*t["values"]) for t in site["tensors"] if t["name"] in ("gate", "bias")}
            entry["final_parameter_l2"] = norms
        orders[order] = entry
    return dict(initial_answer_mean_nll=means(start), orders=orders, lower_nll_in_every_order={
                arm: len(set.intersection(*sets)) for arm, sets in lower.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--originals", action="store_true")
    ap.add_argument("--describe", action="store_true", help="also summarize paired answers; no model execution")
    ap.add_argument("--export", type=Path, help="copy only the fixed receipt allowlist to a new directory")
    args = ap.parse_args()
    result = verify(args.directory, args.originals)
    if args.export is not None:
        check(args.originals, "an export requires verification of original checkpoints and data")
        args.export.mkdir(exist_ok=False)
        for name in (*NAMES, "manifest.json", "result.json"):
            dest = args.export / name
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(args.directory / name, dest)
        verify(args.export)
    if args.describe:
        result["descriptive"] = describe(args.directory, args.originals)
    print(json.dumps(result, sort_keys=True))
