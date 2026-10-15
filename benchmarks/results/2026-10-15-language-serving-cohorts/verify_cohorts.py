"""Pinned new-window evaluation of historical models; no training or ranking.

prepare uses stdlib only and freezes the selection before run imports SpiralTorch.
Outputs contain private data/models: use a private directory, not the repository.
"""
if not __debug__:
    raise RuntimeError("Run without Python optimization: assertions enforce evidence validity")

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

REVISION = "01e12ac8cc9bb10a1b151a8d7b1fda6b157fea64"
SOURCES = [
    ("autograd-prose", "docs/autograd_contract.md", "prose"),
    ("geometry-prose", "docs/causal_zspace_wave.md", "prose"),
    ("rust-code", "docs/ecosystem_integration.md", "rust"),
]
LIMITS = dict(max_mean_ce=10., max_mean_ce_regression=2., max_batch_ce_regression=2.,
              max_geometry_centered_rms=100., max_router_mass_error=1e-5,
              min_router_normalized_entropy=0.)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    raw = value if isinstance(value, str) else json.dumps(value, indent=2) + "\n"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as out:
        out.write(raw)


def spans(data, kind):
    """Offsets in original bytes, excluding all fence delimiters."""
    offset, start, mode = 0, None, "prose"
    result = []
    for line in data.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(b"```"):
            if start is not None:
                result.append((start, offset))
                start = None
            mode = (stripped[3:].decode("ascii") if mode == "prose" else "prose")
        elif mode == kind and start is None:
            start = offset
        offset += len(line)
    if start is not None:
        result.append((start, offset))
    return result


def prepare(repo, archive, output):
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    raw = (archive / "input/request.json").read_bytes()
    request = json.loads(raw)
    assert request["config"]["batch"] == 2 and request["config"]["steps"] == 32
    old_docs = [bytes(d) for d in request["train_documents"] + request["validation_documents"]]
    documents, cohorts, inventory, selected_windows = [], [], [], set()
    for index, (name, path, kind) in enumerate(SOURCES):
        data = subprocess.check_output(["git", "-C", str(repo), "show", f"{REVISION}:{path}"])
        assert data not in old_docs
        candidates, excluded = [], 0
        for begin, end in spans(data, kind):
            for start in range(begin, end - 32, 33):
                window = data[start:start + 33]
                if any(window[:32] in old or window[1:] in old for old in old_docs):
                    excluded += 1
                    continue
                candidates.append(start)
        assert len(candidates) >= 16, (name, len(candidates))
        offsets = [candidates[i * len(candidates) // 16] for i in range(16)]
        for start in offsets:
            window = data[start:start + 33]
            assert len(window) == 33 and window not in selected_windows
            selected_windows.add(window)
        batches = [[[index, offsets[i]], [index, offsets[i+1]]] for i in range(0, 16, 2)]
        documents.append(list(data))
        cohorts.append(dict(name=name, batches=batches, limits=LIMITS.copy()))
        inventory.append(dict(name=name, path=path, source_sha256=sha(data), source_bytes=len(data),
                              eligible_windows=len(candidates), excluded_exact_old_windows=excluded,
                              selected_offsets=offsets, target_bytes=512,
                              window_sha256=[sha(data[p:p+33]) for p in offsets]))
    selections = []
    for seed in (11, 23, 37):
        ordinary = [c for c in request["cases"] if c["seed"] == seed and not c["geometry"]]
        assert len(ordinary) == 1
        for mode in ("train", "frozen"):
            candidates = [c for c in request["cases"] if c["seed"] == seed
                          and c.get("mixture", {}).get("metrics") == ["poincare_squared.v1", "euclidean_chord_squared.v1"]
                          and c.get("mixture", {}).get("query_projection", {}).get("weight_update") == mode]
            assert len(candidates) == 1
            selections.append(dict(seed=seed, query_update=mode, incumbent=ordinary[0]["name"], candidate=candidates[0]["name"]))
    gate = json.dumps(dict(schema="spiraltorch.byte_corpus.serving_gate.v2", documents=documents,
                           steps=32, cohorts=cohorts), separators=(",", ":"))
    hashes = {p: sha((archive / p).read_bytes()) for p in
              ("input/request.json", "native-checkpoint-128.json", "browser-checkpoint-128.json")}
    write(output / "serving-gate.json", gate)
    write(output / "selection.json", dict(schema="spiraltorch.byte_corpus.cohort_selection.v1",
          source_revision=REVISION, source_hashes=hashes, gate_sha256=sha(gate.encode()),
          documents=inventory, selections=selections, new_training_updates=0,
          scope="Predeclared permissive mechanics limits, not quality thresholds. Three repo-document cohorts; no external-domain independence claim.",
          overlap_check="Every selected input and target 32-byte sequence absent from all historical train and validation document bytes. Shorter/near duplicates not excluded."))
    print(json.dumps(dict(prepared=True, cohorts=[c["name"] for c in cohorts], target_bytes=1536, comparisons=12)), flush=True)


def logits_hash(forward):
    values = forward.prediction_tensor().snapshot().read_values()
    return sha(struct.pack("<" + "f" * len(values), *values))


def run(archive, prepared, output):
    selection = json.loads((prepared / "selection.json").read_bytes())
    raw_gate = (prepared / "serving-gate.json").read_bytes()
    assert sha(raw_gate) == selection["gate_sha256"]
    hashes = selection["source_hashes"]
    assert hashes == {p: sha((archive / p).read_bytes()) for p in hashes}
    import spiraltorch as st
    gate = st.nn.ByteCorpusServingGate.from_json(raw_gate)
    assert gate.schema_version == 2 and gate.cohort_names == [s[0] for s in SOURCES]
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    study = st.nn.ByteCorpusStudy.from_json((archive / "input/request.json").read_bytes())
    policy = json.loads(raw_gate)
    host = st.nn.ByteLmBatch.from_documents([bytes(d) for d in policy["documents"]],
                [tuple(w) for w in policy["cohorts"][0]["batches"][0]], policy["steps"])
    records = []
    for client in ("native", "browser"):
        saved = study.checkpoint_from_json((archive / f"{client}-checkpoint-128.json").read_bytes())
        assert saved.completed_updates == 128
        for choice in selection["selections"]:
            initial = study.case_artifact(saved, choice["incumbent"], policy="fixed_window_frozen.v1")
            candidate = study.case_artifact(saved, choice["candidate"], policy="fixed_window_frozen.v1")
            study.validate_artifact(saved, initial)
            study.validate_artifact(saved, candidate)
            slot = gate.bootstrap_wgpu(initial)
            assert slot.tensor_device().adapter_info()["device_type"] != "Cpu"
            batch = slot.prepare_batch(host)
            before = logits_hash(slot.forward(batch))
            permit = slot.prepare_candidate(candidate)
            assessment = json.loads(permit.to_json())
            assert before == assessment["cohorts"][0]["incumbent"]["logits_f32le_sha256"][0]
            record = dict(source_client=client, selection=choice, assessment=assessment,
                          bootstrap=json.loads(slot.bootstrap_json()))
            if permit.accepted:
                record["activation"] = json.loads(slot.activate(permit))
                after = logits_hash(slot.forward(batch))
                assert after == assessment["cohorts"][0]["candidate"]["logits_f32le_sha256"][0]
                record["rollback"] = json.loads(slot.rollback(1))
                assert logits_hash(slot.forward(batch)) == before
                assert slot.artifact_id == initial.artifact_id and slot.generation == 2
                record.update(measured_forward_hashes_exact=True, rollback_exact=True)
            else:
                try:
                    slot.activate(permit)
                except ValueError:
                    pass
                else:
                    raise AssertionError("rejected cohort gate activated")
                assert slot.artifact_id == initial.artifact_id and slot.generation == 0
                record["rejection_preserved_owner"] = True
            if client == "browser" and choice == selection["selections"][0]:
                windows = [b for d, start in policy["cohorts"][0]["batches"][0]
                           for b in policy["documents"][d][start:start + policy["steps"] + 1]]
                write(output / "serving-input.json", dict(gate_json=raw_gate.decode(),
                      incumbent=dict(id=initial.artifact_id, json=initial.to_json()),
                      candidate=dict(id=candidate.artifact_id, json=candidate.to_json()), windows=windows))
            write(output / f"case-{len(records)}.json", record)
            records.append(record)
            print(json.dumps(dict(source=client, **choice, accepted=permit.accepted,
                  cohorts=[dict(name=c["name"], mean_ce_delta=c["mean_ce_delta"], failed_checks=c["failed_checks"])
                           for c in assessment["cohorts"]])), flush=True)
    assert hashes == {p: sha((archive / p).read_bytes()) for p in hashes}
    write(output / "qualification.json", dict(schema="spiraltorch.byte_corpus.cohort_qualification.v1",
          selection=selection, new_training_updates=0, sources_unchanged=True, cases=records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("repo", type=Path)
    p.add_argument("archive", type=Path)
    p.add_argument("output", type=Path)
    p = commands.add_parser("run")
    p.add_argument("archive", type=Path)
    p.add_argument("prepared", type=Path)
    p.add_argument("output", type=Path)
    args = vars(parser.parse_args())
    globals()[args.pop("mode")](**args)
