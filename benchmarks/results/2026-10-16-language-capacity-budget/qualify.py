"""Fixed capacity/budget experiment qualification; no serving promotion."""
import argparse
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


launch = module(Path(__file__).with_name("launch.py"))
require, digest, write = launch.require, launch.digest, launch.write
STOPS = launch.STOPS


def helpers(repo):
    # Reuse the already qualified experiment-only baselines and independent
    # generation oracle, not a second production runtime or selection rule.
    return module(repo / "benchmarks/results/2026-10-15-language-adamw-256/qualify.py")


def inputs(repo, root):
    manifest = launch.validated_inputs(repo, root, allow_recorded_launcher=True)
    prepared_hash = digest((root / "prepared.json").read_bytes())
    for engine in ("reference", "native"):
        completed = json.loads((root / (engine + "-completed.json")).read_bytes())
        require(completed == dict(prepared_sha256=prepared_hash, studies=launch.names()), "incomplete matrix")
    protocol = module(repo / "tools/byte_corpus_study.py")
    paths = (["docs/byte_corpus_learning.md", "docs/byte_corpus_calibration.md"], ["docs/resident_adamw.md"])
    documents = [[list(subprocess.check_output(["git", "show", launch.REVISION+":"+path], cwd=repo))
                  for path in group] for group in paths]
    expected = [(width, seed) for width in launch.WIDTHS for seed in launch.SEEDS]
    require([(e["width"], e["seed"]) for e in manifest["entries"]] == expected, "matrix identities")
    for entry in manifest["entries"]:
        request = protocol.decode_json((root / entry["name"] / "request.json").read_bytes())
        launch.validate_request(protocol, request, entry["width"], entry["seed"], documents)
    return manifest, protocol


def checked_run(repo, root, entry, protocol, st):
    verifier = module(repo / "tools/verify_byte_corpus_resume.py")
    directory = root / entry["name"]
    raw = (directory / "request.json").read_bytes()
    request = protocol.decode_json(raw)
    require((directory / "native/request.json").read_bytes() == raw, "native input differs")
    completed = json.loads((directory / "native/completed.json").read_bytes())
    require(completed["request_sha256"] == digest(raw), "completed request drift")
    require([s["stop_after"] for s in completed["segments"]] == list(STOPS), "saved cursor coverage")
    study = st.nn.ByteCorpusStudy.from_json(raw)
    reports, checkpoints, saved = [], [], []
    for stop, receipt in zip(STOPS, completed["segments"]):
        report = (directory / f"native/report-{stop}.json").read_bytes()
        checkpoint = (directory / f"native/checkpoint-{stop}.json").read_bytes()
        require(digest(report) == receipt["report_sha256"] and digest(checkpoint) == receipt["checkpoint_sha256"], "segment digest")
        require(json.loads((directory / f"native/segment-{stop}.json").read_bytes()) == receipt, "segment receipt differs")
        reports.append(protocol.decode_json(report))
        parsed = study.checkpoint_from_json(checkpoint)
        validate_cursor(parsed,stop)
        checkpoints.append(parsed)
        saved.append(protocol.decode_json(checkpoint))
    for partial, checkpoint in zip(reports, saved):
        # This is saved-state binding to one segmented run, not a second
        # uninterrupted run and not independent exact-resume evidence.
        verifier.verify(request, digest(raw), reports[-1], partial, reports[-1], checkpoint)
    reference_raw = (directory / "torch.json").read_bytes()
    reference = protocol.decode_json(reference_raw)
    comparison = compare_retaining_failure(protocol, raw, reference, reports[-1])
    comparison.update(reference_sha256=digest(reference_raw), native_report_sha256=completed["segments"][-1]["report_sha256"])
    return request, study, reports, checkpoints, comparison


def validate_cursor(checkpoint, stop):
    require(checkpoint.completed_updates == stop,"saved checkpoint cursor differs from declared stop")


def validate_runtime(st, manifest):
    actual = digest(Path(st._rs.__file__).read_bytes())
    require(actual == manifest["native_extension_sha256"],"qualification native extension differs")
    return actual


def compare_retaining_failure(protocol, raw, reference, actual):
    try:
        result = protocol.compare(raw, reference, actual)
        return dict(numerical_checks_passed=True, comparison=result)
    except ValueError as error:
        return dict(numerical_checks_passed=False, error=str(error),
                    scope="Strict qualification failed; diagnostic outputs only, no quality or promotion claim")


def historical(repo, directory, protocol):
    receipts = json.loads((repo / "benchmarks/results/2026-10-15-language-adamw-256/private-receipts.json").read_bytes())
    result = []
    for name in ("request.json", "native/report-256.json"):
        raw = (directory / name).read_bytes()
        declared = next(r for r in receipts["records"] if r["path"] == name)
        require(digest(raw) == declared["sha256"] and len(raw) == declared["bytes"], "historical source drift")
        result.append(protocol.decode_json(raw))
    return result


def vector_comparison(left, right):
    require(len(left) == len(right), "historical vector coverage")
    require(all(math.isfinite(x) for x in left+right), "nonfinite historical vector")
    a = [struct.pack("<f", x) for x in left]
    b = [struct.pack("<f", x) for x in right]
    return dict(exact_f32=a == b, max_abs_error=max((abs(x-y) for x,y in zip(left,right)), default=0.))


def repeat_comparison(old, new):
    groups = {
        "parameters": lambda c: list(itertools.chain.from_iterable(c["final_parameters"])),
        "training": lambda c: [r["ce"] for r in c["training"]],
        "validation": lambda c: [v for row in c["validation"] for v in row["batch_losses"]],
        "first_moments": lambda c: [v for s in c["optimizer_state"]["slots"] for v in s["first"]],
        "second_moments": lambda c: [v for s in c["optimizer_state"]["slots"] for v in s["second"]],
    }
    require(old["name"] == new["name"], "historical case identity")
    return {name: vector_comparison(extract(old), extract(new)) for name, extract in groups.items()}


def historical_comparisons(request, reports, entry, old_request, old_report):
    require(request["train_batches"][:256] == old_request["train_batches"],"historical batch prefix")
    records = []
    if entry["width"] == 16:
        for index, case in enumerate(request["cases"]):
            old_case = next(c for c in old_request["cases"] if c["name"] == case["name"])
            require(launch.canonical(case) == launch.canonical(old_case),"historical initial values")
            previous = next(c for c in old_report["cases"] if c["name"] == case["name"])
            records.append(dict(case_name=case["name"],fields=repeat_comparison(previous,reports[1]["cases"][index])))
    return records


def source_receipts(repo):
    paths = [Path(__file__), Path(__file__).with_name("launch.py"),
             repo / "benchmarks/results/2026-10-15-language-adamw-256/qualify.py",
             repo / "tools/byte_corpus_study.py", repo / "tools/byte_corpus_optimizer.py",
             repo / "tools/verify_byte_corpus_resume.py"]
    return [dict(path=str(p.relative_to(repo)), sha256=digest(p.read_bytes())) for p in paths]


def select(repo, root, old_directory):
    import spiraltorch as st
    manifest, protocol = inputs(repo, root)
    runtime_hash = validate_runtime(st,manifest)
    old_request, old_report = historical(repo, old_directory, protocol)
    output = root / "selection"
    output.mkdir(mode=0o700, exist_ok=False)
    write(output / "claim.json", dict(prepared_sha256=digest((root/"prepared.json").read_bytes()), before_probe_read=True))
    studies = []
    for entry in manifest["entries"]:
        request, study, reports, checkpoints, comparison = checked_run(repo, root, entry, protocol, st)
        repeated = historical_comparisons(request,reports,entry,old_request,old_report)
        decisions, artifacts = [], []
        for index, case in enumerate(request["cases"]):
            chosen, decision_json = study.select_checkpoint(checkpoints, case["name"], policy="fixed_window_frozen.v1")
            decision = json.loads(decision_json)
            means = [sum(protocol.f32(v) for v in report["cases"][index]["validation"][-1]["batch_losses"])/32 for report in reports]
            expected = min(range(len(STOPS)), key=lambda i: (means[i], STOPS[i]))
            require(decision["selected"]["completed_updates"] == STOPS[expected], "selection oracle")
            study.validate_artifact(checkpoints[expected], chosen)
            decisions.append(decision)
            for stop, checkpoint in zip(STOPS, checkpoints):
                artifact = study.case_artifact(checkpoint, case["name"], policy="fixed_window_frozen.v1")
                filename = f"{entry['name']}-case{index}-revision{stop}.json"
                data = artifact.to_json()
                # The artifact itself remains private, including its weights.
                with os.fdopen(os.open(output/filename, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600), "w") as stream:
                    stream.write(data)
                artifacts.append(dict(case_name=case["name"], cursor=stop, file=filename,
                                      artifact_id=artifact.artifact_id, sha256=digest(data.encode()),
                                      selected=artifact.artifact_id == chosen.artifact_id))
        studies.append(dict(**entry, comparison=comparison, decisions=decisions, artifacts=artifacts,
                            segment_binding=True, historical_256=repeated))
        print(json.dumps(dict(selected=entry["name"], numerical_checks_passed=comparison["numerical_checks_passed"],
                              cursors=[d["selected"]["completed_updates"] for d in decisions])), flush=True)
    write(root/"selection.json", dict(schema="spiraltorch.byte_corpus.capacity_selection.v1", studies=studies,
          prepared_sha256=digest((root/"prepared.json").read_bytes()),sources=source_receipts(repo),
          native_extension_sha256=runtime_hash))


def validate_coverage(request, entry, record):
    require(all(record[k] == entry[k] for k in entry), "selection study identity")
    names = [c["name"] for c in request["cases"]]
    require([d["case_name"] for d in record["decisions"]] == names, "all case decisions in order")
    require([(a["case_name"], a["cursor"]) for a in record["artifacts"]] ==
            [(name, stop) for name in names for stop in STOPS], "all saved models in order")
    for index, decision in enumerate(record["decisions"]):
        selected = decision["selected"]
        candidate = selected["candidate_index"]
        require(type(candidate) is int and 0 <= candidate < len(STOPS)
                and selected["completed_updates"] == STOPS[candidate], "selected candidate cursor")
        for item, stop in zip(record["artifacts"][index*4:(index+1)*4], STOPS):
            require(item["file"] == f"{entry['name']}-case{index}-revision{stop}.json", "artifact filename")
            require(type(item["selected"]) is bool and item["selected"] == (stop == STOPS[candidate]), "selected flags")
            if item["selected"]:
                require(item["artifact_id"] == selected["artifact_id"], "selected identity")


def validate_selection(repo, root, manifest, protocol, selection, st, old_inputs):
    require(selection["prepared_sha256"] == digest((root/"prepared.json").read_bytes()), "selection source drift")
    require(selection["sources"] == source_receipts(repo), "qualification source drift")
    require([s["name"] for s in selection["studies"]] == launch.names(), "all six selected studies")
    runtime_hash = validate_runtime(st,manifest)
    require(selection["native_extension_sha256"] == runtime_hash,"selection runtime drift")
    for entry, record in zip(manifest["entries"], selection["studies"]):
        request, study, reports, checkpoints, comparison = checked_run(repo, root, entry, protocol, st)
        require(record["comparison"] == comparison, "comparison differs from actual/reference recomputation")
        expected_history = historical_comparisons(request,reports,entry,*old_inputs)
        require(record["historical_256"] == expected_history,"historical repeat comparison differs")
        validate_coverage(request, entry, record)
        for index, decision in enumerate(record["decisions"]):
            _, current = study.select_checkpoint(checkpoints, decision["case_name"], policy="fixed_window_frozen.v1")
            require(json.loads(current) == decision, "selection differs from Rust")
            for checkpoint, item in zip(checkpoints, record["artifacts"][index*4:(index+1)*4]):
                expected = study.case_artifact(checkpoint, decision["case_name"], policy="fixed_window_frozen.v1")
                data = (root/"selection"/item["file"]).read_bytes()
                require(item["artifact_id"] == expected.artifact_id and digest(data) == item["sha256"]
                        and data == expected.to_json().encode(), "artifact differs from saved source")


def repetition(generated):
    pairs = [(a,b) for row in generated for a,b in zip(row,row[1:])]
    return dict(adjacent_repeat_fraction=sum(a == b for a,b in pairs)/len(pairs),
                mean_repeated_bigram_fraction=statistics.mean(1-len(set(zip(row,row[1:])))/(len(row)-1) for row in generated),
                max_constant_run=max(len(list(g)) for row in generated for _,g in itertools.groupby(row)))


def evaluate(repo, root, old_directory):
    import spiraltorch as st
    manifest, protocol = inputs(repo, root)
    selection_raw = (root/"selection.json").read_bytes()
    selection = json.loads(selection_raw)
    validate_selection(repo,root,manifest,protocol,selection,st,historical(repo,old_directory,protocol))
    output = root / "evaluation"
    output.mkdir(mode=0o700, exist_ok=False)
    write(output/"claim.json", dict(selection_sha256=digest(selection_raw), before_probe_read=True))
    helper = helpers(repo)
    # Every fixed case/cursor, Rust selection, and saved artifact was checked
    # before reading the monitor probe. It is already inspected, not a test set.
    probe_path = "docs/byte_corpus_query_controls.md"
    probe = subprocess.check_output(["git", "show", launch.REVISION+":"+probe_path], cwd=repo)
    available = protocol.windows([probe], 32)
    require(len(available) >= 64, "monitor probe size")
    starts = [available[i*len(available)//64][1] for i in range(64)]
    probe_windows = [probe[s:s+33] for s in starts]
    rows, baselines, prompts_hash = [], None, None
    for record in selection["studies"]:
        directory = root / record["name"]
        request = protocol.decode_json((directory/"request.json").read_bytes())
        require(list(probe) not in request["train_documents"]+request["validation_documents"], "probe document overlap")
        valid = [bytes(request["validation_documents"][d][s:s+33]) for batch in request["validation_batches"] for d,s in batch]
        prompts = [row[:32] for row in valid[:2]]
        current_hash = digest(b"".join(prompts))
        require(prompts_hash in (None, current_hash), "generation prompts differ across conditions")
        prompts_hash = current_hash
        current_baselines = [dict(cursor=stop, development=helper.baselines(request,stop,valid),
                                  probe=helper.baselines(request,stop,probe_windows)) for stop in STOPS]
        require(baselines is None or baselines == current_baselines, "baseline exposure differs across conditions")
        baselines = current_baselines
        reports = {stop: protocol.decode_json((directory/f"native/report-{stop}.json").read_bytes()) for stop in STOPS}
        for item in record["artifacts"]:
            artifact = st.nn.ByteCorpusModelArtifact.from_json((root/"selection"/item["file"]).read_bytes(), expected_id=item["artifact_id"])
            owner = artifact.restore_wgpu()
            adapter = owner.tensor_device().adapter_info()
            require(adapter["device_type"] != "Cpu", "real GPU required")
            forward = owner.forward(owner.prepare_batch(st.nn.ByteLmBatch.from_windows(valid[:2])))
            actual_ce = forward.next_byte_loss().loss_tensor().snapshot().read_values()[0]
            case = next(c for c in reports[item["cursor"]]["cases"] if c["name"] == item["case_name"])
            require(struct.pack("<f",actual_ce) == struct.pack("<f",case["validation"][-1]["batch_losses"][0]), "fresh artifact loss binding")
            generation = json.loads(owner.generate_greedy(prompts,max_new_bytes=32))
            expected, hashes = helper.generation_oracle(artifact.restore_wgpu(),prompts,32,st)
            require(generation["generated_bytes"] == expected and generation["next_logits_f32le_sha256"] == hashes, "generation full-forward oracle")
            flat = list(itertools.chain.from_iterable(expected))
            row = dict(study=record["name"], width=record["width"], seed=record["seed"],
                       case_name=item["case_name"], cursor=item["cursor"], artifact_id=item["artifact_id"],
                       selected=item["selected"], numerical_checks_passed=record["comparison"]["numerical_checks_passed"],
                       development=case["validation"][-1], generation=generation, adapter=adapter,
                       generated_latin1=[bytes(v).decode("latin1") for v in expected],
                       space_fraction=flat.count(32)/len(flat), distinct_bytes=len(set(flat)),
                       repetition=repetition(expected), fresh_loss_exact_f32=True, generation_full_forward_oracle=True)
            if item["cursor"] in (256,512,1024):
                losses = []
                for start in range(0,len(probe_windows),2):
                    batch = st.nn.ByteLmBatch.from_windows(probe_windows[start:start+2])
                    loss = owner.forward(owner.prepare_batch(batch)).next_byte_loss().loss_tensor().snapshot().read_values()[0]
                    require(math.isfinite(loss), "nonfinite probe loss")
                    losses.append(loss)
                row["probe"] = dict(batch_losses=losses,mean_ce=sum(losses)/len(losses),target_bytes=2048)
            write(output/f"{record['name']}-{item['case_name']}-{item['cursor']}.json",row)
            rows.append(row)
            print(json.dumps(dict(evaluated=record["name"],case=item["case_name"],cursor=item["cursor"])),flush=True)
    require(len(rows) == 72, "evaluation matrix incomplete")
    write(root/"qualification.json",dict(schema="spiraltorch.byte_corpus.capacity_qualification.v1",cases=rows,
          selection_sha256=digest(selection_raw), baselines=baselines, prompt_sha256=prompts_hash,
          native_extension_sha256=manifest["native_extension_sha256"],
          probe=dict(path=probe_path,revision=launch.REVISION,sha256=digest(probe),starts=starts,previously_inspected=True)))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("select","evaluate"))
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--historical",type=Path,required=True)
    args = parser.parse_args()
    args.repo, args.root = args.repo.resolve(), args.root.resolve()
    if args.historical is not None:
        args.historical = args.historical.resolve()
    if args.command == "select":
        require(args.historical is not None,"historical directory is required")
        select(args.repo,args.root,args.historical)
    else:
        evaluate(args.repo,args.root,args.historical)


if __name__ == "__main__":
    main()
