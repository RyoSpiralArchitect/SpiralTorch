"""Publish every fixed condition's results, never model or corpus payloads."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess

spec = importlib.util.spec_from_file_location("qualify", Path(__file__).with_name("qualify.py"))
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)
require, digest = qualify.require, qualify.digest
ARMS = ("ordinary", "geometry", "geometry_frozen")


def validate_evaluation(selection, result):
    require([s["name"] for s in selection["studies"]] == qualify.launch.names(), "all six studies")
    expected = [(s["name"], a["case_name"], a["cursor"]) for s in selection["studies"] for a in s["artifacts"]]
    require(len(expected) == 72 and [(r["study"], r["case_name"], r["cursor"]) for r in result["cases"]] == expected,
            "complete ordered evaluation matrix")
    for row in result["cases"]:
        study = next(s for s in selection["studies"] if s["name"] == row["study"])
        artifact = next(a for a in study["artifacts"] if (a["case_name"],a["cursor"]) == (row["case_name"],row["cursor"]))
        require(all(row[k] == study[k] for k in ("width","seed")), "evaluation condition identity")
        require(all(row[k] == artifact[k] for k in ("artifact_id","selected")), "evaluation artifact identity")
        require(row["numerical_checks_passed"] is study["comparison"]["numerical_checks_passed"], "qualification status drift")
        require(row["fresh_loss_exact_f32"] is True and row["generation_full_forward_oracle"] is True, "generation qualification")
        require(("probe" in row) == (row["cursor"] != 0), "probe coverage")
        if "probe" in row:
            require(row["probe"]["target_bytes"] == 2048 and len(row["probe"]["batch_losses"]) == 32, "probe target count")
        require(row["repetition"] == qualify.repetition(row["generation"]["generated_bytes"]), "repetition differs")
    require([b["cursor"] for b in result["baselines"]] == list(qualify.STOPS), "baseline exposure coverage")


def summarize(result):
    groups, pairs = [], []
    for width in qualify.launch.WIDTHS:
        for cursor in qualify.STOPS:
            base = next(b for b in result["baselines"] if b["cursor"] == cursor)
            for arm in ARMS:
                group = [r for r in result["cases"] if r["width"] == width and r["cursor"] == cursor
                         and r["case_name"].split("_",1)[1] == arm]
                require([r["seed"] for r in group] == list(qualify.launch.SEEDS), "all seeds in each arm")
                item = dict(width=width,cursor=cursor,arm=arm,seeds=len(group),
                            numerical_qualified_seeds=sum(r["numerical_checks_passed"] for r in group),
                            mean_space_fraction=statistics.mean(r["space_fraction"] for r in group),
                            mean_repeated_bigram_fraction=statistics.mean(r["repetition"]["mean_repeated_bigram_fraction"] for r in group))
                for metric in ("development", "probe"):
                    if metric not in group[0]:
                        continue
                    values = [r[metric]["mean_ce"] for r in group]
                    item[metric] = dict(mean_ce=statistics.mean(values), seed_min_ce=min(values), seed_max_ce=max(values),
                                        mean_ce_minus_bigram=statistics.mean(values)-base[metric]["bigram_ce"],
                                        seeds_beating_bigram=sum(v < base[metric]["bigram_ce"] for v in values))
                groups.append(item)
            for seed in qualify.launch.SEEDS:
                cases = {r["case_name"].split("_",1)[1]:r for r in result["cases"]
                         if r["width"] == width and r["cursor"] == cursor and r["seed"] == seed}
                item = dict(width=width,cursor=cursor,seed=seed,
                            numerical_checks_passed=all(r["numerical_checks_passed"] for r in cases.values()))
                for metric in ("development", "probe"):
                    if metric in cases["ordinary"]:
                        value = {k:r[metric]["mean_ce"] for k,r in cases.items()}
                        item[metric] = dict(learned_minus_ordinary=value["geometry"]-value["ordinary"],
                                            frozen_minus_ordinary=value["geometry_frozen"]-value["ordinary"],
                                            learned_minus_frozen=value["geometry"]-value["geometry_frozen"])
                pairs.append(item)
    return dict(means=groups, paired_deltas=pairs)


def validate_baselines(repo, request, result, protocol):
    helper = qualify.helpers(repo)
    path = "docs/byte_corpus_query_controls.md"
    probe = subprocess.check_output(["git","show",qualify.launch.REVISION+":"+path],cwd=repo)
    available = protocol.windows([probe],32)
    require(len(available) >= 64,"probe windows")
    starts = [available[i*len(available)//64][1] for i in range(64)]
    probe_windows = [probe[s:s+33] for s in starts]
    valid = [bytes(request["validation_documents"][d][s:s+33]) for batch in request["validation_batches"] for d,s in batch]
    require(result["probe"] == dict(path=path,revision=qualify.launch.REVISION,sha256=digest(probe),
                                   starts=starts,previously_inspected=True),"probe source/window identity")
    require(result["prompt_sha256"] == digest(b"".join(row[:32] for row in valid[:2])),"fixed generation prompt identity")
    expected = [dict(cursor=stop,development=helper.baselines(request,stop,valid),
                     probe=helper.baselines(request,stop,probe_windows)) for stop in qualify.STOPS]
    require(result["baselines"] == expected,"training-only baselines differ from recomputation")


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write("\n")


def public_selection(selection, raw_hash):
    public = copy.deepcopy(selection)
    for study in public["studies"]:
        comparison = study["comparison"]
        message = comparison.get("error","")
        if " != " in message:
            # Scalar comparison exceptions contain original parameter values.
            # Keep those private, while publishing the failed check and digest.
            label, separator, _ = message.partition(": ")
            safe_label = label+": " if separator and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*",label) else ""
            comparison["error"] = safe_label+"strict scalar tolerance exceeded"
            comparison["private_error_sha256"] = digest(message.encode())
    public["private_record_sha256"] = raw_hash
    return public


def publish(repo, root, output, old_directory):
    import spiraltorch as st
    selection_raw = (root/"selection.json").read_bytes()
    selection = json.loads(selection_raw)
    result = json.loads((root/"qualification.json").read_bytes())
    require(result["selection_sha256"] == digest(selection_raw), "selection binding")
    manifest, protocol = qualify.inputs(repo,root)
    require(selection["prepared_sha256"] == digest((root/"prepared.json").read_bytes()), "preparation binding")
    qualify.validate_selection(repo,root,manifest,protocol,selection,st,qualify.historical(repo,old_directory,protocol))
    require(result["native_extension_sha256"] == manifest["native_extension_sha256"],"evaluation runtime drift")
    validate_evaluation(selection,result)
    diagnostic = qualify.module(Path(__file__).with_name("diagnose.py")).measure(repo,root)
    require(diagnostic == json.loads((root/"numerical-diagnostics.json").read_bytes()),"diagnostics differ from recomputation")
    histories, files = [], ["PLAN.md","prepared.json","reference-launch.json","reference-completed.json",
                           "native-launch.json","native-completed.json","selection/claim.json","selection.json",
                           "qualification.json","evaluation/claim.json","numerical-diagnostics.json"]
    if (root/"launch-source.py").exists():
        files.append("launch-source.py")
    # Only completed producer/test logs, never the publisher's live stdout.
    logs = ("preflight.log","preflight-fixed.log","reference.log","native.log",
            "hardened-preflight.log","all-harness-tests-final.log","all-harness-tests-27.log","final-suite.log","segment-runner-tests.log",
            "reused-helper-tests.log","publisher-final-tests.log","selection.log","evaluation.log","numerical-diagnostics.log")
    files.extend(name for name in logs if (root/name).exists())
    if (root/"hardened-preflight/prepared.json").exists():
        files.append("hardened-preflight/prepared.json")
    for entry, study in zip(manifest["entries"],selection["studies"]):
        directory = root/entry["name"]
        request = protocol.decode_json((directory/"request.json").read_bytes())
        qualify.validate_coverage(request,entry,study)
        validate_baselines(repo,request,result,protocol)
        actual = protocol.decode_json((directory/"native/report-1024.json").read_bytes())
        reference = protocol.decode_json((directory/"torch.json").read_bytes())
        for engine, report in (("native",actual),("reference",reference)):
            for case in report["cases"]:
                histories.append(dict(study=entry["name"],engine=engine,case_name=case["name"],
                                      training_ce=[p["ce"] for p in case["training"]],validation=case["validation"]))
        names = ["request.json","torch.json","paired/preparation.json","paired/request.json",
                 "native/request.json","native/claim.json","native/completed.json"]
        names += [f"native/{kind}-{stop}.json" for kind in ("report","checkpoint","segment") for stop in qualify.STOPS]
        files += [entry["name"]+"/"+name for name in names]
        for engine in ("native","reference"):
            child = f"{entry['name']}/{engine}-child.json"
            if (root/child).exists():
                files.append(child)
        for item in study["artifacts"]:
            files.append("selection/"+item["file"])
            file = f"evaluation/{entry['name']}-{item['case_name']}-{item['cursor']}.json"
            files.append(file)
            raw = json.loads((root/file).read_bytes())
            expected = next(r for r in result["cases"] if (r["study"],r["case_name"],r["cursor"]) ==
                            (entry["name"],item["case_name"],item["cursor"]))
            require(raw == expected,"per-case evaluation receipt differs")
    summary = dict(schema="spiraltorch.byte_corpus.capacity_summary.v1",**summarize(result),
                   **{k:result[k] for k in ("baselines","prompt_sha256","probe","cases")},
                   scope="Previously inspected related-document capacity/budget diagnostic, not untouched test, equal compute, pretrained FT or promotion. Numerical flags at every saved cursor refer to the complete1024-update study, not independent prefix qualification.")
    # Finish all consistency checks before emitting public output.
    receipts = [dict(path=name,bytes=(root/name).stat().st_size,sha256=digest((root/name).read_bytes())) for name in files]
    write(output/"results.json",summary)
    write(output/"selection.json",public_selection(selection,digest(selection_raw)))
    write(output/"numerical-diagnostics.json",diagnostic)
    write(output/"preparation.json",manifest)
    write(output/"learning-curves.json",dict(records=histories))
    write(output/"private-receipts.json",dict(records=receipts))
    print(json.dumps(summary["means"],indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--historical",type=Path,required=True)
    args = parser.parse_args()
    publish(args.repo.resolve(),args.root.resolve(),args.output.resolve(),args.historical.resolve())


if __name__ == "__main__":
    main()
