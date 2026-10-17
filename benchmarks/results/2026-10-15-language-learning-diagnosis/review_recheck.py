"""Recheck the fixed historical diagnosis without rerunning either model.

The old reports and producer plan stay immutable. Old helper identities are
verified from their original Git revision, not from a newer working tree.
Only derived metrics, checks and hashes are emitted; raw logits stay private.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import types

ORIGINAL = "2308aaeee17b23a55100546b6ad4b8d8bffbc4d7"
DIRECTORY = "benchmarks/results/2026-10-15-language-learning-diagnosis"
RECHECK_SOURCE = Path(__file__).read_bytes()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_checker(path):
    raw = path.read_bytes()
    module = types.ModuleType("review_diagnose")
    module.__file__ = str(path)
    exec(compile(raw,str(path),"exec"),module.__dict__)
    return module, raw


def require_source_unchanged(path, raw):
    require(path.read_bytes() == raw, "source changed during revalidation")


def recheck(repo, private):
    require_source_unchanged(Path(__file__),RECHECK_SOURCE)
    def blob(path):
        return subprocess.check_output(["git", "-C", str(repo), "show", ORIGINAL+":"+path])

    receipt = json.loads(blob(DIRECTORY+"/validation.json"))
    raw_plan = (private/"plan-reviewed.json").read_bytes()
    require(sha(raw_plan) == receipt["private_plan_sha256"], "original plan identity differs")
    plan = json.loads(raw_plan)
    source_modes = {"original_git_blob": 0, "retained_private_file": 0}
    for filename, expected in plan["sources"].items():
        path = Path(filename)
        if path.is_relative_to(plan["repo"]):
            raw = blob(path.relative_to(plan["repo"]).as_posix())
            source_modes["original_git_blob"] += 1
        else:
            raw = path.read_bytes()
            source_modes["retained_private_file"] += 1
        require(sha(raw) == expected, "historical producer source/input differs")

    original_source = blob(DIRECTORY+"/diagnose.py")
    require(sha(original_source) == receipt["files"]["diagnose.py"], "historical checker identity")
    original = types.ModuleType("historical_diagnose")
    exec(compile(original_source, "historical_diagnose.py", "exec"), original.__dict__)
    source = Path(__file__).with_name("diagnose.py")
    current, checker_source = load_checker(source)

    archive = Path(plan["archive"])
    request_raw = (archive/"input/request.json").read_bytes()
    request = json.loads(request_raw)
    saved = json.loads((archive/"browser-checkpoint-128.json").read_bytes())
    require(saved["request_sha256"] == sha(request_raw) and saved["completed_updates"] == 128,
            "checkpoint lineage")
    gate = json.loads(Path(plan["gate"]).read_bytes())
    cohorts = [("historical-heldout",request["validation_documents"],request["validation_batches"])]
    cohorts += [(c["name"],gate["documents"],c["batches"]) for c in gate["cohorts"]]
    require(len(cohorts) == 4, "historical cohort coverage")
    directories = [private/"native-reviewed", private/"torch-reviewed"]
    reports = []
    for directory, name in zip(directories,("native-result.json","torch-result.json")):
        raw = (directory/"result.json").read_bytes()
        require(sha(raw) == receipt["files"][name], "historical result identity")
        reports.append(json.loads(raw))
    current.require_engines(*reports)
    require(all(r["plan_sha256"] == sha(raw_plan) for r in reports), "producer plan binding")
    baseline = current.fitted_baselines(request,cohorts,saved["completed_updates"])
    require(all(r[k] == v for r in reports for k,v in baseline.items()), "seen-only baseline differs")
    require(all(len(r["cases"]) == len(plan["selections"]) == 9 for r in reports), "case coverage")
    results, compared, maximum_ce_change = [], 0, 0.0
    for ci, selected in enumerate(plan["selections"]):
        pair = [r["cases"][ci] for r in reports]
        model = next(c for c in saved["cases"] if c["name"] == selected["name"])
        for case in pair:
            require(case["selection"] == selected and case["model_json_sha256"] == sha(model["model_json"].encode()),
                    "checkpoint selection differs")
            require(len(case["cohorts"]) == len(cohorts), "reported cohort coverage")
            current.validate_generation(case["generated_bytes"],request["config"]["batch"],plan["generation_bytes"])
        require(pair[0]["generated_bytes"] == pair[1]["generated_bytes"], "generation equality")
        corrected = []
        for ni, cohort in enumerate(cohorts):
            summaries, operands = [], []
            for directory, case in zip(directories,pair):
                record = case["cohorts"][ni]
                args = (directory,cohort,ci,ni,request["config"]["batch"],
                        request["config"]["steps"],plan["context_batches_per_cohort"])
                old_values = original.verified_scores(record,*args)
                values, metrics = current.read_scores(record,*args)
                require(old_values == values, "raw operands changed")
                changes = {key:metrics[key]["mean_ce"]-record[key]["mean_ce"]
                           for key in ("metrics","original_last","reversed_prefix_last")}
                maximum_ce_change = max(maximum_ce_change,*(abs(value) for value in changes.values()))
                summaries.append(dict(metrics=metrics,mean_ce_changes=changes))
                operands.append(values)
            require(len(operands[0]) == len(operands[1]) and operands[0], "logit coverage")
            differences = [abs(a-b) for a,b in zip(*operands)]
            require(all(error <= plan["parity"]["atol"]+plan["parity"]["rtol"]*abs(b)
                        for error,b in zip(differences,operands[1])), "unchanged logit parity failed")
            compared += len(differences)
            corrected.append(dict(cohort=cohort[0],compared_logits=len(differences),
                                  max_logit_absolute_error=max(differences),
                                  native=summaries[0],torch=summaries[1]))
        results.append(dict(selection=selected,cohorts=corrected,valid_generation_equal=True))
    require_source_unchanged(source,checker_source)
    require_source_unchanged(Path(__file__),RECHECK_SOURCE)
    return dict(schema="spiraltorch.frozen_diagnosis.review_recheck.v1",original_commit=ORIGINAL,
                original_plan_sha256=sha(raw_plan),original_result_sha256={
                    name:receipt["files"][name] for name in ("native-result.json","torch-result.json")},
                checker_sha256=sha(checker_source),recheck_sha256=sha(RECHECK_SOURCE),
                producer_source_checks=source_modes,cases=results,compared_logits=compared,
                maximum_mean_ce_change=maximum_ce_change,baseline_unchanged=True,
                original_summaries_verified=True,unchanged_numeric_gate_passed=True,
                original_outputs_rewritten=False,new_model_executions=0,new_training_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo","private","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    result = recheck(args.repo,args.private)
    with args.output.open("x",encoding="utf-8") as output:
        json.dump(result,output,indent=2,allow_nan=False)
        output.write("\n")
    print(json.dumps(dict(passed=True,compared_logits=result["compared_logits"],
                          maximum_mean_ce_change=result["maximum_mean_ce_change"])))


if __name__ == "__main__":
    main()
