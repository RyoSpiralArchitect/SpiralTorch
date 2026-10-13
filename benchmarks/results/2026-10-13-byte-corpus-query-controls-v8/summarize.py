"""Publish scalar-only contextual corpus evidence from retained local reports."""
import argparse
import hashlib
import importlib.util
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "raw", "previous", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location("study", a.root / "tools/byte_corpus_study.py")
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    exact = study.load_reference("verify_byte_corpus_resume").exact
    raw = (a.raw / "input/request.json").read_bytes()
    request = study.decode_json(raw)
    if study.request_version(request) != 8:
        raise ValueError("expected v8")
    reports = {e:study.decode_json((a.raw / (e+".json")).read_bytes())
               for e in ("native","browser","torch")}
    a.output.mkdir(parents=True,exist_ok=True)
    qualification, inherited, contrasts, frozen_paths = {}, [], {}, []
    for engine, report in reports.items():
        if (report["request_sha256"] != study.digest(raw)
                or report["schema"] != "spiraltorch.byte_corpus.result.v8"
                or len(report["cases"]) != len(request["cases"])
                or any(c["name"] != s["name"] or len(c["training"]) != len(request["train_batches"])
                       for c,s in zip(report["cases"],request["cases"]))):
            raise ValueError("incomplete result")
        scalar = {k:v for k,v in report.items() if k != "cases"}
        scalar["cases"] = [{k:v for k,v in c.items() if k != "final_parameters"} for c in report["cases"]]
        scalar["raw_weights_omitted"] = True
        study.write_new(a.output / (engine+".json"),scalar)
        old = study.decode_json((a.previous / (engine+".json")).read_bytes())
        matches = [exact(before,after) for before,after in zip(old["cases"],report["cases"])]
        inherited.append(dict(engine=engine,source_cases=len(old["cases"]),all_exact=all(matches),
                              differing_cases=[c["name"] for c,ok in zip(old["cases"],matches) if not ok]))
        rows = []
        for case in report["cases"]:
            bpb = study.heldout_bpb(case["validation"][-1])
            rows.append(dict(name=case["name"],seed=case["seed"],mixture=case.get("mixture",{}),
                             final_heldout_bpb=bpb,reference_final_heldout_bpb=bpb))
        contrasts[engine] = [{k:v for k,v in c.items() if k in ("seed","metrics","measured")}
                             for c in study.load_reference("byte_corpus_query").contrasts(rows)]
        by_name = {c["name"]:c for c in report["cases"]}
        for case,row in zip(request["cases"],report["cases"]):
            q = case.get("mixture",{}).get("query_projection")
            if q is None or q["weight_update"] != "frozen":
                continue
            shared = by_name[case["name"].removesuffix("-query-frozen")]
            common = [v for d,v in zip(case["parameters"],row["final_parameters"])
                      if not d["name"].startswith("geometry.query_projection.")]
            frozen_paths.append(dict(engine=engine,name=case["name"],
                common_final_parameters_exact=exact(common,shared["final_parameters"]),
                training_exact=exact(row["training"],shared["training"]),
                validation_exact=exact(row["validation"],shared["validation"]),
                scope="Structural zero-path diagnostic; not a required cross-path bitwise gate."))
        if engine != "torch":
            try:
                checks = study.compare(raw,reports["torch"],report)
            except ValueError as error:
                checks = dict(numerical_checks_passed=False,error=str(error),
                              request_sha256=study.digest(raw),criteria_unchanged=True)
            study.write_new(a.output / (engine+"-comparison.json"),checks)
            qualification[engine] = checks["numerical_checks_passed"]
    study.write_new(a.output / "inheritance.json",dict(
        source_request_sha256=request["query_controls"]["source_request_sha256"],cases=inherited))
    study.write_new(a.output / "contrasts.json",dict(
        request_sha256=study.digest(raw),cases=contrasts,
        scope="All fixed-recipe contrasts, descriptive even if numerical qualification fails. Three seeds, not twelve independent samples."))
    activity = {}
    for engine in reports:
        file = a.raw / (engine+"-activity.json")
        if file.exists():
            record = study.decode_json(file.read_bytes())
            if (record["report_sha256"] != hashlib.sha256((a.raw/(engine+".json")).read_bytes()).hexdigest()
                    or record["request_sha256"] != study.digest(raw)):
                raise ValueError("foreign activity diagnostic")
            activity[engine] = record
    study.write_new(a.output / "activity.json",activity)
    study.write_new(a.output / "frozen-paths.json",frozen_paths)
    inputs = sorted(p for p in a.raw.rglob("*") if p.is_file())
    study.write_new(a.output / "raw-inventory.json",dict(
        artifacts=[dict(path=str(p.relative_to(a.raw)),bytes=p.stat().st_size,
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],
        local_raw_only=True,qualification=qualification,
        scope="Hashes identify retained local artifacts, not execution attestation."))
    print(qualification)


if __name__ == "__main__":
    main()
