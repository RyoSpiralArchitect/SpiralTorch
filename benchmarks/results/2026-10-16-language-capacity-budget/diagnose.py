"""Post-hoc localization of strict numerical failures, never a replacement gate."""
import argparse
import importlib.util
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location("qualify",Path(__file__).with_name("qualify.py"))
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)
require = qualify.require


def errors(actual, expected, protocol):
    require(len(actual) == len(expected),"diagnostic vector coverage")
    require(all(type(v) in (int,float) and math.isfinite(v) for v in actual+expected),"nonfinite diagnostic vector")
    failures = [i for i,(a,e) in enumerate(zip(actual,expected)) if abs(a-e) > protocol.ATOL+protocol.RTOL*abs(e)]
    norm = math.sqrt(math.fsum(v*v for v in expected))
    difference = math.sqrt(math.fsum((a-e)**2 for a,e in zip(actual,expected)))
    return dict(max_abs_error=max((abs(a-e) for a,e in zip(actual,expected)),default=0.),
                failed_indices=failures, reference_l2=norm,error_l2=difference,
                relative_l2_error=difference/norm if norm else (0. if difference == 0 else None))


def qkv_parts(indices,width):
    require(all(type(i) is int and 0 <= i < 3*width for i in indices),"QKV bias index")
    return {name:sum(start <= i < stop for i in indices)
            for name,start,stop in (("query",0,width),("key",width,2*width),("value",2*width,3*width))}


def measure(repo,root):
    manifest,protocol = qualify.inputs(repo,root)
    optimizer = qualify.module(repo/"tools/byte_corpus_optimizer.py")
    records = []
    for entry in manifest["entries"]:
        directory = root/entry["name"]
        request_raw = (directory/"request.json").read_bytes()
        request = protocol.decode_json(request_raw)
        ref_raw = (directory/"torch.json").read_bytes()
        actual_raw = (directory/"native/report-1024.json").read_bytes()
        reference,actual = protocol.decode_json(ref_raw),protocol.decode_json(actual_raw)
        comparison = qualify.compare_retaining_failure(protocol,request_raw,reference,actual)
        expected_names = [c["name"] for c in request["cases"]]
        require(all([c["name"] for c in report["cases"]] == expected_names for report in (actual,reference)),"diagnostic case coverage")
        for case,a,e in zip(request["cases"],actual["cases"],reference["cases"]):
            parameters = []
            require(len(a["final_parameters"]) == len(e["final_parameters"]) == len(case["parameters"]),"parameter family coverage")
            for index,descriptor in enumerate(case["parameters"]):
                value = errors(a["final_parameters"][index],e["final_parameters"][index],protocol)
                item = dict(name=descriptor["name"],scalars=len(descriptor["values"]),parameters=value)
                if descriptor["name"].endswith("qkv.bias"):
                    item["qkv_failure_counts"] = qkv_parts(value["failed_indices"],entry["width"])
                for moment in ("first","second"):
                    result = errors(a["optimizer_state"]["slots"][index][moment],e["optimizer_state"]["slots"][index][moment],protocol)
                    ratio = result["relative_l2_error"]
                    result["relative_gate_passed"] = ratio is not None and ratio <= optimizer.MOMENT_RTOL
                    item[moment+"_moment"] = result
                parameters.append(item)
            training = errors([v["ce"] for v in a["training"]],[v["ce"] for v in e["training"]],protocol)
            validation = errors([v for p in a["validation"] for v in p["batch_losses"]],
                                [v for p in e["validation"] for v in p["batch_losses"]],protocol)
            records.append(dict(study=entry["name"],width=entry["width"],seed=entry["seed"],case_name=case["name"],
                                request_sha256=entry["request_sha256"],reference_sha256=qualify.digest(ref_raw),
                                actual_sha256=qualify.digest(actual_raw),study_gate_passed=comparison["numerical_checks_passed"],
                                training=training,validation=validation,parameters=parameters))
    return dict(schema="spiraltorch.byte_corpus.numerical_localization.v1",post_hoc=True,
                criteria=dict(atol=protocol.ATOL,rtol=protocol.RTOL,moment_relative_l2=optimizer.MOMENT_RTOL),
                records=records,scope="Localization only; all original strict failures remain failures. No parameter values or relaxed gate.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo","root","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    qualify.write(args.output.resolve(),measure(args.repo.resolve(),args.root.resolve()))


if __name__ == "__main__":
    main()
