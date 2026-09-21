"""Publish result records; retain native products and raw GELU arrays locally."""
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

repo, local, out = map(Path, sys.argv[1:4])
out.mkdir(parents=True)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_bytes())

def copy(source, relative):
    destination = out / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)

for folder in ["baseline-build", "candidate-build", "regression-before", "regression-after", "layout-after", "verification", "verification-final"]:
    for path in sorted((local / folder).rglob("*")):
        if path.is_file() and path.suffix in [".json", ".log"]:
            copy(path, path.relative_to(local))
for folder in ["gelu", "nn"]:
    for path in sorted((local / folder).iterdir()):
        if path.is_file() and not path.name.endswith(".raw.json"):
            copy(path, path.relative_to(local))
copy(local / "torch-run/stdout.log", "torch.json")
copy(local / "torch-run/receipt.json", "torch-receipt.json")
copy(local / "torch-run/stderr.log", "torch.stderr")
for name in ["run.py", "run.initial.py", "check.py", "execute.py", "measure.py", "torch_gelu.py", "publish.py"]:
    copy(local / name, "reproduction/" + name)
copy(local.parent / "cpu-nn-layout-20260921/measure.py", "reproduction/nn_measure.py")
copy(local.parent / "resident-graph-forward-clients-20260910/python-client.py", "reproduction/python-client.py")
for path in ["crates/st-bench/examples/cpu_gelu.rs", "crates/st-bench/examples/cpu_nn_layout.rs",
             "crates/st-nn/tests/gelu_layout_contract.rs", "bindings/st-py/tests/test_gelu_host.py",
             "bindings/st-wasm/tests/gelu_host.cjs"]:
    copy(repo / path, "reproduction/" + Path(path).name)
for name in ["verify.py", "test_verify.py"]:
    copy(local / name, name)
products = {lane + "/" + name: sha(local / (lane + "-build") / name)
            for lane in ["baseline", "candidate"] for name in ["cpu_gelu", "cpu_nn_layout"]}
for name in ["python-cpu.dylib", "wasm/spiraltorch_wasm.js", "wasm/spiraltorch_wasm_bg.wasm"]:
    products["final/" + name] = sha(local / "verification-final" / name)
baseline = read(local / "baseline-build/receipt.json")["source"]["commit"]
candidate = read(local / "candidate-build/receipt.json")["source"]["commit"]
validated = read(local / "verification-final/nn/receipt.json")["source"]["commit"]
provenance = {"baseline_source": baseline, "candidate_source": candidate, "validated_source": validated,
              "products": products, "raw_root": str(local), "platform": platform.platform(),
              "toolchain_observed_at_publication": subprocess.check_output(["rustc", "+1.98.0", "-Vv"], text=True),
              "host_exclusivity": "unknown", "thermal_state": "unknown", "numerical_replay": False,
              "raw_gelu_files": [{"path": p.relative_to(local).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
                                 for p in sorted((local / "gelu").glob("*.raw.json"))]}
(out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
for name, start, finish, paths in [
    ("runtime.patch", baseline, candidate, ["crates/st-tensor/src/pure.rs", "crates/st-nn/src/layers/gelu.rs"]),
    ("python-test-followup.patch", candidate, validated, ["bindings/st-py/tests/test_gelu_host.py"]),
]:
    (out / name).write_bytes(subprocess.check_output(["git", "diff", start, finish, "--", *paths], cwd=repo))
spec = importlib.util.spec_from_file_location("verifier", out / "verify.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
summary = module.summarize(out)
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"gelu": summary["gelu"]["groups"], "torch": summary["torch"]["groups"], "nn": summary["nn_groups"]}, indent=2))
