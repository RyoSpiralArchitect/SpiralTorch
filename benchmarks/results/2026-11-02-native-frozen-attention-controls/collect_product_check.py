"""Verify the separate development-wheel handoff against its two archived step records."""

import argparse
import json
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_contract import PARENT, read, require, rows, sha
from probe_masked_backward import CASES, descriptor


def collect(v, primary):
    p = v / "archived-257"
    result = read(p / "candidate.json")
    pin = sha(p / "claim.json")
    package = result["product_package"]
    require(
        package["wheel_sha256"]
        == "1942f61c2f3830a08a95519c63d2144097bfeb6f556dd974a8291a343e1fc0e2"
        and package["native_sha256"]
        == "c56226af5ea3c976328eef4d9b1f8430e3450f18e49338d6be037b6ebb467796",
        "not the measured development product",
    )
    require(
        not (p / "first-error.json").exists()
        and read(p / "started.json") == {"claim_sha256": pin}
        and read(p / "claim.json") == descriptor(package["wheel_sha256"])
        and read(p / "status.json")
        == {
            "status": "complete",
            "claim_sha256": pin,
            "result_sha256": sha(p / "candidate.json"),
        }
        and result["claim_sha256"] == pin
        and result["gradient_calls"] == 6
        and result["new_fits"] == 0
        and set(result["cases"]) == set(CASES),
        "not the completed product probe",
    )
    require(
        sha(primary / "results.json")
        == read(p / "claim.json")["historical_result_sha256"]
        == sha(PARENT / "results.json"),
        "different study",
    )
    historical = read(primary / "results.json")["fits"]
    for case, values in result["cases"].items():
        step = rows(primary / case / "second.jsonl")[0]
        stage = read(primary / case / "second.json")
        require(
            sha(primary / case / "second.json")
            == historical[case]["terminal_receipt_sha256"]
            and stage["steps_sha256"] == sha(primary / case / "second.jsonl")
            and sha(primary / case / "first.json")
            == stage["previous_sha256"]
            == historical[case]["halfway_receipt_sha256"]
            and values["source_checkpoint"]
            == "sha256:" + sha(primary / case / "checkpoint-256.json")
            == read(primary / case / "first.json")["boundary"]["checkpoint_id"]
            and step["update"] == values["matched_historical_update"] == 257
            and struct.pack("<f", step["loss"]).hex() == values["loss_f32_le_hex"]
            and (step["post_lora_artifact_id"], step["post_geometry_artifact_id"])
            == (values["post_lora_id"], values["post_geometry_id"])
            and values["matched_full_and_selected_gradient_counts"]
            == {
                "lora": {"transformer.h.8.attn.c_proj": 6144},
                "geometry": {"transformer.h.8.attn": 1536 + int(case.endswith("wave"))},
            },
            "different archived update or returned parameter roster",
        )
    proofs = [
        v / n
        for n in (
            "pre-install-proof.json",
            "post-tests-proof.json",
            "post-probe-proof.json",
        )
    ]
    require(
        len({sha(file) for file in proofs}) == 1
        and read(proofs[0])["installed"] == package
        and read(proofs[0])["same_member_roster"]
        and read(proofs[0])["foreign_ml_installed"] is False
        and package["all_original_files_match"]
        and package["python_bytecode_absent"],
        "installed package changed",
    )
    core, nn = re.findall(
        r"test result: ok\. (\d+) passed; 0 failed; 0 ignored;",
        (v / "rust-with-declared-targets.log").read_text(),
    )
    suite = (v / "installed-suite.log").read_text()
    require(
        suite.rstrip().endswith("OK") and " ... skipped " not in suite,
        "incomplete installed suite",
    )
    python = re.search(r"Ran (\d+) tests in ", suite).group(1)
    for test in (
        "test_frozen_branch_does_not_evaluate_an_overflowing_gate_gradient",
        "test_elliptic_frozen_readout_does_not_reduce_unused_weight_products",
        "test_python_rust_python_checkpoint_handoff",
        "test_checkpoint_file_uses_native_bytes_and_refuses_existing_names",
    ):
        require(
            any(
                line.startswith(test + " ") and line.endswith(" ... ok")
                for line in suite.splitlines()
            ),
            "missing binding regression",
        )
    require(
        "Finished" in (v / "wasm-no-incremental.log").read_text(),
        "incomplete wasm compile",
    )
    return {
        "schema": "spiraltorch.native_lm.masked_backward_product_check.v1",
        "product_source": "5ef4d837bf89a9aada8e0cdc448fb9c7334ea2e3",
        "installed_checks": read(proofs[0]),
        "rust_library_tests": {"st-core": int(core), "st-nn": int(nn)},
        "installed_python_tests": int(python),
        "wasm32": "compile only, no incremental cache",
        "archived_step_probe": result,
        "claim_sha256": pin,
        "result_sha256": sha(p / "candidate.json"),
        "log_sha256": {
            name: sha(v / name)
            for name in (
                "rust-with-declared-targets.log",
                "rust-cli.log",
                "wasm-no-incremental.log",
                "installed-suite.log",
                "wheel-with-hip-build-support.log",
            )
        },
        "scope": "Native CPU compatibility, frozen adjoints and checkpoint persistence; not a completed new fit, throughput claim or release.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--primary", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(collect(a.validation, a.primary), sort_keys=True, indent=2))
