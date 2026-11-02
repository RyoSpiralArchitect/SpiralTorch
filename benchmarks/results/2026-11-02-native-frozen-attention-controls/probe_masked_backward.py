"""Product compatibility only: replay one retained step on two trained GPT-2 snapshots."""

import argparse
from pathlib import Path
import os
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_contract import (
    HERE,
    PARENT,
    PLAN,
    ordering,
    read,
    require,
    rows,
    sha,
    verify_primary,
)
from control_study import ENV
from joint_protocol import PLAN as TRAIN, guard_ml, save
from joint_worker import checked_stage, load_base, paths, wheels
from common import FORBIDDEN
import owned_process

CASES = ("131-affine", "131-wave")


def descriptor(product_pin):
    return {
        "source_sha256": sha(Path(__file__)),
        "historical_result_sha256": PLAN["primary_result"],
        "product_wheel_sha256": product_pin,
        "cases": list(CASES),
        "boundaries": [256, 257],
        "seconds": 600,
        "workers": 1,
        "retries": 0,
    }


def probe(a):
    require(
        sha(a.product_wheel) == a.product_pin
        and sha(a.probe / "claim.json") == a.claim
        and read(a.probe / "claim.json") == descriptor(a.product_pin)
        and sha(a.out / "results.json") == PLAN["primary_result"]
        and sha(a.old / "data" / "train.json") == TRAIN["train_sha256"]
        and all(os.environ.get(k) == v for k, v in ENV.items())
        and not (a.probe / "status.json").exists()
        and not (a.probe / "first-error.json").exists(),
        "unbound product probe",
    )
    save(a.probe / "started.json", {"claim_sha256": a.claim})
    guard_ml()
    import spiraltorch as st

    inventory = wheels.wheel_archive_inventory(a.product_wheel)
    pkg = wheels.installed_identity(inventory)
    require(
        pkg["wheel_sha256"] == a.product_pin
        and pkg["all_original_files_match"]
        and pkg["native_sha256"] == sha(st._rs.__file__)
        and a.product_pin
        != read(PARENT / "execution.json")["development_wheel_sha256"],
        "not the new product wheel",
    )
    base = load_base(st, a.model)
    train = read(a.old / "data" / "train.json")
    window = train[ordering(131, len(train))[256]]
    fitted = read(a.out / "fitted.json")["cases"]
    evidence = {}
    for case in CASES:
        first = checked_stage(a.out, case, "first", fitted[case]["first"])
        checked_stage(a.out, case, "second", fitted[case]["second"])
        b = first["boundary"]
        line = rows(a.out / case / "second.jsonl")[0]
        require(
            line["update"] == 257
            and line["window_start"] == 256
            and line["mode"] == "joint",
            "not first post-resume update",
        )
        model = st.NativeCausalLm.from_session_checkpoint_at(
            base,
            (a.out / case / "checkpoint-256.json").read_text(),
            b["checkpoint_id"],
            update=256,
            recorded_data_id=b["recorded_data_id"],
            post_lora_id=b["post_lora_id"],
            post_geometry_id=b["post_geometry_id"],
        )
        all_g = model.adaptation_loss_gradients(window, [1, 64])
        losses = struct.pack("<f", all_g.loss)
        require(
            losses == struct.pack("<f", line["loss"])
            and all_g.target_tokens == 63
            and (model.lora_artifact_id, model.geometry_artifact_id)
            == (line["pre_lora_artifact_id"], line["pre_geometry_artifact_id"]),
            "different archived training event",
        )
        full = all_g.parameter_gradients()
        require(len(full) == 2, "missing full gradient group")
        joint = model.step_adaptation_sgd(all_g, "joint", 0.01)
        post = (line["post_lora_artifact_id"], line["post_geometry_artifact_id"])
        require(
            (joint.lora_artifact_id, joint.geometry_artifact_id) == post,
            "did not reproduce archived joint update",
        )
        groups = {}
        for mode in ("lora", "geometry"):
            partial = model.adaptation_loss_gradients(window, [1, 64], mode=mode)
            g = partial.parameter_gradients()
            require(
                len(g) == 1
                and all(g[k] == full[k] for k in g)
                and struct.pack("<f", partial.loss) == losses
                and partial.target_tokens == 63
                and (partial.lora_artifact_id, partial.geometry_artifact_id)
                == (model.lora_artifact_id, model.geometry_artifact_id),
                "different selected gradient or source",
            )
            next_model = model.step_adaptation_sgd(partial, mode, 0.01)
            expected = (
                (post[0], model.geometry_artifact_id)
                if mode == "lora"
                else (model.lora_artifact_id, post[1])
            )
            require(
                (next_model.lora_artifact_id, next_model.geometry_artifact_id)
                == expected,
                "selected update does not match archive",
            )
            groups[mode] = {k: sum(map(len, v.values())) for k, v in g.items()}
        require(
            set(groups["lora"]) | set(groups["geometry"]) == set(full),
            "different selected path roster",
        )
        evidence[case] = {
            "source_checkpoint": b["checkpoint_id"],
            "matched_historical_update": 257,
            "matched_full_and_selected_gradient_counts": groups,
            "loss_f32_le_hex": losses.hex(),
            "post_lora_id": post[0],
            "post_geometry_id": post[1],
        }
    require(
        wheels.installed_identity(inventory) == pkg
        and not FORBIDDEN.intersection(k.partition(".")[0] for k in sys.modules),
        "changed runtime",
    )
    return {
        "schema": "spiraltorch.native_lm.masked_backward_product_probe.v1",
        "claim_sha256": a.claim,
        "product_package": pkg,
        "cases": evidence,
        "gradient_calls": 6,
        "new_fits": 0,
    }


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    paths(p)
    p.add_argument("--product-wheel", type=Path, required=True)
    p.add_argument("--product-pin", required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--python", type=Path)
    p.add_argument("--claim")
    a = p.parse_args()
    if a.claim is not None:
        save(a.probe / "candidate.json", probe(a))
    else:
        require(
            a.python and a.python.is_file() and sha(a.product_wheel) == a.product_pin,
            "missing isolated product interpreter/wheel",
        )
        a.probe.mkdir(mode=0o700)
        c = descriptor(a.product_pin)
        save(a.probe / "claim.json", c)
        try:
            verify_primary(a)
            require(
                sha(a.out / "results.json") == PLAN["primary_result"],
                "different historical result",
            )
            cp = sha(a.probe / "claim.json")
            cmd = [
                str(a.python),
                "-I",
                "-B",
                str(HERE / "probe_masked_backward.py"),
                "--claim",
                cp,
                "--product-pin",
                a.product_pin,
            ]
            for k in ("out", "old", "model", "data", "wheel", "probe", "product_wheel"):
                cmd.extend(
                    [
                        "--" + k.replace("_", "-"),
                        str(getattr(a, k).resolve(strict=True)),
                    ]
                )
            os.environ.update(ENV)
            with (
                (a.probe / "stdout").open("xb") as out,
                (a.probe / "stderr").open("xb") as err,
            ):
                proc = owned_process.run(
                    cmd, timeout=c["seconds"], stdout=out, stderr=err
                )
            require(proc.returncode == 0, "first product worker failed")
            candidate = read(a.probe / "candidate.json")
            require(
                candidate["claim_sha256"] == cp
                and candidate["gradient_calls"] == 6
                and candidate["product_package"]["wheel_sha256"] == a.product_pin
                and set(candidate["cases"]) == set(CASES),
                "first product probe did not complete",
            )
            save(
                a.probe / "status.json",
                {
                    "status": "complete",
                    "claim_sha256": cp,
                    "result_sha256": sha(a.probe / "candidate.json"),
                },
            )
        except BaseException as exc:
            save(
                a.probe / "first-error.json",
                {"type": type(exc).__name__, "message": str(exc)},
            )
            raise
