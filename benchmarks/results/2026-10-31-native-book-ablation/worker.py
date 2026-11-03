"""Torch-free workers for the published native GPT-2 book comparison."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[2] / "tools"
sys.path[:0] = [str(HERE), str(TOOLS)]
import byte_release_artifacts as wheels  # noqa: E402 - frozen checkout under isolated Python
from common import (  # noqa: E402
    FORBIDDEN,
    PLAN,
    SPLITS,
    cases,
    digest,
    guard_ml,
    ordering,
    read,
    require,
    save,
    sha,
    weighted,
)


def source_pins():
    return {
        p.name: sha(p)
        for p in [
            HERE / n
            for n in ("plan.json", "common.py", "prepare.py", "worker.py", "study.py")
        ]
        + [TOOLS / "byte_release_artifacts.py", TOOLS / "owned_process.py"]
    }


def tensor(name, values, *, scalar=False, trainable=True):
    return {
        "name": name,
        "shape": [] if scalar else [len(values)],
        "values": values,
        "role": "parameter" if trainable else "buffer",
        "requires_grad": trainable,
    }


def initial(st, base, arm):
    width = PLAN["features"]
    zeros = [0.0] * width
    kernel = st.WaveGateKernel(**PLAN["kernel"])
    logr = math.log(PLAN["radius"])
    sites = []
    for layer in PLAN["layers"]:
        bias = [
            PLAN["bias_amplitude"]
            * (1 if hashlib.sha256(f"{layer}:{i}".encode()).digest()[0] % 2 else -1)
            for i in range(width)
        ]
        if arm == "affine":
            # At gate=0 the radial branch is independent of input. Copy that
            # exact nonzero vector, not merely its norm, into the ordinary arm.
            bias = kernel.forward_with_log_radius(
                zeros, zeros, bias, 1, width, logr
            ).output
            recipe = {
                "schema": "spiraltorch.channel_affine_adapter.v1",
                "max_values": PLAN["kernel"]["max_values"],
            }
        else:
            require(arm in ("fixed4", "learned4"), "unknown treatment")
            recipe = {
                "schema": "spiraltorch.wave_gate_adapter.v2",
                "kernel": PLAN["kernel"],
                "learnable_radius": arm == "learned4",
            }
        parameters = [tensor("gate", zeros), tensor("bias", bias)]
        if arm != "affine":
            parameters.append(
                tensor("log_radius", [logr], scalar=True, trainable=arm == "learned4")
            )
        sites.append(
            {
                "path": f"transformer.h.{layer}.mlp",
                "training": False,
                "recipe": dict(recipe, features=width, strength=PLAN["strength"]),
                "tensors": parameters,
            }
        )
    return st.GeometryAdapterArtifact.create_json(
        json.dumps(
            {
                "schema": "spiraltorch.geometry_adapter_artifact.v1",
                "base_model_identity": base.base_model_identity,
                "policy": "full_prefix_f32_no_cache.v1",
                "training": False,
                "placements": sites,
            }
        )
    )


def load_base(st, model):
    for name, expected in PLAN["model"].items():
        require(sha(model / name) == expected, "model/tokenizer bytes changed: " + name)
    weights = st.SafeTensorArchive(model / "model.safetensors")
    base = st.NativeCausalLm.from_gpt2((model / "config.json").read_text(), weights)
    require(
        base.base_model_identity == PLAN["base_id"]
        and base.readout_accumulator == "f32"
        and base.execution_backend == "rust_f32_cpu_prepacked",
        "native base/execution route differs",
    )
    return base


def logits_id(model, window):
    raw = model.forward(window, [1, len(window)], last_only=True).values_buffer()
    return hashlib.sha256(raw).hexdigest()


def artifact(st, base, path, expected):
    require(
        "sha256:" + sha(path) == expected, "artifact file differs from external pin"
    )
    return st.GeometryAdapterArtifact.from_json(
        path.read_text(), expected, base.base_model_identity
    )


def put_payload(path, payload, expected):
    require(
        "sha256:" + hashlib.sha256(payload.encode()).hexdigest() == expected,
        "native payload identity differs",
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def prepare_native(st, base, data, manifest, output, package):
    windows = read(data / "train.json")
    require(
        sha(data / "train.json") == manifest["files"]["train.json"],
        "training windows changed",
    )
    ids, site_hashes, logits = {}, {}, {}
    indices = (0, 170, 341, 511)
    logits["base"] = [logits_id(base, windows[i]) for i in indices]
    for arm in PLAN["arms"]:
        state = initial(st, base, arm)
        put_payload(output / (arm + ".json"), state.to_json(), state.artifact_id)
        ids[arm] = state.artifact_id
        route = st.NativeGeometryRuntime(state)
        probe = [((i % 31) - 15) / 8 for i in range(2 * PLAN["features"])]
        site_hashes[arm] = [
            hashlib.sha256(
                route.forward(f"transformer.h.{layer}.mlp", probe, 2).output_buffer()
            ).hexdigest()
            for layer in PLAN["layers"]
        ]
        model = base.with_geometry(state)
        logits[arm] = [logits_id(model, windows[i]) for i in indices]
    require(
        all(
            logits[a] == logits["affine"] and site_hashes[a] == site_hashes["affine"]
            for a in PLAN["arms"]
        )
        and all(x != y for x, y in zip(logits["affine"], logits["base"])),
        "nonzero matched start not verified",
    )
    return {
        "schema": "native_book_preflight.v1",
        "plan_sha256": sha(HERE / "plan.json"),
        "source_sha256": source_pins(),
        "data_manifest_sha256": sha(data / "data-manifest.json"),
        "installed": package,
        "base": base.base_model_identity,
        "initial": ids,
        "train_probe_logits": logits,
        "train_probe_site_hashes": site_hashes,
        "evaluation_scored": False,
        "updates": 0,
    }


def fit(st, base, args, manifest, contract, output):
    arm, seed = args.arm, args.seed
    require((seed, arm) in cases(), "unknown case")
    require(
        sha(args.data / "train.json") == manifest["files"]["train.json"],
        "training bytes changed",
    )
    windows = read(args.data / "train.json")
    order = ordering(seed, len(windows))
    state = artifact(
        st, base, args.preflight / (arm + ".json"), contract["initial"][arm]
    )
    optimizer = st.NativeGeometryAdamW(
        state, max_gradient_norm=PLAN["max_gradient_norm"]
    )
    ordered = [windows[i] for i in order]
    trainer = st.NativeCausalLmTrainer(
        base.with_geometry(state), optimizer, ordered, **PLAN["recipe"]
    )
    require(
        trainer.total_updates == PLAN["updates"], "unexpected number of native updates"
    )
    prev = state.artifact_id
    total = 0
    with (output / "steps.jsonl").open("x", buffering=1) as steps:
        for index in range(trainer.total_updates):
            step = trainer.step()
            require(
                step is not None
                and step.update == index + 1
                and step.target_tokens == 126
                and (step.window_start, step.window_end) == (2 * index, 2 * index + 2)
                and step.epoch == 0
                and step.microbatches == 1
                and step.pre_geometry_artifact_id == prev
                and math.isfinite(step.loss),
                "incomplete or nonfinite update",
            )
            steps.write(step.to_json() + "\n")
            prev = step.post_geometry_artifact_id
            total += step.target_tokens
    require(
        trainer.exhausted
        and total == PLAN["train_windows"] * (PLAN["recipe"]["context_length"] - 1),
        "training budget differs",
    )
    final = trainer.model.geometry_artifact()
    require(final.artifact_id == prev, "last step/geometry mismatch")
    put_payload(output / "checkpoint.json", trainer.to_json(), trainer.checkpoint_id)
    put_payload(output / "geometry.json", final.to_json(), final.artifact_id)
    start_sites, end_sites = [
        json.loads(a.to_json())["placements"] for a in (state, final)
    ]
    changes = {"channel_sites_changed": 0, "radius_sites_changed": 0}
    for before, after in zip(start_sites, end_sites, strict=True):
        require(
            before["path"] == after["path"] and before["recipe"] == after["recipe"],
            "placement changed",
        )
        changes["channel_sites_changed"] += any(
            a["values"] != b["values"]
            for a, b in zip(before["tensors"], after["tensors"], strict=True)
            if a["name"] in ("gate", "bias")
        )
        for a, b in zip(before["tensors"], after["tensors"], strict=True):
            require(
                a["name"] == b["name"] and all(math.isfinite(x) for x in b["values"]),
                "invalid fitted state",
            )
            if a["name"] == "log_radius":
                changed = a["values"] != b["values"]
                require(arm == "learned4" or not changed, "fixed radius mutated")
                changes["radius_sites_changed"] += changed
    require(
        logits_id(base, windows[0]) == contract["train_probe_logits"]["base"][0],
        "frozen base changed",
    )
    return {
        "arm": arm,
        "seed": seed,
        "updates": trainer.completed_updates,
        "target_tokens": total,
        "data_id": trainer.data_id,
        "order_sha256": digest(order),
        "checkpoint_id": trainer.checkpoint_id,
        "geometry_id": final.artifact_id,
        "steps_sha256": sha(output / "steps.jsonl"),
        "evaluation_scored": False,
        **changes,
    }


def score(st, base, args, manifest, contract):
    require(
        sha(args.fits / "fitted.json") == args.fit_pin, "frozen final roster changed"
    )
    roster = read(args.fits / "fitted.json")
    require(
        list(roster["order"]) == [f"{s}-{a}" for s, a in cases()]
        and set(roster["cases"]) == set(roster["order"])
        and roster["preflight_pin"] == args.preflight_pin,
        "incomplete/different case roster",
    )
    require(
        sha(args.data / "evaluation.json") == manifest["files"]["evaluation.json"],
        "evaluation bytes changed",
    )
    groups = read(args.data / "evaluation.json")
    require(set(groups) == set(SPLITS), "evaluation splits differ")

    def variants():
        yield "base", base
        yield (
            "initial",
            base.with_geometry(
                artifact(
                    st,
                    base,
                    args.preflight / "affine.json",
                    contract["initial"]["affine"],
                )
            ),
        )
        for key in roster["order"]:
            fitted = roster["cases"][key]
            state = artifact(
                st, base, args.fits / key / "geometry.json", fitted["geometry_id"]
            )
            yield key, base.with_geometry(state)

    results = {}
    for key, model in variants():
        split_values = {}
        for split in SPLITS:
            chapters = {}
            for number, windows in sorted(
                groups[split].items(), key=lambda a: int(a[0])
            ):
                metric = model.evaluate_documents(
                    windows, PLAN["recipe"]["context_length"], microbatch_windows=2
                )
                require(
                    metric.target_tokens == sum(len(w) - 1 for w in windows)
                    and metric.ignored_targets == 0
                    and metric.skipped_windows == 0
                    and math.isfinite(metric.nll),
                    "missing evaluation targets",
                )
                chapters[number] = {
                    "nll": metric.nll,
                    "targets": metric.target_tokens,
                    "data_id": metric.data_id,
                }
            split_values[split] = {
                "nll": weighted([(x["nll"], x["targets"]) for x in chapters.values()]),
                "targets": sum(x["targets"] for x in chapters.values()),
                "chapters": chapters,
            }
        save(args.output / (key + ".json"), split_values)
        results[key] = split_values
        print("scored " + key, flush=True)
    return results


def main(args):
    os.umask(0o077)
    args.output.mkdir(mode=0o700, exist_ok=False)
    attempted = guard_ml()
    require(sha(args.wheel) == PLAN["wheel_sha256"], "not the published macOS wheel")
    inventory = wheels.wheel_inventory(args.wheel)
    package = wheels.installed_identity(inventory)
    require(
        package["version"] == PLAN["package"]
        and package["native_sha256"] == PLAN["native_sha256"],
        "installed package differs",
    )
    require(
        sha(args.data / "data-manifest.json") == args.data_pin,
        "data preparation external pin differs",
    )
    manifest = read(args.data / "data-manifest.json")
    require(
        manifest["plan_sha256"] == sha(HERE / "plan.json"),
        "data belongs to another plan",
    )
    import spiraltorch as st

    require(
        sha(Path(st._rs.__file__)) == PLAN["native_sha256"],
        "wrong loaded native library",
    )
    base = load_base(st, args.model)
    t = time.monotonic()
    if args.mode == "preflight":
        result = prepare_native(st, base, args.data, manifest, args.output, package)
    else:
        require(
            sha(args.preflight / "result.json") == args.preflight_pin,
            "external preflight pin differs",
        )
        contract = read(args.preflight / "result.json")
        require(
            contract["source_sha256"] == source_pins()
            and contract["data_manifest_sha256"] == args.data_pin
            and contract["installed"] == package,
            "frozen code, data or installed package changed",
        )
        if args.mode == "fit":
            result = fit(st, base, args, manifest, contract, args.output)
        else:
            result = score(st, base, args, manifest, contract)
    require(
        not (FORBIDDEN & {n.partition(".")[0] for n in sys.modules}),
        "foreign ML module loaded",
    )
    require(wheels.installed_identity(inventory) == package, "installed bytes changed")
    save(
        args.output / "runtime.json",
        {
            "arm": args.arm,
            "seconds_after_model_load": time.monotonic() - t,
            "foreign_loaded": False,
            "blocked_optional_imports": attempted,
            "native": package["native_sha256"],
        },
    )
    save(args.output / "result.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["preflight", "fit", "score"])
    for name in ("model", "data", "output", "wheel", "preflight", "fits"):
        p.add_argument(
            "--" + name,
            type=Path,
            required=name in ("model", "data", "output", "wheel"),
        )
    p.add_argument("--data-pin", required=True)
    p.add_argument("--preflight-pin")
    p.add_argument("--fit-pin")
    p.add_argument("--arm", choices=PLAN["arms"])
    p.add_argument("--seed", type=int, choices=PLAN["seeds"])
    main(p.parse_args())
