"""Bounded training-only probes of native chart features and input Jacobians."""

import argparse
import json
import platform
from pathlib import Path

import torch
import transformers
import spiraltorch as st
import spiraltorch.spiraltorch as native
import spiraltorch.geometry_autograd as bridge

import hf_elliptic_anchored_study as client


def distribution(value):
    value = value.detach().double().flatten()
    client.study.require(
        bool(value.numel()) and bool(torch.isfinite(value).all()),
        "empty or nonfinite metric",
    )
    return {
        "mean": float(value.mean()),
        "quantiles": dict(
            zip(
                ["min", "p10", "p50", "p90", "max"],
                torch.quantile(
                    value, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)
                ).tolist(),
            )
        ),
    }


def metrics(features, jacobian, anchor):
    singular = torch.linalg.svdvals(jacobian.double())
    centered = features.double() - features.double().mean(0)
    energy = torch.linalg.svdvals(centered).square()
    client.study.require(
        bool((singular[:, -1] > 0).all()) and float(energy.sum()) > 0,
        "degenerate chart probe",
    )
    return {
        "feature_mean": features.double().mean(0).tolist(),
        "feature_std": features.double().std(0, correction=0).tolist(),
        "feature_covariance_effective_rank": float(
            energy.sum().square() / energy.square().sum()
        ),
        "centered_feature_energy": energy.tolist(),
        "anchor_displacement_l2": distribution(
            (features.double() - anchor.double()).norm(dim=-1)
        ),
        "jacobian_sigma_max": distribution(singular[:, 0]),
        "jacobian_sigma_min": distribution(singular[:, -1]),
        "jacobian_condition": distribution(singular[:, 0] / singular[:, -1]),
    }


def main():
    parser = argparse.ArgumentParser()
    for key in ("study", "model-dir", "corpus", "transfer-corpus", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    driver = client.study
    driver.require(not args.output.exists(), "refusing to replace a probe")
    receipts = {
        name: driver.pilot.digest((args.study / name).read_bytes())
        for name in ("plan.json", "journal.json", "results.json")
    }
    plan = json.loads((args.study / "plan.json").read_text())
    journal = json.loads((args.study / "journal.json").read_text())
    driver.require(
        driver.completed_result(args.study, journal, plan) is not None,
        "study must be completed",
    )
    config = plan["config"]
    torch.set_num_threads(config["threads"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True
    )
    tokenizer.model_max_length = 10**9
    train, _, _, data = driver.prepare_data(
        tokenizer, args.corpus.read_bytes(), args.transfer_corpus.read_bytes(), config
    )
    driver.require(data == plan["data"], "tokenized data differs from the frozen study")
    indices = driver.pilot.spaced_indices(len(train), 16)
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            args.model_dir, local_files_only=True, torch_dtype=torch.float32
        )
        .cpu()
        .eval()
        .requires_grad_(False)
    )
    model.config.use_cache = False
    driver.require(
        driver.pilot.model_digest(model) == plan["base_parameter_sha256"],
        "base weights differ from the frozen study",
    )
    hidden = []
    handle = model.get_submodule(config["block"]).register_forward_hook(
        lambda module, inputs, output: hidden.append(output.detach().clone())
    )
    try:
        with torch.no_grad():
            for start in range(0, len(indices), config["batch_size"]):
                model(train[indices[start : start + config["batch_size"]]])
    finally:
        handle.remove()
    hidden = torch.cat(hidden).reshape(-1, config["features"])
    warp = st.EllipticWarp(**config["warp"])
    anchor_snapshot = warp.map_orientations_batch([1.0, 0.0, 0.0])
    anchor = torch.tensor(anchor_snapshot.features)
    tangent = torch.tensor(
        [anchor_snapshot.vjp([float(i == j) for j in range(9)])[1:] for i in range(9)]
    )
    report = {
        "schema": "spiraltorch.elliptic_chart_probe.v2",
        "study_id": plan["study_id"],
        "study_files_sha256": receipts,
        "base_parameter_sha256": plan["base_parameter_sha256"],
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        },
        "source_sha256": {
            name: driver.pilot.digest(Path(module.__file__).read_bytes())
            for name, module in {
                "native": native,
                "bridge": bridge,
                "client": client,
                "driver": driver,
                "pilot": driver.pilot,
            }.items()
        },
        "probe_sha256": driver.pilot.digest(Path(__file__).read_bytes()),
        "training_block_indices": indices,
        "hidden_shape": list(hidden.shape),
        "hidden_sha256": driver.pilot.digest(hidden.numpy().tobytes()),
        "scope": "No training updates or endpoint evaluation; both maps probed at the same orientations",
        "rows": [],
    }
    for seed in config["seeds"]:
        for state in ("initial", "anchored_tangent", "anchored_elliptic"):
            adapter = client.adapter_for(
                "anchored_tangent" if state == "initial" else state, config, seed
            )
            if state != "initial":
                key = f"{seed}:{state}"
                saved = driver.load_checkpoint(
                    args.study,
                    journal["runs"][key]["checkpoint"],
                    plan["study_id"],
                    key,
                )
                adapter.load_state_dict(saved["adapter"])
            with torch.no_grad():
                coords = adapter.orientation(hidden)
            orientation = torch.cat([torch.ones_like(coords[:, :1]), coords], -1)
            snapshot = warp.map_orientations_batch(orientation.flatten().tolist())
            features = torch.tensor(snapshot.features).reshape(-1, 9)
            columns = []
            for coordinate in (1, 2):
                direction = torch.zeros_like(orientation)
                direction[:, coordinate] = 1
                columns.append(
                    torch.tensor(snapshot.jvp(direction.flatten().tolist())).reshape(
                        -1, 9
                    )
                )
            jacobian = torch.stack(columns, -1)
            # Independently recover the transpose through the existing reverse API.
            reverse_rows = []
            for feature in range(9):
                upstream = torch.zeros_like(features)
                upstream[:, feature] = 1
                reverse_rows.append(
                    torch.tensor(snapshot.vjp(upstream.flatten().tolist())).reshape(
                        -1, 3
                    )[:, 1:]
                )
            driver.require(
                torch.equal(jacobian, torch.stack(reverse_rows, 1)),
                "native JVP/VJP basis mismatch",
            )
            ordinary = anchor + coords @ tangent.T
            row = {
                "seed": seed,
                "projection_state": state,
                "raw_mix": float(adapter.raw_mix.detach()),
                "checkpoint_sha256": (
                    None
                    if state == "initial"
                    else journal["runs"][key]["checkpoint"]["sha256"]
                ),
                "jvp_vjp_basis_exact": True,
                "coordinate_l2": distribution(coords.norm(dim=-1)),
                "elliptic": metrics(features, jacobian, anchor),
                "tangent": metrics(
                    ordinary, tangent.expand(len(coords), -1, -1), anchor
                ),
                "max_gain_ratio": distribution(
                    torch.linalg.svdvals(jacobian.double())[:, 0]
                    / torch.linalg.svdvals(tangent.double())[0]
                ),
            }
            report["rows"].append(row)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "state": state,
                        "coords": row["coordinate_l2"],
                        "gain": row["max_gain_ratio"],
                    }
                ),
                flush=True,
            )
    driver.require(
        driver.pilot.model_digest(model) == plan["base_parameter_sha256"],
        "probe mutated base weights",
    )
    driver.require(
        receipts
        == {
            name: driver.pilot.digest((args.study / name).read_bytes())
            for name in receipts
        },
        "study records changed during probe",
    )
    driver.atomic_json(args.output, report)


if __name__ == "__main__":
    main()
