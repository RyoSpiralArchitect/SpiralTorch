"""Read-only saved-state checks; never loads HF weights or runs training."""

import hashlib
import json
import pathlib
import sys

import torch
import spiraltorch as st

if sys.flags.optimize:
    raise RuntimeError("saved-state verification requires assertions enabled")
if len(sys.argv) != 2:
    raise SystemExit("usage: verify_saved_state.py PRIVATE_RUN_DIRECTORY")
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
plan_bytes = pathlib.Path(__file__).with_name("plan.json").read_bytes()
plan = json.loads(plan_bytes)
cases = [(arm["name"], seed) for seed in plan["seeds"] for arm in plan["arms"]]
if (
    summary["status"] != "twelve_runs_completed"
    or summary["plan_id"] != "sha256:" + hashlib.sha256(plan_bytes).hexdigest()
    or [(r["arm"], r["seed"]) for r in summary["runs"]] != cases
):
    raise ValueError("saved-state check requires every declared completed case")
checks = []
for record in summary["runs"]:
    directory = root / f"{record['arm']}-seed-{record['seed']}"
    exported = st.GeometryAdapterArtifact.from_json(
        (directory / "geometry-export/spiraltorch-geometry.json").read_text(),
        record["artifact_id"],
        record["runtime_identity"],
    )
    final = json.loads(exported.state_json())
    for step in (256, 512):
        checkpoint = directory / f"checkpoint-{step}"
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        assert state["global_step"] == step and state["max_steps"] == 512
        optimizer = torch.load(
            checkpoint / "optimizer.pt", map_location="cpu", weights_only=True
        )
        scheduler = torch.load(
            checkpoint / "scheduler.pt", map_location="cpu", weights_only=True
        )
        assert len(optimizer["state"]) == 8
        assert sum(s["exp_avg"].numel() for s in optimizer["state"].values()) == 6144
        assert all(
            s["step"].item() == step
            and torch.isfinite(s["exp_avg"]).all().item()
            and torch.isfinite(s["exp_avg_sq"]).all().item()
            for s in optimizer["state"].values()
        )
        assert scheduler["last_epoch"] == step
        assert all(
            group["lr"] == 0.001 * (1 - step / 512)
            for group in optimizer["param_groups"]
        )
        names = [
            "spiraltorch-geometry.json",
            "optimizer.pt",
            "scheduler.pt",
            "trainer_state.json",
            "rng_state.pth",
            "spiraltorch-mps-rng.pt",
        ]
        ids = {
            name: "sha256:"
            + hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
            for name in names
        }
        matches = None
        if step == 512:
            artifact = st.GeometryAdapterArtifact.from_json(
                (checkpoint / "spiraltorch-geometry.json").read_text(),
                record["checkpoint_artifact_id"],
                record["runtime_identity"],
            )
            saved = json.loads(artifact.state_json())
            canonical = lambda r: [
                {k: v for k, v in p.items() if k != "training"} for p in r["placements"]
            ]
            matches = canonical(saved) == canonical(final)
            assert matches
        checks.append(
            {
                "arm": record["arm"],
                "seed": record["seed"],
                "step": step,
                "optimizer_parameters": 8,
                "trainable_values": 6144,
                "optimizer_scheduler_step_exact": True,
                "moments_finite": True,
                "checkpoint_export_state_exact_except_training_mode": matches,
                "private_file_ids": ids,
            }
        )
print(
    json.dumps(
        {
            "schema": "spiraltorch.hf_book_saved_state_check.v1",
            "status": "passed",
            "checkpoints": checks,
            "book_interrupt_resume_executed": False,
        }
    )
)
