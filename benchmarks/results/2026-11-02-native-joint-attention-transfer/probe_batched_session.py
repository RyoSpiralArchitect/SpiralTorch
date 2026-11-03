"""Functional GPT-2 check: one 2,2,1-window schedule and a Rust process handoff."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from joint_protocol import PLAN, digest, guard_ml, read, require, save, sha
from joint_worker import FORBIDDEN, artifact, load_base, put_payload, wheels
import owned_process


def probe(a):
    require(
        sha(a.wheel) == a.wheel_pin
        and sha(a.rust) == a.rust_pin
        and sha(a.initial / "receipt.json") == a.init_pin
        and sha(a.old / "data" / "train.json") == PLAN["train_sha256"],
        "unbound probe inputs",
    )
    a.out.mkdir(mode=0o700, exist_ok=False)
    guard_ml()
    import spiraltorch as st

    inv = wheels.wheel_archive_inventory(a.wheel)
    pkg = wheels.installed_identity(inv)
    require(
        pkg["wheel_sha256"] == a.wheel_pin
        and pkg["native_sha256"] == sha(st._rs.__file__),
        "wrong installed code",
    )
    base = load_base(st, a.model)
    ids = read(a.initial / "receipt.json")["owners"]
    initial = base.with_lora(8, 4, 8.0).with_geometry(
        artifact(st, base, a.initial / "wave.json", ids["wave"])
    )
    require(initial.lora_artifact_id == ids["lora"], "initial LoRA")
    train = read(a.old / "data" / "train.json")
    docs = [train[i] for i in (0, 170, 341, 511, 17)]
    save(a.out / "docs.json", docs)
    session = st.NativeCausalLmTrainingSession(
        initial, docs, 64, learning_rate=0.01, mode="joint", epochs=2, batch_windows=2
    )
    manual = initial
    require(
        session.total_updates == 6 and session.batch_windows == 2, "batched schedule"
    )
    seals = []
    for i, (left, right) in enumerate([(0, 2), (2, 4), (4, 5)] * 2, 1):
        batch = [n for d in docs[left:right] for n in d]
        gradients = manual.adaptation_loss_gradients(batch, [right - left, 64])
        manual = manual.step_adaptation_sgd(gradients, "joint", 0.01)
        r = session.step()
        require(
            r.update == i
            and r.target_tokens == 63 * (right - left)
            and r.loss == gradients.loss
            and manual.geometry_artifact_id == session.model.geometry_artifact_id
            and manual.lora_artifact_id == session.model.lora_artifact_id,
            "batch differs from direct Rust gradient and update",
        )
        payload, pin = session.checkpoint_with_id()
        seals.append(pin)
        if i == 2:
            put_payload(a.out / "before-rust.json", payload, pin)
        if i == 3:
            cmd = [
                str(a.rust),
                str(a.model / "config.json"),
                str(a.model / "model.safetensors"),
                str(a.out / "docs.json"),
                str(a.out / "before-rust.json"),
                seals[1],
                str(a.out / "after-rust.json"),
            ]
            with (
                (a.out / "rust.stdout").open("xb") as stdout,
                (a.out / "rust.stderr").open("xb") as stderr,
            ):
                status = owned_process.run(
                    cmd, timeout=180, stdout=stdout, stderr=stderr
                )
            require(status.returncode == 0, "Rust process failed")
            rr = read(a.out / "rust.stdout")
            require(
                rr["checkpoint_id"] == pin == "sha256:" + sha(a.out / "after-rust.json")
                and rr["step"]["target_tokens"] == 63
                and rr["step"]["update"] == 3,
                "Rust tail handoff differs",
            )
            session = st.NativeCausalLmTrainingSession.from_json(
                base, docs, (a.out / "after-rust.json").read_bytes(), pin
            )
            require(
                session.next_window == 0 and session.epoch == 1, "wrong resumed epoch"
            )
    require(session.exhausted and session.step() is None, "incomplete real-model probe")
    restored = st.NativeCausalLm.from_session_checkpoint_at(
        base,
        payload.encode(),
        pin,
        update=6,
        recorded_data_id=session.data_id,
        post_lora_id=manual.lora_artifact_id,
        post_geometry_id=manual.geometry_artifact_id,
    )

    def fingerprint(m):
        return hashlib.sha256(
            m.forward(docs[0][:7], [1, 7], last_only=True).values_buffer()
        ).hexdigest()

    last = fingerprint(restored)
    require(
        last == fingerprint(manual)
        and wheels.installed_identity(inv) == pkg
        and not FORBIDDEN.intersection(n.partition(".")[0] for n in sys.modules),
        "inference or install differs",
    )
    return {
        "schema": "spiraltorch.native_lm.joint_minibatch_handoff.v1",
        "wheel_sha256": a.wheel_pin,
        "native_sha256": pkg["native_sha256"],
        "rust_executable_sha256": a.rust_pin,
        "initial_receipt_sha256": a.init_pin,
        "data_id": session.data_id,
        "batch_windows": 2,
        "epoch_window_counts": [2, 2, 1, 2, 2, 1],
        "target_counts": [126, 126, 63, 126, 126, 63],
        "matched_direct_updates": 6,
        "cross_process_tail_update": 3,
        "cursor_after_handoff": [1, 0],
        "checkpoint_ids": seals,
        "terminal_logits_sha256": last,
        "scope": "mechanics only; separate from the nine single-window fits",
    }


if __name__ == "__main__":
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("old", "model", "initial", "wheel", "rust", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("wheel-pin", "rust-pin", "init-pin"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    save(a.out / "result.json", probe(a))
