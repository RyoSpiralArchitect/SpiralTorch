"""Compare a tiny native CPU run with an observed run cut across two processes."""

import argparse
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

HERE = Path(__file__).resolve()
CUT_EXIT = 43
DOCS = [[1, 2, 3, 4], [2, 3, 4, 5]]


class NoForeignML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in {
            "torch",
            "numpy",
            "transformers",
            "safetensors",
            "accelerate",
        }:
            raise AssertionError("foreign ML import: " + fullname)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def child(root, phase, receipt_id):
    sys.meta_path.insert(0, NoForeignML())
    import spiraltorch as st

    fixtures = runpy.run_path(str(HERE.with_name("test_native_causal_lm.py")))
    config, weights = fixtures["fixture"]()
    base = st.NativeCausalLm.from_gpt2(config, weights)
    dst = root / phase
    dst.mkdir(mode=0o700)
    (dst / "native.sha256").write_text(sha(Path(st._rs.__file__).read_bytes()))
    if phase == "resume":
        from spiraltorch.native_corpus_runs import restore_observed_checkpoint

        trainer = restore_observed_checkpoint(
            root / "cut",
            2,
            expected_observation_id=receipt_id,
            base=base,
            token_documents=DOCS,
        )
    else:
        artifact = fixtures["artifact"](base, trainable=True)
        trainer = st.NativeCausalLmTrainer(
            base.with_geometry(artifact),
            st.NativeGeometryAdamW(artifact, max_gradient_norm=0.1),
            DOCS,
            4,
            learning_rate=0.01,
            microbatch_windows=1,
            accumulation_steps=1,
            epochs=2,
        )
    if trainer.total_updates != 4 or trainer.completed_updates != (
        2 if phase == "resume" else 0
    ):
        raise ValueError("unexpected training recipe or cursor")
    with (dst / "steps.jsonl").open("xb") as log:
        while not trainer.exhausted:
            cursor = trainer.completed_updates + 1
            if phase == "cut" and cursor == 2:
                from spiraltorch.native_corpus_runs import step_and_save_observation

                step, saved = step_and_save_observation(trainer, dst)
                log.write(step.to_json().encode() + b"\n")
                log.flush()
                print(saved["observation_id"], flush=True)
                os._exit(CUT_EXIT)  # Bypass Python cleanup and any terminal summary.
            step = trainer.step()
            log.write(step.to_json().encode() + b"\n")
            data = trainer.to_json().encode()
            if trainer.checkpoint_id != "sha256:" + sha(data):
                raise ValueError("native checkpoint pin differs")
            with (dst / f"checkpoint-{cursor:06d}.json").open("xb") as ckpt:
                ckpt.write(data)


def compare(root, reference, candidate):
    root.mkdir(mode=0o700, exist_ok=False)

    def run(python, phase, receipt_id="", expected_exit=0):
        with (
            (root / (phase + ".stdout")).open("xb") as out,
            (root / (phase + ".stderr")).open("xb") as err,
        ):
            proc = subprocess.run(
                [python, "-I", "-B", HERE, "--child", phase, root, receipt_id],
                stdout=out,
                stderr=err,
                timeout=60,
                env={
                    **os.environ,
                    "RAYON_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
            )
        if proc.returncode != expected_exit:
            raise RuntimeError(
                f"{phase}: exit {proc.returncode}, expected {expected_exit}; see private output"
            )

    run(reference, "plain")
    run(candidate, "cut", expected_exit=CUT_EXIT)
    receipt = (root / "cut.stdout").read_text().strip()
    if (root / "cut/result.json").exists() or receipt != "sha256:" + sha(
        (root / "cut/observation-000002.json").read_bytes()
    ):
        raise ValueError("cut did not preserve exactly the selected observation")
    run(candidate, "resume", receipt)
    plain = (root / "plain/steps.jsonl").read_bytes()
    if (
        len(plain.splitlines()) != 4
        or plain
        != (root / "cut/steps.jsonl").read_bytes()
        + (root / "resume/steps.jsonl").read_bytes()
    ):
        raise ValueError("step report bytes differ")
    checkpoints = []
    for u in range(1, 5):
        name = f"checkpoint-{u:06d}.json"
        baseline = (root / "plain" / name).read_bytes()
        if baseline != (root / ("cut" if u <= 2 else "resume") / name).read_bytes():
            raise ValueError(f"checkpoint differs at update {u}")
        checkpoints.append(sha(baseline))
    reference_native, candidate_native, resumed_native = [
        (root / d / "native.sha256").read_text() for d in ("plain", "cut", "resume")
    ]
    if candidate_native != resumed_native:
        raise ValueError("candidate native library changed across processes")
    report = {
        "schema": "spiraltorch.native_corpus.observed_process_probe.v1",
        "status": "matched",
        "reference_native_sha256": reference_native,
        "candidate_native_sha256": candidate_native,
        "cut_exit": CUT_EXIT,
        "resumed_cursor": 2,
        "steps_sha256": sha(plain),
        "matched_checkpoint_sha256": checkpoints,
        "observation_id": receipt,
    }
    with (root / "report.json").open("x") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    return report


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(Path(sys.argv[3]), sys.argv[2], sys.argv[4])
    else:
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument("--reference-python", required=True, type=Path)
        p.add_argument("--candidate-python", required=True, type=Path)
        p.add_argument(
            "--output", required=True, type=Path, help="new private directory"
        )
        args = p.parse_args()
        print(
            json.dumps(
                compare(args.output, args.reference_python, args.candidate_python),
                indent=2,
            )
        )
