"""Exercise the post-review input gates on mixed-length real GPT-2 windows."""

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from transfer import (
    PLAN,
    guard_ml,
    load_base,
    location,
    read,
    require,
    save,
    sha,
    wheels,
)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("old", "drift", "out", "model", "wheel", "receipt"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    primary = read(HERE / "results.json")
    require(
        not a.receipt.exists()
        and sha(a.old / "data/train.json") == PLAN["train_sha256"]
        and sha(a.wheel)
        == "5896ed192e1328abd10bd6614a9ffc1776a6a7d44b9a2a496bd991a4232a60f8",
        "wrong inputs",
    )
    guard_ml()
    import spiraltorch as st

    inv = wheels.wheel_archive_inventory(a.wheel)
    package = wheels.installed_identity(inv)
    require(package["native_sha256"] == sha(st._rs.__file__), "wrong native extension")
    base = load_base(st, a.model)
    models = [base]
    for key in ("109-affine", "109-free"):
        case = primary["cases"][key]
        model = st.NativeCausalLm.from_training_checkpoint(
            base, location(a, key).read_text(), case["checkpoint_id"]
        )
        require(
            model.geometry_artifact_id == case["geometry_id"], "wrong placement owner"
        )
        models.append(model)
    train = read(a.old / "data/train.json")
    docs = [[], train[3][:1], train[11][:3], train[17][:40], train[31][:63]]
    labels = [
        [-100 if j % 3 == 0 or (i == 3 and j < 17) else tok for j, tok in enumerate(d)]
        for i, d in enumerate(docs)
    ]
    comp = st.NativeCausalLm.evaluate_variants(
        models, docs, 16, labels=labels, microbatch_windows=2
    )
    reports = [ev.to_json() for ev in comp.evaluations]
    require(
        comp.shared_prefix_layers == 8
        and reports
        == [
            m.evaluate_documents(
                docs, 16, labels=labels, microbatch_windows=2
            ).to_json()
            for m in models
        ],
        "masked scalar and shared reports differ",
    )
    first = comp.evaluations[0]
    require(
        first.skipped_windows > 0
        and first.ignored_targets > 0
        and package == wheels.installed_identity(inv),
        "probe coverage",
    )
    save(
        a.receipt,
        {
            "schema": "spiraltorch.native_lm.masked_variant_probe.v1",
            "engine_source": "c10b315cdf20ef99b2be04ff7e6f6e9a2eea903b",
            "wheel_sha256": package["wheel_sha256"],
            "native_sha256": package["native_sha256"],
            "states": ["base", "109-affine", "109-free"],
            "full_scalar_report_sha256": [
                hashlib.sha256(s.encode()).hexdigest() for s in reports
            ],
            "shared_prefix_layers": comp.shared_prefix_layers,
            "selected_targets": first.target_tokens,
            "ignored_targets": first.ignored_targets,
            "skipped_windows": first.skipped_windows,
            "data_id": first.data_id,
        },
    )
