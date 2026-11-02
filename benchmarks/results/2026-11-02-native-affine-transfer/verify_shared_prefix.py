"""Independent backend check after primary scoring. Never train or choose an arm."""

import argparse
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from transfer import (
    BOOKS,
    FORBIDDEN,
    NAMES,
    arm,
    guard_ml,
    load_base,
    location,
    paths,
    read,
    require,
    save,
    sha,
    wheels,
)


def verify(a):
    spec = read(HERE / "shared_prefix_plan.json")
    require(
        a.engine_sha256 == sha(a.wheel) == spec["comparison_wheel_sha256"],
        "wrong comparison wheel",
    )
    completed = read(a.out / "status.json")
    require(
        completed["status"] == "complete"
        and completed["results_sha256"] == sha(a.out / "results.json") == a.result_pin,
        "primary scoring must already be sealed",
    )
    result = read(a.out / "results.json")
    require(
        result["source_wheel_sha256"] != a.engine_sha256
        and sha(a.out / "scored.json") == result["scores_sha256"]
        and sha(a.out / "declaration.json") == result["declaration_sha256"],
        "wrong primary evidence",
    )
    decl = read(a.out / "declaration.json")
    require(
        sha(a.data / "windows.json") == decl["windows_sha256"],
        "scoring windows changed",
    )
    inputs = read(a.data / "windows.json")
    guard_ml()
    import spiraltorch as st

    inv = wheels.wheel_archive_inventory(a.wheel)
    pkg = wheels.installed_identity(inv)
    require(
        pkg["wheel_sha256"] == a.engine_sha256
        and pkg["native_sha256"]
        == sha(st._rs.__file__)
        == spec["comparison_native_sha256"]
        and pkg["all_original_files_match"],
        "installed engine is not the declared wheel",
    )
    base = load_base(st, a.model)
    models = [
        base,
        base.with_geometry(arm(st, base, a, "affine", decl["initial"])),
        base.with_geometry(arm(st, base, a, "learned4", decl["initial"])),
    ]
    scores = read(a.out / "scored.json")
    references = []
    for i, name in enumerate(NAMES):
        f = a.out / "scores" / (name + ".json")
        require(sha(f) == scores["files"][name], "scalar receipt changed")
        ref = read(f)
        if i >= 3:
            state = result["cases"][name]
            models.append(
                st.NativeCausalLm.from_training_checkpoint(
                    base, location(a, name).read_text(), state["checkpoint_id"]
                )
            )
        require(
            ref["geometry_id"] == models[i].geometry_artifact_id
            and ref["model"] == name,
            "variant order or identity changed",
        )
        references.append(ref)
    checked = 0
    columns = []
    for book in BOOKS:
        for k, docs in inputs[book].items():
            comp = st.NativeCausalLm.evaluate_variants(
                models, docs, 64, microbatch_windows=2
            )
            require(
                comp.shared_prefix_layers == 8 and len(comp.evaluations) == len(NAMES),
                "unexpected shared depth or results",
            )
            for name, ev, ref in zip(NAMES, comp.evaluations, references, strict=True):
                row = ref["scores"][book]["sections"][k]
                require(
                    ev.data_id == row["data_id"]
                    and ev.target_tokens == row["targets"] == 756
                    and ev.nll == row["nll"]
                    and ev.window_count == 12
                    and ev.microbatches == 6
                    and ev.ignored_targets == ev.skipped_windows == 0,
                    f"scalar/combined parity differs: {name}/{book}/{k}",
                )
                checked += 1
            columns.append(f"{book}/{k}")
    require(
        checked == 270
        and not FORBIDDEN.intersection(m.partition(".")[0] for m in sys.modules)
        and wheels.installed_identity(inv) == pkg,
        "incomplete or changed runtime",
    )
    return {
        "schema": "spiraltorch.native_lm.shared_prefix_validation.v1",
        "engine_source": spec["comparison_engine_source"],
        "plan_sha256": sha(HERE / "shared_prefix_plan.json"),
        "primary_result_sha256": a.result_pin,
        "scalar_wheel_sha256": result["source_wheel_sha256"],
        "comparison_wheel_sha256": a.engine_sha256,
        "comparison_native_sha256": pkg["native_sha256"],
        "window_set_sha256": decl["windows_sha256"],
        "variants": NAMES,
        "section_batches": columns,
        "shared_prefix_layers": 8,
        "decoder_layers": 12,
        "exact_nll_targets_and_data_id": checked,
        "plain_nll_targets_per_state": 13608,
        "new_training_updates": 0,
        "timing_claim": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    paths(p)
    for arg in ("result-pin", "engine-sha256"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    os.umask(0o077)
    require(not a.receipt.exists(), "do not overwrite a prior result")
    save(a.receipt, verify(a))
