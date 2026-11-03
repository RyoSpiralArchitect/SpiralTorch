"""Native scoring plus independent, model-free verification of every returned row."""

import json
import math

from control_contract import (
    BOOKS,
    CASES,
    PARENT,
    STATES,
    contrasts,
    digest,
    read,
    require,
    sha,
)
from joint_protocol import save
from freeze_binding import section_identity


def owners(case, mode, fits, parent):
    v = (fits if mode == "fixed" else parent["fits"])[case]
    return (v["lora_id"], v["geometry_id"])


def verify_section(key, raw, fits, parent, group):
    book, section = key.rsplit("-", 1)
    common = group["common"]
    require(
        raw.get("schema") == "spiraltorch.native_causal_lm_evaluation_comparison.v1"
        and raw["shared_prefix_layers"] == 8
        and len(raw["evaluations"]) == len(STATES),
        "native comparison envelope",
    )
    result = {}
    for name, row in zip(STATES, raw["evaluations"], strict=True):
        case, mode = name.split(":")
        require(
            set(row)
            == set(common)
            | {"lora_artifact_id", "geometry_artifact_id", "nll", "perplexity"}
            and digest({k: row[k] for k in common}) == digest(common)
            and (row["lora_artifact_id"], row["geometry_artifact_id"])
            == owners(case, mode, fits, parent),
            "score has wrong section, coverage, model or owners",
        )
        nll = row["nll"]
        require(
            type(nll) is float
            and math.isfinite(nll)
            and type(row["perplexity"]) is float
            and math.isclose(math.exp(nll), row["perplexity"], rel_tol=1e-14),
            "invalid native NLL/perplexity",
        )
        if mode == "joint":
            require(
                nll == parent["section_nll"][book][section][case],
                "joint anchor differs",
            )
        result[name] = nll
    return result


def score(st, base, a, fits):
    parent = read(PARENT / "results.json")
    models = []
    for name in STATES:
        case, mode = name.split(":")
        ref = (fits if mode == "fixed" else parent["fits"])[case]
        ck = (a.run if mode == "fixed" else a.out) / case / "checkpoint-512.json"
        model = st.NativeCausalLm.from_session_checkpoint_at(
            base,
            ck.read_text(),
            ref["checkpoint_id"],
            update=512,
            recorded_data_id=ref["data_id"],
            post_lora_id=ref["lora_id"],
            post_geometry_id=ref["geometry_id"],
        )
        models.append(model)
    groups = read(PARENT / "score_roster.json")["groups"]
    mapping = read(PARENT / "input_binding.json")["sections"]
    windows = read(a.data / "windows.json")
    folder = a.run / "native-scores"
    folder.mkdir(mode=0o700)
    pins = {}
    first = "mansfield-5"
    for key in [first, *sorted(set(mapping) - {first})]:
        book, section = key.rsplit("-", 1)
        docs = windows[book][section]
        require(
            section_identity(docs) == mapping[key], "different evaluation documents"
        )
        ev = st.NativeCausalLm.evaluate_variants(models, docs, 64, microbatch_windows=2)
        raw = json.loads(ev.to_json())
        verify_section(key, raw, fits, parent, groups[key])
        if key == first:
            scalar = {
                name: json.loads(
                    m.evaluate_documents(docs, 64, microbatch_windows=2).to_json()
                )
                for m, name in zip(models, STATES, strict=True)
                if name.endswith(":fixed")
            }
            check_scalar(raw, scalar)
            save(a.run / "scalar.json", scalar)
        path = folder / (key + ".json")
        save(path, raw)
        pins[key] = sha(path)
    require(set(pins) == set(groups) == set(mapping), "incomplete native scores")
    return {
        "native_scores": pins,
        "scalar_sha256": sha(a.run / "scalar.json"),
        "fitted_sha256": sha(a.run / "fitted.json"),
    }


def check_scalar(raw, scalar):
    require(
        scalar
        == {
            name: row
            for name, row in zip(STATES, raw["evaluations"], strict=True)
            if name.endswith(":fixed")
        },
        "scalar/batched frozen reports differ",
    )


def summarize(fits, reports, scalar):
    parent = read(PARENT / "results.json")
    groups = read(PARENT / "score_roster.json")["groups"]
    require(
        set(fits) == set(CASES) and set(reports) == set(groups),
        "missing fit or section",
    )
    sections = {b: {} for b in BOOKS}
    for key, raw in reports.items():
        book, i = key.rsplit("-", 1)
        sections[book][i] = verify_section(key, raw, fits, parent, groups[key])
    check_scalar(reports["mansfield-5"], scalar)
    books = contrasts(sections)
    return {
        "schema": "spiraltorch.native_lm.frozen_attention_effects.v1",
        "books": books,
        "primary_all_wave_joint_better": books["mansfield"]["wave_learning_wins"] == 3,
        "new_fits": 6,
        "updates": 3072,
        "training_targets": 193536,
        "process_resumes": 6,
        "target_tokens_per_book_per_state": 4536,
        "joint_anchor_nlls": 108,
        "new_fixed_nlls": 108,
        "scalar_parity_reports": 6,
    }
