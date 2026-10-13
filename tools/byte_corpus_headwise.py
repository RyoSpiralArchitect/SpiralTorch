"""Independent validation and Torch mathematics for fixed two-head controls.

This module does not prepare production requests or fit gains.
"""
import copy
import itertools
import math
import struct
import importlib.util
from pathlib import Path

P, E, F = "poincare_squared.v1", "euclidean_chord_squared.v1", "categorical_fisher_rao_squared.v1"
ROUTES = list(itertools.permutations((P, E, F), 2))


def study():
    spec = importlib.util.spec_from_file_location("byte_corpus_study", Path(__file__).with_name("byte_corpus_study.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_request(request):
    spec = request.get("headwise_controls")
    if (type(spec) is not dict or set(spec) != {"source_request_sha256"}
            or type(spec["source_request_sha256"]) is not str or len(spec["source_request_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in spec["source_request_sha256"])
            or request["config"]["heads"] != 2 or len(request["config"]["blocks"]) != 1):
        raise ValueError("invalid one-block/two-head source contract")
    cases = request["cases"]
    if not 21 <= len(cases) <= 63 or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("invalid headwise case count/names")
    homogeneous = copy.deepcopy(request)
    homogeneous["schema"] = "spiraltorch.byte_corpus.request.v5"
    del homogeneous["headwise_controls"]
    homogeneous["cases"] = [c for c in homogeneous["cases"] if "head_pair_metrics" not in c]
    study().request_version(homogeneous)
    seeds = {c["seed"] for c in homogeneous["cases"]}
    if any(type(c["seed"]) is not int or c["seed"] not in seeds for c in cases):
        raise ValueError("unknown headwise seed")
    def bits(parameters):
        for p in parameters:
            if (type(p["shape"]) is not list or not p["shape"]
                    or any(type(d) is not int or d <= 0 for d in p["shape"])
                    or type(p["values"]) is not list or math.prod(p["shape"]) != len(p["values"])
                    or any(type(v) not in (float, int) or not math.isfinite(v) for v in p["values"])):
                raise ValueError("invalid headwise parameter shape or value")
        return [(p["name"], p["shape"], [struct.pack("<f", v) for v in p["values"]]) for p in parameters]
    for seed in seeds:
        group = [c for c in cases if c["seed"] == seed]
        if len(group) != 21:
            raise ValueError("incomplete headwise seed")
        sources = {metric: next(c for c in homogeneous["cases"] if c["seed"] == seed
                   and c.get("pair_metric") == metric and c["geometry_update"] == "train"
                   and c["bias_initialization"] == ("original" if metric == P else "matched_poincare_rms"))
                   for metric in (P, E, F)}
        for route in ROUTES:
            expected = copy.deepcopy(sources[P]["parameters"])
            selected = [next(p for p in sources[m]["parameters"] if p["name"] == "geometry.raw_gain.0")["values"][h]
                        for h, m in enumerate(route)]
            next(p for p in expected if p["name"] == "geometry.raw_gain.0")["values"] = selected
            for update in ("train", "frozen"):
                found = [c for c in group if c.get("head_pair_metrics") == [list(route)]
                         and c.get("geometry_update") == update]
                if (len(found) != 1 or found[0]["geometry"] is not True or "pair_metric" in found[0]
                        or found[0].get("bias_initialization") != "matched_poincare_rms"
                        or bits(found[0]["parameters"]) != bits(expected)):
                    raise ValueError("missing/duplicate route or changed initial parameter bits")


def pair_bias(coordinates, gains, route, curvature, poincare, fisher):
    import torch
    import torch.nn.functional as functional
    scores = []
    for head, metric in enumerate(route):
        gain = gains[head:head+1]
        if metric == E:
            square = (coordinates[:, :, None, :] - coordinates[:, None, :, :]).square().sum(-1)
            scores.append((-functional.softplus(gain)[None, :, None, None] * (4 * square[:, None])).tril())
        elif metric == F:
            if fisher is None:
                raise ValueError("missing Fisher reference")
            scores.append(fisher(coordinates, gain))
        elif metric == P:
            scores.append(poincare(coordinates, gain, curvature))
        else:
            raise ValueError("unknown head metric")
    return torch.cat(scores, dim=1)


def gain_deltas(initial, actual, expected):
    rows = []
    for head, (before, got, ref) in enumerate(zip(initial, actual, expected)):
        norm = abs(ref - before)
        relative = abs(got - ref) / norm if norm else math.inf
        if norm <= 1e-8 or got == before or not math.isfinite(relative) or relative > .002:
            raise ValueError("unqualified individual head gain learning delta")
        rows.append(dict(head=head, reference_delta=ref-before, actual_delta=got-before, relative_error=relative))
    return rows


def contrasts(summaries):
    rows = []
    for seed in sorted({c["seed"] for c in summaries}):
        group = [c for c in summaries if c["seed"] == seed]
        for route in ROUTES:
            pair = [next(c for c in group if c.get("head_pair_metrics") == [list(route)]
                         and c["geometry_update"] == update) for update in ("train", "frozen")]
            def differences(key):
                result = {"learned_minus_frozen_bpb": pair[0][key] - pair[1][key]}
                for metric, label in ((P, "poincare"), (E, "calibrated_flat"), (F, "fisher")):
                    for c, update in zip(pair, ("train", "frozen")):
                        base = next(b for b in group if b.get("pair_metric") == metric
                                    and b["geometry_update"] == update
                                    and b["bias_initialization"] == ("original" if metric == P else "matched_poincare_rms"))
                        result[f"{update}_minus_{label}_bpb"] = c[key] - base[key]
                for c, update in zip(pair, ("train", "frozen")):
                    reverse = next(b for b in group if b.get("head_pair_metrics") == [list(reversed(route))]
                                   and b["geometry_update"] == update)
                    result[f"{update}_minus_reversed_bpb"] = c[key] - reverse[key]
                return result
            measured, reference = differences("final_heldout_bpb"), differences("reference_final_heldout_bpb")
            rows.append(dict(seed=seed, head_pair_metrics=[list(route)], measured=measured, reference=reference,
                absolute_torch_discrepancy={k:abs(v-reference[k]) for k,v in measured.items()},
                same_sign_as_torch={k:((v>0)-(v<0)) == ((reference[k]>0)-(reference[k]<0)) for k,v in measured.items()}))
    return rows
