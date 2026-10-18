"""Deterministic public view of private outcomes, without scalar-weight maxima."""

import copy
import re


def public_results(private, source_id):
    if (private["schema"] != "spiraltorch.hf_learned_radius_results.v1"
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", source_id)
            or "publication" in private):
        raise ValueError("expected original learned-radius results and retained content ID")
    result = copy.deepcopy(private)
    for run in result["runs"]:
        for site in run["learned_sites"]:
            # A dominant scalar radius can be recovered from its exact maximum.
            del site["parameter_delta_max_abs"]
    result["publication"] = {
        "schema": "spiraltorch.hf_learned_radius_publication.v1",
        "private_source_results_id": source_id,
        "projection": "omit_per_site_parameter_delta_max_abs_from_every_run",
        "reason": "a_per_site_maximum_can_reveal_an_individual_learned_scalar_weight",
    }
    return result
