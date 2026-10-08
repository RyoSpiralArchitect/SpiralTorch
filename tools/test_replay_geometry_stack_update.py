import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("geometry_replay", Path(__file__).with_name("replay_geometry_stack_update.py"))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def fixture():
    record = {"step": 3, "loss_sha256": "fixed"}
    final = {"adapter": [1.], "optimizer": {"step": 3}, "rng": [23]}
    actual = {"start_cursor": 2, "records": [record], "gradients": [[.1, .2]],
              "final": final, "base_sha256": "base"}
    expected = {"public": {"gradients": [[0.], [.1], [.1, .2]], "final": copy.deepcopy(final)}}
    report = {"config": {"checkpoint_after": 2}, "public_records": [{}, {}, copy.deepcopy(record)],
              "base_sha256": "base"}
    return SimpleNamespace(equal=lambda a, b: a == b), actual, expected, report


def test_matching_single_update_is_accepted():
    replay.compare_update(*fixture())


@pytest.mark.parametrize("field", ["start_cursor", "records", "gradients", "final", "base_sha256"])
def test_missing_or_changed_evidence_is_rejected(field):
    proof, actual, expected, report = fixture()
    actual[field] = {"start_cursor": 1, "records": [], "gradients": [[.1, .21]],
                     "final": {"adapter": [2.]}, "base_sha256": "changed"}[field]
    with pytest.raises(ValueError):
        replay.compare_update(proof, actual, expected, report)


@pytest.mark.parametrize("captured", [False, True])
def test_transport_budget_is_explicit_and_rejects_extra_reuploads(captured):
    report = {"shape": [2, 128, 768], "config": {"features": 768}}
    shape = report["shape"]
    calls = [shape, [768], [768]] + ([shape] * 3 if captured else []) + [shape]
    replay.check_transport(calls, report, captured)
    for bad in (calls[:-1], calls + [shape], calls + [shape, shape]):
        with pytest.raises(ValueError, match="sequence"):
            replay.check_transport(bad, report, captured)
