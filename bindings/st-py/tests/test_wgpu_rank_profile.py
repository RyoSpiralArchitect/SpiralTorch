"""Opt-in GPU timestamps: the same Rust report, never a CPU timing fallback."""

import math
import os

import pytest
import spiraltorch as st

pytestmark = pytest.mark.skipif(
    os.getenv("SPIRALTORCH_RUN_WGPU_TIMESTAMP_TESTS") != "1",
    reason="explicit GPU timestamp capability test",
)


def test_default_context_is_not_silently_upgraded():
    ordinary = st.WgpuRank("midk", 1, 257, 7)
    assert not ordinary.timestamp_queries_enabled
    ordinary.upload(st.Tensor(1, 257, [float(i) for i in range(257)]))
    with pytest.raises(RuntimeError, match="timestamp queries"):
        ordinary.profile()
    assert not ordinary.output_is_current
    ordinary.dispatch()
    assert ordinary.readback()["values"] == list(map(float, range(125, 132)))


@pytest.mark.parametrize("kind", ["topk", "midk", "bottomk"])
@pytest.mark.parametrize("tile", [32, 256, 512])
def test_profile_is_exact_and_uses_rust_stage_selection(kind, tile):
    rows, cols, k = 2, 8193, 65
    workspace = st.WgpuRank(kind, rows, cols, k, tile_cols=tile, timestamp_queries=True)
    assert workspace.timestamp_queries_enabled
    with pytest.raises(ValueError, match="upload"):
        workspace.profile()
    values = [float(i * 37 % 101 - 50) for i in range(rows * cols)]
    workspace.upload(st.Tensor(rows, cols, values))
    for invalid in (0, 1025):
        with pytest.raises(ValueError, match="repetitions"):
            workspace.profile(invalid)
    assert not workspace.output_is_current
    report = workspace.profile(16)
    assert workspace.output_is_current
    assert report["schema"] == "spiraltorch.rank_gpu_profile.v1"
    assert report["instrumented"] and report["generation"] == "1"
    assert report["repetitions"] == 16 and len(report["passes"]) == 32
    assert report["compute_submissions"] == 1 and not report["host_paced_chunks"]
    assert report["timestamp_period_ns"] > 0
    assert report["tile_cols"] == tile and report["kind"] == kind
    expected_entry = (
        "rankk_exact_2ce_midk_tournament"
        if kind == "midk" and 33 <= math.ceil(cols / tile) <= 256
        else "rankk_exact_2ce_row_merge"
    )
    assert report["merge_entry_point"] == expected_entry
    for index, item in enumerate(report["passes"]):
        assert item["repetition"] == index // 2
        assert item["stage"] == ("tile_sort" if index % 2 == 0 else "row_merge")
        assert isinstance(item["start_tick"], str) and isinstance(item["end_tick"], str)
        assert item["elapsed_ns"] == (int(item["end_tick"]) - int(item["start_tick"])) * report["timestamp_period_ns"]
    assert report["tile_sort_total_ns"] == sum(p["elapsed_ns"] for p in report["passes"][::2])
    assert report["row_merge_total_ns"] == sum(p["elapsed_ns"] for p in report["passes"][1::2])
    result = workspace.readback()
    ids, selected = [], []
    for row in range(rows):
        data = values[row * cols:(row + 1) * cols]
        ordered = sorted(range(cols), key=lambda i: (-data[i] if kind == "topk" else data[i], i))
        start = (cols - k) // 2 if kind == "midk" else 0
        chosen = ordered[start:start + k]
        ids.extend(chosen)
        selected.extend(data[i] for i in chosen)
    assert result == {"values": selected, "indices": ids, "generation": 1}
    assert not st.WgpuRank("topk", 1, 8, 2).timestamp_queries_enabled


def test_profiled_adaptation_does_not_credit_or_mutate_policy():
    base = st.plan("midk", 1, 8193, 65, backend="wgpu", strict_accelerator=True)
    session = st.RankAdaptationSession(base, ["u2: true; ctile: 256;"])
    before = session.snapshot()
    workspace = st.WgpuRank.from_adaptation(session, 0, timestamp_queries=True)
    workspace.upload(st.Tensor(1, 8193, [float(i) for i in range(8193)]))
    assert workspace.profile()["merge_entry_point"] == "rankk_exact_2ce_midk_tournament"
    assert session.snapshot() == before
