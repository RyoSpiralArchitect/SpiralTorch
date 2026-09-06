"""Kernel reports must name the same exact-rank pipeline as the Rust executor."""

import pytest
import spiraltorch as st

pytestmark = pytest.mark.skipif(
    not st.wgpu_kernel_reports_available(), reason="requires native WGPU report bindings"
)


@pytest.mark.parametrize("tiles", [32, 33, 255, 256, 257])
@pytest.mark.parametrize("k", [1, 2, 65])
def test_midk_report_matches_merge_boundary(tiles, k):
    report = st.wgpu_rank_kernel_report(
        "midk",
        2,
        tiles * 256 - 1,
        k,
        use_two_stage=True,
        rank_tile=32,
        compaction_tile=256,
    )
    expected = (
        "rankk_exact_2ce_midk_tournament"
        if 33 <= tiles <= 256 and k > 1
        else "rankk_exact_2ce_row_merge"
    )
    assert report["primary"]["name"] == expected
    assert report["primary"]["entry_point"] == expected
    assert report["stages"] == ["tile_sort", "row_merge"]


def test_catalog_exposes_tournament_descriptor():
    descriptor = st.wgpu_kernel_descriptor("rankk_exact_2ce_midk_tournament.wgsl")
    assert descriptor is not None
    assert descriptor["entry_point"] == "rankk_exact_2ce_midk_tournament"
    assert descriptor["pipeline_label"] == "st.rankk.exact_2ce.row_merge_tournament"
    assert descriptor["portable"] and not descriptor["subgroup"]


def test_rank_plan_report_uses_rust_candidate_selection():
    base = st.plan("midk", 2, 8193, 65, backend="wgpu", strict_accelerator=True)
    session = st.RankAdaptationSession(
        base,
        ["u2: true; ctile: 256;", "u2: true; ctile: 512;"],
        policy="ucb",
        seed=17,
    )
    selection = session.choose()
    report = st.wgpu_kernel_report_from_rank_plan(selection.plan)
    tile = report["request"]["compaction_tile"]
    assert tile in (256, 512)
    expected = (
        "rankk_exact_2ce_midk_tournament"
        if tile == 256
        else "rankk_exact_2ce_row_merge"
    )
    assert report["primary"]["name"] == expected
    session.abandon(selection.selection_id)
