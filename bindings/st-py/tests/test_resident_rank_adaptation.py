"""Rust-owned candidate geometry, real resident dispatch, and measured feedback."""
import os
from time import perf_counter

import pytest
import spiraltorch as st

pytestmark = pytest.mark.skipif(
    not os.getenv("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS"), reason="explicit GPU test"
)


@pytest.mark.parametrize("kind", ["topk", "midk", "bottomk"])
@pytest.mark.parametrize("policy", ["ucb", "thompson_sampling"])
def test_candidates_dispatch_without_rebuilding_plan_in_python(kind, policy):
    base = st.plan(kind, 1, 257, 7, backend="wgpu", strict_accelerator=True)
    session = st.RankAdaptationSession(
        base, [f"u2: true; rank_tile: {tile}; ctile: {tile};" for tile in [32, 128, 512]],
        policy=policy, seed=17,
    )
    before = session.snapshot()
    workspaces = [st.WgpuRank.from_adaptation(session, index) for index in range(3)]
    assert session.snapshot() == before
    assert [ws.tile_cols for ws in workspaces] == [32, 128, 257]
    values = [float(i * 37 % 101 - 50) for i in range(257)]
    ids = sorted(range(257), key=lambda i: (-values[i] if kind == "topk" else values[i], i))
    start = 125 if kind == "midk" else 0
    ids = ids[start:start + 7]
    expected = {"values": [values[i] for i in ids], "indices": ids, "generation": 1}
    for ws in workspaces:
        assert ws.shape == (1, 257, 7) and ws.kind == kind
        ws.upload(st.Tensor(1, 257, values))
        ws.dispatch()
        assert ws.readback() == expected
    for _ in range(12):
        selection = session.choose()
        ws = workspaces[selection.candidate_index]
        assert selection.execution_signature.endswith(f"/tile={ws.tile_cols}")
        ws.synchronize()
        start_time = perf_counter()
        ws.dispatch(16)
        ws.synchronize()
        elapsed_ms = (perf_counter() - start_time) * 1000
        assert ws.readback() == expected
        receipt = session.observe(selection.selection_id, elapsed_ms, True)
        assert receipt["credited"] and receipt["elapsed_ms"] == elapsed_ms
        assert receipt["reward"] == 1 / (1 + elapsed_ms)
    assert sum(session.snapshot()["observation_counts"]["rank_plan_variant"].values()) == 12
    assert session.pending_selection_id is None


def test_invalid_factory_requests_do_not_choose_or_credit_a_candidate():
    base = st.plan("topk", 1, 256, 8, backend="wgpu", strict_accelerator=True)
    session = st.RankAdaptationSession(base, ["u2: false;", "u2: true;"])
    before = session.snapshot()
    for index in [0, 2, 2**31]:
        with pytest.raises(ValueError):
            st.WgpuRank.from_adaptation(session, index)
        assert session.snapshot() == before
    with pytest.raises((TypeError, OverflowError)):
        st.WgpuRank.from_adaptation(session, -1)
    with pytest.raises(TypeError):
        st.WgpuRank.from_adaptation(session, 0.5)
    with pytest.raises(TypeError):
        st.WgpuRank.from_adaptation(object(), 0)
