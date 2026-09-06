"""Packaged resident rank execution; opt in on hosts with a real WGPU adapter."""
import math
import os

import pytest
import spiraltorch as st


def test_rank_surface_is_shared_with_wgpu_module():
    assert st.WgpuRank is st.wgpu.WgpuRank


@pytest.mark.skipif(not os.getenv("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS"), reason="explicit GPU test")
@pytest.mark.parametrize("kind", ["topk", "midk", "bottomk"])
@pytest.mark.parametrize("tile", [3, 8, 256])
def test_rank_lifecycle_and_exact_ties(kind, tile):
    values = [3.0, 1.0, 3.0, -2.0, 7.0, 4.0, 0.0, 1.0]
    ids = sorted(range(8), key=lambda i: (-values[i] if kind == "topk" else values[i], i))
    start = 2 if kind == "midk" else 0
    expected_ids = ids[start:start + 3]
    ws = st.WgpuRank(kind, 1, 8, 3, tile_cols=tile)
    assert ws.shape == (1, 8, 3)
    assert ws.tile_cols == min(tile, 8)
    with pytest.raises(ValueError):
        ws.dispatch()
    with pytest.raises(ValueError):
        ws.readback()
    ws.upload(st.Tensor(1, 8, values))
    for n in [0, 1025]:
        with pytest.raises(ValueError):
            ws.dispatch(n)
    assert ws.dispatch(3) == 1
    ws.synchronize()
    result = ws.readback()
    assert result == {"values": [values[i] for i in expected_ids], "indices": expected_ids, "generation": 1}
    with pytest.raises(ValueError):
        ws.upload(st.Tensor(2, 4, values))
    assert ws.output_is_current
    ws.upload(st.Tensor(1, 8, [math.nan] * 8))
    assert not ws.output_is_current
    with pytest.raises(ValueError):
        ws.readback()
    ws.dispatch()
    result = ws.readback()
    assert result["generation"] == 2
    assert result["indices"] == [-1] * 3
    assert all(math.isnan(v) for v in result["values"])
