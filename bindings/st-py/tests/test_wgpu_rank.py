"""Packaged resident rank execution; opt in on hosts with a real WGPU adapter."""
import math
import os

import pytest
import spiraltorch as st


def test_rank_surface_is_shared_with_wgpu_module():
    assert st.WgpuRank is st.wgpu.WgpuRank


@pytest.mark.skipif(not os.getenv("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS"), reason="explicit GPU test")
@pytest.mark.parametrize("kind", ["topk", "midk", "bottomk"])
def test_readback_reuse_keeps_previous_results_owned(kind):
    source = st.WgpuMatmul(1, 1, 8)
    rank = st.WgpuRank(kind, 1, 8, 3)
    source.upload(st.Tensor(1, 1, [1.]), st.Tensor(1, 8, [0.] * 8))
    retained = []
    for scale in range(1, 5):
        values = [scale * v for v in [3., 1., 3., -2., 7., 4., 0., 1.]]
        source.upload_rhs(st.Tensor(1, 8, values))
        generation = rank.dispatch_from_matmul(source)
        ids = sorted(range(8), key=lambda i: (-values[i] if kind == "topk" else values[i], i))
        start = 2 if kind == "midk" else 0
        ids = ids[start:start + 3]
        retained.append((source.readback(), rank.readback(), values, ids, generation))
    del source, rank
    for matrix, selected, values, ids, generation in retained:
        assert matrix.tolist() == [values]
        assert selected == {"values": [values[i] for i in ids], "indices": ids, "generation": generation}


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


@pytest.mark.skipif(not os.getenv("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS"), reason="explicit GPU test")
@pytest.mark.parametrize("kind", ["topk", "midk", "bottomk"])
@pytest.mark.parametrize("cols,tile", [(1024, 1024), (1025, 1024), (1025, 1025), (2049, 2048)])
def test_shared_and_storage_sort_boundaries(kind, cols, tile):
    values = [float(i * 37 % 101 - 50) for i in range(cols)]
    values[-1] = math.nan
    ids = sorted(range(cols - 1), key=lambda i: (-values[i] if kind == "topk" else values[i], i))
    start = (len(ids) - 7) // 2 if kind == "midk" else 0
    expected_ids = ids[start:start + 7]
    ws = st.WgpuRank(kind, 1, cols, 7, tile_cols=tile)
    ws.upload(st.Tensor(1, cols, values))
    ws.dispatch(2)
    assert ws.readback() == {
        "values": [values[i] for i in expected_ids],
        "indices": expected_ids,
        "generation": 1,
    }


@pytest.mark.skipif(not os.getenv("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS"), reason="explicit GPU test")
@pytest.mark.parametrize("kind", ["topk", "midk", "bottomk"])
def test_device_local_matmul_copy_is_owned_and_transactional(kind):
    import gc
    values = [3., 1., 3., -2., 7., 4., 0., 1., -1., 8., 0., 2., 2., 3., 9., -2.]
    expected_ids = []
    for row in range(2):
        ids = sorted(range(8), key=lambda i: (-values[row*8+i] if kind == "topk" else values[row*8+i], i))
        start = 2 if kind == "midk" else 0
        expected_ids.extend(ids[start:start+3])
    expected_values = [values[(i//3)*8+index] for i,index in enumerate(expected_ids)]
    source = st.WgpuMatmul(2,2,8)
    rank = st.WgpuRank(kind,2,8,3)
    with pytest.raises(ValueError):
        rank.set_input_from_matmul(source)
    source.upload(st.Tensor(2,2,[1.,0.,0.,1.]),st.Tensor(2,8,values))
    source.dispatch()
    rank.set_input_from_matmul(source)
    rank.dispatch()
    generation = rank.generation
    wrong = st.WgpuMatmul(1,1,16)
    wrong.upload(st.Tensor(1,1,[1.]),st.Tensor(1,16,[99.]*16))
    wrong.dispatch()
    with pytest.raises(ValueError):
        rank.set_input_from_matmul(wrong)
    with pytest.raises(TypeError):
        rank.set_input_from_matmul(object())
    source.upload_rhs(st.Tensor(2,8,[99.]*16))
    with pytest.raises(ValueError):
        rank.set_input_from_matmul(source)
    assert rank.output_is_current and rank.generation == generation
    source.dispatch()
    del source
    gc.collect()
    rank.dispatch(2)
    assert rank.readback() == {"values":expected_values,"indices":expected_ids,"generation":generation}


@pytest.mark.skipif(not os.getenv("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS"), reason="explicit GPU test")
@pytest.mark.parametrize("kind", ["topk", "midk", "bottomk"])
@pytest.mark.parametrize("kernel,accumulation", [
    ("scalar", "sequential"), ("register_2x2", "tiled"), ("register_2x2", "compensated"),
])
def test_single_submit_matmul_rank_chain(kind, kernel, accumulation):
    source = st.WgpuMatmul(1, 2, 8, kernel=kernel, accumulation=accumulation)
    rank = st.WgpuRank(kind, 1, 8, 3)
    with pytest.raises(ValueError):
        rank.dispatch_from_matmul(source)
    assert not source.output_is_current and rank.generation == 0
    values = [3., 1., 3., -2., 7., 4., 0., 1.]
    source.upload(st.Tensor(1, 2, [1., 0.]), st.Tensor(2, 8, values + [0.] * 8))
    for repetitions in [0, 1025]:
        with pytest.raises(ValueError):
            rank.dispatch_from_matmul(source, repetitions)
    with pytest.raises(TypeError):
        rank.dispatch_from_matmul(object())
    assert not source.output_is_current and rank.generation == 0
    assert rank.dispatch_from_matmul(source, 3) == 1
    assert source.output_is_current and rank.output_is_current
    ids = sorted(range(8), key=lambda i: (-values[i] if kind == "topk" else values[i], i))
    start = 2 if kind == "midk" else 0
    ids = ids[start:start+3]
    assert rank.readback() == {"values": [values[i] for i in ids], "indices": ids, "generation": 1}
    wrong = st.WgpuMatmul(2, 1, 4)
    wrong.upload(st.Tensor(2, 1, [1., 1.]), st.Tensor(1, 4, [99.] * 4))
    with pytest.raises(ValueError):
        rank.dispatch_from_matmul(wrong)
    assert not wrong.output_is_current and rank.generation == 1
    source.upload_rhs(st.Tensor(2, 8, [99.] * 16))
    assert rank.dispatch_from_matmul(source) == 2
    source.upload_rhs(st.Tensor(2, 8, [-9.] * 16))
    source.dispatch()
    del source
    rank.dispatch()
    assert rank.readback() == {"values": [99.] * 3, "indices": list(range(start, start+3)), "generation": 2}
