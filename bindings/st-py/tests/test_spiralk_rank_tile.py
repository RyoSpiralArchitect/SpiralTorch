"""The public hint constructor must accept the same rank knob as Rust's DSL."""
import pytest
import spiraltorch.spiralk as sk


def context():
    return sk.SpiralKContext(2, 257, 7, False, 32, 256, 512, 2, 1)


def test_hard_rank_tile_does_not_replace_fft_or_compaction():
    result = context().eval("rank_tile: 64; tile_cols: 512; ctile: 128;")
    assert result["hard"]["rank_tile"] == 64
    assert result["hard"]["tile_cols"] == 512
    assert result["hard"]["compaction_tile"] == 128


def test_public_hint_synthesizes_a_real_rank_soft_rule():
    hint = sk.SpiralKHeuristicHint("rank_tile", "128", 0.75, "true")
    assert hint.field == "rank_tile"
    source = sk.synthesize_program("rank_tile: 64;", [hint])
    result = context().eval(source)
    assert result["hard"]["rank_tile"] == 64
    assert result["soft"] == [{"field": "rank_tile", "value": 128, "weight": 0.75}]


@pytest.mark.parametrize("value", ["-1", "0", "1.5", "4294967296", "true"])
def test_invalid_rank_tile_is_rejected_by_rust(value):
    with pytest.raises(ValueError, match="rank_tile"):
        context().eval(f"rank_tile: {value};")
