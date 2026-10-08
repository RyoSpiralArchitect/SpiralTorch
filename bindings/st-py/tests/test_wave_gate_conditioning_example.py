import runpy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
example = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "examples"
        / "hf_wave_gate_conditioning.py"
    )
)


def test_corpus_footer_removed_before_disjoint_text_split_and_token_packing():
    body, offset = example["corpus_body"]("alpha\n\nbeta\n\ngamma<END>legal", "<END>")
    assert body == "alpha\n\nbeta\n\ngamma"
    assert offset == len(body)
    train, dev, cut = example["split_text"](body)
    assert train + dev == body
    assert train == body[:cut] and dev == body[cut:]
    with pytest.raises(ValueError):
        example["corpus_body"]("<END><END>", "<END>")
    with pytest.raises(ValueError):
        example["corpus_body"]("no marker", "<END>")
    packed = example["pack_tokens"](list(range(11)), 4)
    assert packed.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    with pytest.raises(ValueError):
        example["pack_tokens"]([1], 4)


def test_schedules_are_seed_paired_and_probe_has_no_duplicate_blocks():
    schedule = example["schedule"]
    first = schedule(41, 100, 32, 2)
    assert first == schedule(41, 100, 32, 2)
    assert first != schedule(43, 100, 32, 2)
    assert len(set(i for batch in first for i in batch)) == 64
    assert example["spaced_indices"](100, 4) == [0, 33, 66, 99]
    with pytest.raises(ValueError):
        example["spaced_indices"](3, 4)
    with pytest.raises(ValueError):
        schedule(41, 1, 2, 2)


def test_radius_protocol_pairs_initial_maps_and_only_adds_one_learnable_scalar():
    config = {"features": 3, "strength": 0.1}
    value = torch.tensor([[0.1, -0.2, 0.3]])
    for suffix in ("radius1", "radius4"):
        fixed = example["adapter_for"](f"wave_gate_{suffix}", config)
        learned = example["adapter_for"](f"wave_gate_learnable_{suffix}", config)
        assert sum(p.numel() for p in fixed.parameters()) == 6
        assert sum(p.numel() for p in learned.parameters()) == 7
        assert torch.equal(fixed.log_radius, learned.log_radius)
        with torch.no_grad():
            fixed.gate.fill_(0.3)
            learned.gate.fill_(0.3)
        assert torch.equal(fixed(value), learned(value))
