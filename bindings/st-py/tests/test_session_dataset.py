import pytest


try:
    import spiraltorch
except ModuleNotFoundError:  # pragma: no cover - exercised when extension missing
    pytest.skip(
        "spiraltorch module unavailable; dataset helpers require the native build",
        allow_module_level=True,
    )
except ImportError as exc:  # pragma: no cover - exercised when extension import fails
    pytest.skip(
        f"spiraltorch import failed ({exc}); dataset helpers require the native build",
        allow_module_level=True,
    )


pytestmark = pytest.mark.skipif(
    not hasattr(spiraltorch, "dataset")
    or not hasattr(spiraltorch.dataset, "Dataset"),
    reason="spiraltorch dataset helpers require the native extension",
)


def _list_samples():
    return [
        ([[0.0]], [[1.0]]),
        ([[1.0]], [[0.0]]),
        ([[2.0]], [[1.0]]),
    ]


def _generator_samples():
    for index in range(3):
        yield [[float(index)]], [[float(index) + 0.5]]


def test_session_dataset_coerces_generator_samples():
    session = spiraltorch.SpiralSession()
    dataset = session.dataset(_generator_samples())
    samples = dataset.samples()
    assert len(samples) == 3
    for idx, (inp, target) in enumerate(samples):
        assert isinstance(inp, spiraltorch.Tensor)
        assert isinstance(target, spiraltorch.Tensor)
        assert inp.tolist() == [[float(idx)]]
        assert target.tolist() == [[float(idx) + 0.5]]


def test_session_dataloader_accepts_iterable_samples():
    session = spiraltorch.SpiralSession()
    loader = session.dataloader(_list_samples(), batch_size=2, prefetch=1, shuffle=True)
    batches = list(loader)
    assert len(batches) == 2
    for input_batch, target_batch in batches:
        assert isinstance(input_batch, spiraltorch.Tensor)
        assert isinstance(target_batch, spiraltorch.Tensor)
        assert input_batch.shape()[1] == 1
        assert target_batch.shape()[1] == 1
