import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import spiraltorch as st


@pytest.fixture
def benchmark_module():
    source = Path(__file__).with_name("benchmark_fractional_learning.py")
    spec = importlib.util.spec_from_file_location("fractional_benchmark_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("alpha", [.5, 1., 1.5])
@pytest.mark.parametrize("time", [1, 5])
def test_same_math_routes_match_without_timing(benchmark_module, history, alpha, time):
    torch.set_num_threads(2)
    value = torch.linspace(-.8, .9, 6*time).reshape(2, 3, time).transpose(1, 2)
    upstream = value.cos()
    options = {"alpha": alpha, "kernel_len": 4, "step": .7, "history": history,
               "kernel": st.FractionalGlKernel(kernel_len=4, step=.7)}
    reference = benchmark_module.run_route("rust_joint", value, upstream, **options)
    for name in benchmark_module.ROUTES:
        actual = benchmark_module.run_route(name, value, upstream, **options)
        benchmark_module.check_result(actual, reference, exact=name != "torch_conv1d")


def test_reference_rejects_mismatches_and_nonfinite_results(benchmark_module):
    good = (torch.tensor([1.]), torch.tensor(2.))
    with pytest.raises(ValueError, match="differ"):
        benchmark_module.check_result((torch.tensor([1.1]), good[1]), good, exact=True)
    with pytest.raises(AssertionError):
        benchmark_module.check_result((torch.tensor([1.1]), good[1]), good, exact=False)
    with pytest.raises(ValueError, match="nonfinite"):
        benchmark_module.check_result((good[0], torch.tensor(float("nan"))), good, exact=False)
