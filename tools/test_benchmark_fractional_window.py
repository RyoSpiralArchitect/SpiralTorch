import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import spiraltorch as st


@pytest.fixture
def window_benchmark():
    spec = importlib.util.spec_from_file_location("window_benchmark", Path(__file__).with_name("benchmark_fractional_window.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("alpha", [.08, .55, 1., 2.])
@pytest.mark.parametrize("window", [(1, 3), (3, 8), (1, 8)])
@pytest.mark.parametrize("need_input", [False, True])
@pytest.mark.parametrize("length", [1, 11])
def test_same_window_normalization_and_requested_gradients(window_benchmark, alpha, window, need_input, length):
    benchmark = window_benchmark
    torch.set_num_threads(2)
    value = torch.linspace(-.8, .9, 6 * length).reshape(2, 3, length).transpose(1, 2).requires_grad_(need_input)
    options = {"alpha": alpha, "log_gain": .3, "kernel_len": 8,
               "window": window, "kernel": st.FractionalGlKernel(kernel_len=8)}
    reference = benchmark.run_route("rust_buffer", value, value.detach().cos(), **options)
    actual = benchmark.run_route("torch_window_conv1d", value, value.detach().cos(), **options)
    benchmark.check_result(actual, reference)
    if alpha == 2. and window == (3, 8) and length > 3:
        assert torch.count_nonzero(actual[0]) == 0 and actual[1] != 0


def test_comparison_rejects_shape_requests_numerics_and_signed_zero_mismatches(window_benchmark):
    benchmark = window_benchmark
    good = (torch.ones(2), torch.tensor(1.), torch.tensor(2.))
    with pytest.raises(ValueError, match="request"):
        benchmark.check_result(good[:-1], good)
    with pytest.raises(ValueError, match="shape"):
        benchmark.check_result((torch.ones(1, 2), *good[1:]), good)
    with pytest.raises(ValueError, match="nonfinite"):
        benchmark.check_result((torch.full((2,), float("nan")), *good[1:]), good)
    with pytest.raises(ValueError, match="bitwise"):
        benchmark.check_result((torch.tensor(-0.), *good[1:]), (torch.tensor(0.), *good[1:]), exact=True)
    with pytest.raises(AssertionError):
        benchmark.check_result((torch.zeros(2), *good[1:]), good)


def test_buffer_benchmark_refuses_silent_fallback(window_benchmark, monkeypatch):
    benchmark = window_benchmark
    monkeypatch.setattr(benchmark.bridge, "_buffer_transport_available", lambda: False)
    with pytest.raises(ValueError, match="buffer transport"):
        benchmark.run_route("rust_buffer", torch.ones(1, 3, 2), torch.ones(1, 3, 2), alpha=.5,
                            log_gain=0., kernel=st.FractionalGlKernel(kernel_len=8), kernel_len=8, window=(1, 3))
