import importlib.util
import json
from pathlib import Path

import pytest
import torch
import spiraltorch as st

spec = importlib.util.spec_from_file_location("topos_benchmark", Path(__file__).with_name("benchmark_topos_learning.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize("iterations,coupling", [(1, 0.), (4, .25), (16, .75)])
@pytest.mark.parametrize("porosity", [0., .2])
@pytest.mark.parametrize("origin", [False, True])
def test_matched_finite_unroll_and_both_vjps(iterations, coupling, porosity, origin):
    rng = torch.Generator().manual_seed(19)
    values = [torch.randn(2, 7, 5, generator=rng), torch.randn(5, generator=rng)]
    if origin:
        values[1].zero_()
    dy = torch.randn(2, 7, 5, generator=rng)
    kernel = st.ToposResonatorKernel(coupling=coupling, iterations=iterations, porosity=porosity)
    config = json.loads(kernel.configuration_json())
    reference = bench.run_route("rust_list", values, dy, kernel, config)
    for name in bench.routes(kernel):
        assert len(bench.check_result(bench.run_route(name, values, dy, kernel, config), reference,
                                     exact=name != "torch_reference")) == 3


def test_signed_zero_nonfinite_missing_gradient_and_unbalanced_order_rejected():
    zero = (torch.tensor(0.),) * 3
    for actual, label in [((torch.tensor(-0.), *zero[1:]), "bits"),
                          ((torch.tensor(float("nan")), *zero[1:]), "nonfinite"), ((), "arity")]:
        with pytest.raises(ValueError, match=label):
            bench.check_result(actual, zero, exact=True)
    for count in (3, 4):
        names = list(range(count))
        orders = bench.round_orders(names, 12)
        for position in range(count):
            assert all(sum(row[position] == n for row in orders) == 12 // count for n in names)
        with pytest.raises(ValueError, match="balance"):
            bench.round_orders(names, 1)
