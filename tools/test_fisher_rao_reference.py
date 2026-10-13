"""Independent autograd and categorical-geometry tests; CPU Torch is optional."""
import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    spec = importlib.util.spec_from_file_location("fisher", Path(__file__).with_name("fisher_rao_reference.py"))
    fisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fisher)


@unittest.skipIf(torch is None, "optional CPU Torch")
class FisherReferenceTests(unittest.TestCase):
    def data(self):
        x = torch.tensor([[[.2, -.5, 1., .7], [.8, .1, -.2, -.4], [-.7, .9, .3, -.2]]],
                         dtype=torch.float64, device="cpu", requires_grad=True)
        g = torch.tensor([-.6, .3], dtype=torch.float64, device="cpu", requires_grad=True)
        return x, g

    def test_autograd_matches_numerical_jacobian(self):
        self.assertTrue(torch.autograd.gradcheck(fisher.metric, self.data(), eps=1e-6, atol=1e-7, rtol=1e-5))

    def test_closed_form_off_diagonal_and_shift_invariance(self):
        x, g = self.data()
        y = fisher.metric(x, g)
        roots = torch.softmax(x, -1).sqrt()
        affinity = (roots[:, :, None] * roots[:, None, :]).sum(-1)
        distance = 4 * affinity.clamp(0., 1.).acos().square()
        expected = -torch.nn.functional.softplus(g)[None, :, None, None] * distance[:, None]
        selected = torch.ones((3, 3), dtype=torch.bool, device="cpu").tril(-1)
        torch.testing.assert_close(y[:, :, selected], expected[:, :, selected], atol=1e-13, rtol=1e-12)
        shifted = x + torch.tensor([[[8.], [-4.], [16.]]], device="cpu")
        torch.testing.assert_close(y, fisher.metric(shifted, g), atol=1e-13, rtol=1e-12)
        torch.testing.assert_close(y, fisher.metric(x[:, :, [2, 0, 3, 1]], g), atol=1e-13, rtol=1e-12)

    def test_identity_and_causal_endpoint_pullback(self):
        x, g = self.data()
        seed = torch.zeros((1, 2, 3, 3), dtype=x.dtype, device="cpu")
        seed[0, 0, 1, 0] = 1.
        dx, dg = torch.autograd.grad(fisher.metric(x, g), (x, g), seed)
        self.assertTrue(bool((dx[0, 0] != 0).any()) and bool((dx[0, 1] != 0).any()))
        self.assertTrue(bool((dx[0, 2] == 0).all()))
        self.assertGreater(abs(dg[0].item()), 0.)
        identity = torch.zeros_like(x, requires_grad=True)
        scores = fisher.metric(identity, g)
        gradient = torch.autograd.grad(scores.sum(), (identity, g))
        self.assertTrue(bool((scores == 0).all()))
        self.assertTrue(all(bool((v == 0).all()) for v in gradient))


if __name__ == "__main__":
    unittest.main()
