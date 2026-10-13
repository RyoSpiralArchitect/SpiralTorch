import unittest
import torch
from projective_reference import metric


class ProjectiveReferenceTests(unittest.TestCase):
    def test_known_chord_and_identity(self):
        x = torch.tensor([[[0., 0.], [1., 0.]]], dtype=torch.float64)
        result = metric(x, torch.zeros(1, dtype=torch.float64))
        self.assertAlmostEqual(result[0, 0, 1, 0].item(), -0.5 * torch.log(torch.tensor(2., dtype=torch.float64)).item())
        self.assertEqual(result[0, 0, 0, 1].item(), 0.)
        self.assertEqual(result[0, 0, 1, 1].item(), 0.)

    def test_complex_projector_full_gradcheck(self):
        x = torch.tensor([[[.3, -.2, .5, .1], [-.4, .7, .2, -.8]]], dtype=torch.float64, requires_grad=True)
        g = torch.tensor([-.4, .9], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(metric, (x, g), eps=1e-6, atol=1e-6))

    def test_common_phase_of_chart_is_invariant(self):
        x = torch.tensor([[[.3, -.2, .5, .1], [-.4, .7, .2, -.8]]], dtype=torch.float64)
        z = torch.view_as_complex(x.reshape(1, 2, 2, 2))
        phase = torch.polar(torch.tensor(1., dtype=torch.float64), torch.tensor(.73, dtype=torch.float64))
        rotated = torch.view_as_real(z * phase).flatten(-2)
        self.assertTrue(torch.allclose(metric(x, torch.zeros(1)), metric(rotated, torch.zeros(1)), atol=1e-14))


if __name__ == "__main__":
    unittest.main()
