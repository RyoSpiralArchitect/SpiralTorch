import runpy
import unittest
from pathlib import Path

study=runpy.run_path(str(Path(__file__).with_name("interaction.py")))
interaction=study["interaction"]


class FourMapTests(unittest.TestCase):
    def test_additive_change_has_no_interaction(self):
        out=interaction(dict(affine=[2,2],saturation=[3,2],projection=[2,1],both=[3,1]))
        self.assertEqual(out["residual_over_full"],0.)
        self.assertAlmostEqual(out["additive_cosine_to_full"],1.)

    def test_interaction_is_signed_before_taking_the_norm(self):
        out=interaction(dict(affine=[1,0],saturation=[2,0],projection=[0,0],both=[-2,0]))
        self.assertEqual(out,dict(residual_over_affine=3.,residual_over_full=1.5,additive_cosine_to_full=-1.))

    def test_undefined_is_not_zero(self):
        out=interaction({k:[0.,0.] for k in ("affine","saturation","projection","both")})
        self.assertTrue(all(v is None for v in out.values()))
        self.assertEqual(study["summarize"]([None,0.,2.])["median"],1.)
        with self.assertRaises(ValueError):
            interaction(dict(affine=[1.],saturation=[1.,2.],projection=[2.],both=[0.]))


if __name__=="__main__":
    unittest.main()
