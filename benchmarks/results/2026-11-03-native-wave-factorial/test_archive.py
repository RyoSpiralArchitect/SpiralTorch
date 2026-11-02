import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import verify as v  # noqa: E402


class Readback(unittest.TestCase):
    def test_snapshot_crossings_and_per_window_adjoint_equalities(self):
        root=v.d.HERE/"receipts"
        self.assertEqual(v.verify(root,True)["factual_matches"],576)
        waves=[r["observation"]["sites"][v.m.SITE]["wave_factorial"] for line in v.lines(root,True)
               if (r:=json.loads(line))["arm"]=="wave_fixed"]
        self.assertEqual(len(waves),288)
        self.assertTrue(all(f["affine_outside"]==1 and f["gate_outside"]==0 and
              f["affine_on_outside"]["correction_sq"]==0. and f["saturation"]["correction_sq"]==f["affine"]["correction_sq"]
              and f["both"]["correction_sq"]==f["projection"]["correction_sq"] for f in waves))

    def test_reusing_another_seed_or_adding_an_unknown_payload_is_rejected(self):
        root=v.d.HERE/"receipts"
        row=json.loads(next(v.lines(root,True)))
        expected=v.d.anchors()
        with self.assertRaisesRegex(ValueError,"factual CE"):
            v.d.validate(row,expected["order-113","affine",0])
        bad=copy.deepcopy(row)
        bad["observation"]["sites"][v.m.SITE]["wave_factorial"]["hidden_vector"]=[1.]
        with self.assertRaisesRegex(ValueError,"local anchor"):
            v.d.validate(bad,expected["order-109","affine",0])
        bad=copy.deepcopy(row)
        bad["observation"]["sites"][v.m.SITE]["wave_factorial"]["affine"]["residual_sq"]*=2
        with self.assertRaisesRegex(ValueError,"factual energy"):
            v.d.validate(bad,expected["order-109","affine",0])


if __name__=="__main__":
    unittest.main()
