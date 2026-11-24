from pathlib import Path
import runpy
import unittest

P = runpy.run_path(str(Path(__file__).with_name("channel_partition.py")))
pack = P["M"]["packed"]


class PartitionTests(unittest.TestCase):
    def test_removing_matched_gate_and_bias_not_individual_coordinates(self):
        # Per-channel products (-5,+1) -> (+2,+1); both have positive bias.
        ones = pack([1.,1.,1.,1.])
        first, second = pack([-6.,0.,1.,1.]), pack([1.,0.,1.,1.])
        x = P["partition"]((ones,first),(ones,second))
        self.assertEqual((x["affine_dot"],x["saturation_dot"],x["delta_gate"],x["delta_bias"]),(-4.,3.,7.,0.))
        self.assertEqual(x["exceptions"],[dict(channel=0,signs=[1,1])])
        self.assertEqual((x["same_sign_pair_without_one"],x["max_share_of_absolute_change"]),(1,1.))
        self.assertEqual((x["only_gate_replaced_dot"],x["only_bias_replaced_dot"]),(3.,-4.))

    def test_zero_and_ties_are_not_strict_signs(self):
        v = pack([1.,0.])
        x = P["partition"]((v,v),(v,v))
        self.assertEqual(x["exceptions"],[dict(channel=0,signs=[0,0])])
        self.assertIsNone(x["max_share_of_absolute_change"])
        with self.assertRaises(ValueError): P["partition"]((v,v),(v,pack([1.])))

    def test_raw_gates_inside_is_not_post_affine_inside(self):
        g, s = pack([2.,2.,1.,1.]), pack([0.,2.,.5,1.])
        args = [[g,g],[s,s],[.5,.5],[0.,0.],[3.,2.],1.,0]
        r = P["support"](*args)
        self.assertEqual((r["post_affine_outside"],r["changed_parameter_channels"],r["top_delta_pre_saturation"]),([0],[0],1.5))
        args[4] = [1.,1.]
        with self.assertRaises(ValueError): P["support"](*args)
        args[2] = [2.,0.]
        with self.assertRaises(ValueError): P["support"](*args)


if __name__=="__main__":
    unittest.main()
