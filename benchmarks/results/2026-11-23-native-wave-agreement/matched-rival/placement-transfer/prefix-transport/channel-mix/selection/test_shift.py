import runpy
from pathlib import Path
import unittest

m=runpy.run_path(str(Path(__file__).with_name("shift_readback.py")))
rows=m["u"].fresh("rehearsal")[::2]

def bank(d):
    return {r["key"]:[0.]*7+[x,0.,0.,0.] for r,x in zip(rows,d)}

class Shift(unittest.TestCase):
    def test_train_shift_is_not_an_oracle_for_a_new_frame(self):
        origin=bank([-1.,-1.,1.,1.])
        train=bank([-2.,-2.,0.,0.]);new=bank([1.,1.,3.,3.])
        r=m["one"](rows,rows,origin,train,origin,new,"head")
        self.assertEqual((r["train_offset"],r["transfer_offset"],r["frame_mean_energy_fraction"]),(-1.,2.,1.))
        self.assertEqual((r["views"]["full"]["up"],r["views"]["frame_shift_only"]["reproduced"]),(2,2))
        self.assertEqual(r["views"]["frame_shift_removed"]["correctness_flips"],0)
        self.assertEqual(r["views"]["train_shift_only"]["reproduced"],0)

    def test_role_changes_remain_after_removing_a_constant(self):
        origin=bank([-.5,-.5,.5,.5]);new=bank([.5,.5,-.5,-.5])
        r=m["one"](rows,rows,origin,new,origin,new,"head")
        self.assertEqual(r["frame_mean_energy_fraction"],0.)
        self.assertEqual(r["views"]["frame_shift_removed"]["up"],4)
        self.assertEqual(r["views"]["train_shift_only"]["correctness_flips"],0)

if __name__=="__main__":unittest.main()
