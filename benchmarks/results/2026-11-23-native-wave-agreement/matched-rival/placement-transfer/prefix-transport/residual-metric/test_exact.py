import runpy
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

exact=runpy.run_path(str(Path(__file__).with_name("verify_exact.py")))

class Exact(unittest.TestCase):
    def test_halfway_and_just_past(self):
        halfway=1+Fraction(1,1<<24)
        nearest=exact["nearest32"]
        self.assertEqual(nearest(halfway),1.)
        self.assertEqual(nearest(halfway+Fraction(1,1<<100)),1.+2**-23)
        self.assertEqual(nearest(-halfway),-1.)
        self.assertEqual(nearest(1+3*Fraction(1,1<<24)),1.+2**-22)
        self.assertEqual(nearest(Fraction(1,1<<150)),0.)

    def test_signed_relative_lag_and_unrotated_tail(self):
        self.assertEqual(exact["means"]((2.,-1.,4.,1.,2.,-6.),2,"quarter_turn"),(1.,2.,-1.))

    def test_new_launcher_refuses_an_existing_split_before_input_or_model_access(self):
        run=runpy.run_path(str(Path(__file__).with_name("run_exact.py")))["run"]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"formal").mkdir()
            with self.assertRaises(FileExistsError):
                run("formal",root,root/"missing",root/"missing","invalid")

    def test_boundary_counts_only_initially_wrong_rows(self):
        fraction=runpy.run_path(str(Path(__file__).with_name("boundary.py")))["fraction"]
        self.assertIsNone(fraction(0.,1.))
        self.assertEqual(fraction(-2.,-1.),.5)
        self.assertEqual(fraction(-2.,0.),1.)
        self.assertEqual(fraction(-2.,-3.),-.5)

if __name__=="__main__":
    unittest.main()
