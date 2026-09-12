"""Reject missing, duplicated or incorrectly shaped classification probes."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("replay", Path(__file__).with_name("validate_resident_graph_training_vs_torch.py"))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def fixture():
    keys = [(n,r,s) for n in ("nd","uniform","strided","broadcast")
            for r in ("none","sum","mean") for s in (0.,.2,1.)]
    keys += [("wide_mean","mean",0.), ("tiny_smoothing","mean",1e-40), ("tiny_tail","mean",0.),
             ("wide_vocab","mean",.2), ("single_class","mean",.2), ("empty_none","none",0.), ("empty_sum","sum",0.)]
    return [dict(name=n,reduction=r,label_smoothing=s,shape=[1,50257] if n=="wide_vocab" else [1,2],
                 prediction=[0.]*(50257 if n=="wide_vocab" else 2),target=[0.]) for n,r,s in keys]


class Admission(unittest.TestCase):
    def test_complete_membership_is_required(self):
        probes = fixture()
        replay.admit_classification_probes(probes)
        for mutate in [lambda p:p.pop(), lambda p:p.append(copy.deepcopy(p[0])),
                       lambda p:p.__setitem__(1,copy.deepcopy(p[0])), lambda p:p[-4].update(shape=[1,2],prediction=[0.,0.]),
                       lambda p:p[0].update(target=[]), lambda p:p[0].update(prediction=[])]:
            broken = copy.deepcopy(probes); mutate(broken)
            with self.assertRaises(AssertionError): replay.admit_classification_probes(broken)


if __name__ == "__main__": unittest.main()
