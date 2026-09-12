"""Reject missing windows, changed batching and incorrect normalization weights."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("reference",Path(__file__).with_name("validate_resident_graph_training_vs_torch.py"))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

def fixture():
    counts = [6,4,2,5,3,1,6]
    data = [dict(input=[0.]*24,target=[0.]*count+[-100.]*(6-count)) for count in counts]
    windows = []
    for i in range(32):
        ids = [(i*3+j)%7 for j in range(2+i%3)]
        total = sum(counts[j] for j in ids)
        windows.append(dict(rate=0. if i%11 == 0 else .1,
                            microbatches=[dict(batch=j,weight=counts[j]/total) for j in ids]))
    return dict(policy="Exact",input_shape=[2,3,4],plan=dict(input_shape=[2,3,4]),
                observations_after_updates=32,microbatches=95,module_parameters_applied=7,
                optimizer="explicit_sgd_not_ModuleTrainer",windows=windows,dataset=data,
                reduction="mean",label_smoothing=.1,ignore_index=-100,
                evaluation=[dict(batch=i) for i in range(7)])

class Admission(unittest.TestCase):
    def test_momentum_schedule_history_and_resets_are_required(self):
        good = fixture();good.update(gradient_clip=True,topos_momentum=True)
        for i,w in enumerate(good["windows"]):
            d=[.6,.85,0.,None,.3][i%5]
            w.update(grad_clip_max_norm=[.05,None,.1,2.,.001][i%5],momentum_damping=d,
                     momentum=None if d is None else [[] for _ in range(7)],reset_momentum=d is not None and i%13==0)
        reference.admit_microbatch(good,clipped=True,momentum=True)
        with self.assertRaises(AssertionError): reference.admit_microbatch(good,clipped=True)
        for mutate in [lambda c:c.pop("topos_momentum"),lambda c:c["windows"][0].update(momentum_damping=.9),
                       lambda c:c["windows"][0].update(reset_momentum=False),lambda c:c["windows"][0].update(momentum=None),
                       lambda c:c["windows"][3].update(momentum_damping=.6)]:
            bad=copy.deepcopy(good);mutate(bad)
            with self.assertRaises((AssertionError,KeyError,TypeError)):
                reference.admit_microbatch(bad,clipped=True,momentum=True)

    def test_clipping_schedule_cannot_be_relabelled_or_omitted(self):
        good = fixture(); good["gradient_clip"] = True
        for i,w in enumerate(good["windows"]): w["grad_clip_max_norm"] = [.05,None,.1,2.,.001][i%5]
        reference.admit_microbatch(good,clipped=True)
        with self.assertRaises(AssertionError): reference.admit_microbatch(good)
        for mutate in [lambda c:c.pop("gradient_clip"), lambda c:c["windows"][0].pop("grad_clip_max_norm"),
                       lambda c:c["windows"][0].update(grad_clip_max_norm=1.),lambda c:c["windows"][1].update(grad_clip_max_norm=.05)]:
            bad=copy.deepcopy(good); mutate(bad)
            with self.assertRaises((AssertionError,KeyError)): reference.admit_microbatch(bad,clipped=True)

    def test_frozen_microbatch_sequence_and_weights_are_required(self):
        good = fixture(); reference.admit_microbatch(good)
        mutations = [
            lambda c:c["windows"].pop(),
            lambda c:c["windows"][0]["microbatches"][0].update(weight=0.5),
            lambda c:c["windows"][0]["microbatches"][0].update(batch=1),
            lambda c:c["dataset"][1].update(target=[0.]*6),
            lambda c:c.update(microbatches=96),
            lambda c:c.update(observations_after_updates=1),
            lambda c:c.update(optimizer="ModuleTrainer"),
            lambda c:c["evaluation"].pop(),
        ]
        for mutate in mutations:
            bad = copy.deepcopy(good); mutate(bad)
            with self.subTest(mutate=mutate), self.assertRaises(AssertionError): reference.admit_microbatch(bad)

if __name__ == "__main__": unittest.main()
