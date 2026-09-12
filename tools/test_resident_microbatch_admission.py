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
