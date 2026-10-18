import importlib.util
import math
from pathlib import Path
import unittest

SPEC=importlib.util.spec_from_file_location("qualify",Path(__file__).with_name("qualify.py"))
Q=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


class QualificationTests(unittest.TestCase):
    def test_baseline_sees_only_completed_training(self):
        request=dict(config=dict(steps=2),train_documents=[[0,1,2],[255,255,255]],
                     train_batches=[[(0,0)],[(1,0)]],validation_documents=[[255,255,255]],validation_batches=[[(0,0)]])
        early=Q.baseline(request,1)
        self.assertEqual(early["training_target_presentations"],2)
        self.assertAlmostEqual(early["unigram_ce"],math.log(258))
        late=Q.baseline(request,2)
        self.assertEqual(late["training_target_presentations"],4)
        self.assertAlmostEqual(late["unigram_ce"],math.log(260/3))

    def test_repetition_counts_raw_bytes(self):
        result=Q.repetition([[32,32,32,32],[0,255,255,0]])
        self.assertEqual(result,dict(bytes=8,space_bytes=4,distinct_bytes_per_row=[1,2],longest_identical_run_per_row=[4,2]))

    def test_validation_windows_must_be_complete(self):
        request=dict(config=dict(steps=2),train_documents=[[0,1,2]],train_batches=[[(0,0)]],
                     validation_documents=[[1,2]],validation_batches=[[(0,0)]])
        with self.assertRaisesRegex(ValueError,"validation window"):
            Q.baseline(request,1)

    def test_checkpoint_weights_are_bound_bitwise(self):
        desc=[dict(shape=[2])]
        saved=[dict(shape=[2],values=[1.,-0.])]
        Q.bind_tensors(desc,saved,[[1.,-0.]])
        with self.assertRaisesRegex(ValueError,"weights differ"):
            Q.bind_tensors(desc,saved,[[1.01,-0.]])
        with self.assertRaisesRegex(ValueError,"weights differ"):
            Q.bind_tensors(desc,saved,[[1.,0.]])

    def test_checkpoint_history_is_bound(self):
        import copy
        report=dict(training=[dict(revision=1,ce=2.)],validation=[dict(revision=0,batch_losses=[3.])])
        Q.bind_history(report,report)
        changed=copy.deepcopy(report)
        changed["training"][0]["ce"]=1.
        with self.assertRaisesRegex(ValueError,"training history"):
            Q.bind_history(changed,report)
        changed=copy.deepcopy(report)
        changed["validation"][0]["batch_losses"]=[2.]
        with self.assertRaisesRegex(ValueError,"validation history"):
            Q.bind_history(changed,report)

    def test_checkpoint_tensor_coverage(self):
        with self.assertRaisesRegex(ValueError,"coverage"):
            Q.bind_tensors([dict(shape=[1])],[],[[1.]])


if __name__=="__main__":
    unittest.main()
