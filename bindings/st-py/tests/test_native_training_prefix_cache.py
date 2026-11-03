"""Synthetic native models only. A cache can change cost, never optimizer semantics."""
import json
from pathlib import Path
import pickle
import runpy
import unittest

import spiraltorch as st

fixtures = runpy.run_path(str(Path(__file__).with_name("test_native_wave_factorial.py")))
Session = st.NativeCausalLmTrainingSession
Cache = st.NativeCausalLmTrainingPrefixCache
DOCS = [[1,2,3,4], [3,6,5,7], [7,8,9,1]]
MASKS = [[-100,-100,3,4], [-100,6,-100,7], [-100,-100,-100,1]]


class PrefixCache(unittest.TestCase):
    def setUp(self):
        self.base = st.NativeCausalLm.from_gpt2(*fixtures["fixture"]())
        self.models = [fixtures["attach"](self.base,k) for k in fixtures["KINDS"]]

    def cohort(self):
        a = Session(self.models[0], DOCS, 4, labels=MASKS, mode="joint", epochs=3, learning_rate=.02, window_policy="covering_targets")
        return [a] + [a.fork_variant(m,"joint") for m in self.models[1:]]

    def test_four_operators_repeat_and_resume_identically_with_no_hidden_in_checkpoint(self):
        cached, plain = self.cohort(), self.cohort()
        cache = Cache(self.base,max_values=36,max_entries=3)
        for i in range(9):
            a, reports = Session.step_matched(cached, observe_geometry=True, prefix_cache=cache)
            b, expected = Session.step_matched(plain, observe_geometry=True)
            self.assertEqual((a,b),(1,1))
            for j,(r,e) in enumerate(zip(reports,expected,strict=True)):
                self.assertEqual(r.to_json(),e.to_json())
                self.assertEqual(r.observation_json(),e.observation_json())
                payload,pin = cached[j].checkpoint_with_id()
                self.assertEqual((payload,pin),plain[j].checkpoint_with_id())
                if i==2:
                    self.assertNotIn("prefix_cache",json.loads(payload))
                    cached[j] = Session.from_json(self.base,DOCS,payload,pin,labels=MASKS)
            self.assertEqual(cache.stats()["last_action"],"inserted" if i<3 else "hit")
        self.assertEqual(cache.stats(),dict(entries=3,retained_values=36,hits=6,insertions=3,
                         skipped_block_calls=6,capacity_bypasses=0,no_frozen_block=0,
                         max_values=36,max_entries=3,last_action="hit"))
        self.assertIsNone(Session.step_matched(cached,prefix_cache=cache))
        exported = cache.stats()
        exported["entries"] = 0
        self.assertEqual(cache.stats()["entries"],3)
        with self.assertRaises((TypeError,pickle.PicklingError)):
            pickle.dumps(cache)
        cache.clear()
        self.assertIsNone(cache.stats()["last_action"])
        self.assertEqual(cache.stats()["retained_values"],0)

    def test_one_session_and_different_targets_hit_without_reusing_a_loss(self):
        cache=Cache(self.base,max_values=12,max_entries=1)
        def one(ids,mask,model):
            return Session(model,[ids],4,labels=[mask],mode="joint",learning_rate=.1)
        for j,(ids,mask) in enumerate([(DOCS[0],MASKS[0]),([1,2,3,9],[-100,7,6,-100])]):
            session=one(ids,mask,self.models[j])
            expected=one(ids,mask,self.models[j])
            self.assertEqual(session.step(prefix_cache=cache).to_json(),expected.step().to_json())
            self.assertEqual(session.checkpoint_id,expected.checkpoint_id)
        self.assertEqual((cache.stats()["hits"],cache.stats()["insertions"]),(1,1))
        self.assertIsNone(session.step(prefix_cache=cache))
        wrong=Cache(self.base.with_readout_accumulator("f64"),max_values=12,max_entries=1)
        with self.assertRaisesRegex(ValueError,"base or readout"):
            one(DOCS[0],MASKS[0],self.models[0]).step(prefix_cache=wrong)
        self.assertEqual(wrong.stats()["insertions"],0)

    def test_admission_and_reentrancy_do_not_advance_or_populate(self):
        self.assertIn("NativeCausalLmTrainingPrefixCache",st.__all__)
        for v,e in [(True,1),(1,False),(0,1),(1,0),(Cache.MAX_VALUES+1,1),(1,Cache.MAX_ENTRIES+1)]:
            with self.assertRaises((TypeError,ValueError)):
                Cache(self.base,max_values=v,max_entries=e)
        cache=Cache(self.base,max_values=12,max_entries=1)
        a,b,*_=self.cohort()
        with self.assertRaises(ValueError):
            Session.step_matched([a,a],prefix_cache=cache)
        class ForeignRoster:
            def __len__(self): return 2
            def __getitem__(self,i):
                cache.clear()  # resolve foreign objects before borrowing the cache
                return [a,b][i]
        self.assertEqual(a.completed_updates,0)
        self.assertEqual(cache.stats()["entries"],0)
        Session.step_matched(ForeignRoster(),prefix_cache=cache)
        self.assertEqual((a.completed_updates,b.completed_updates,cache.stats()["entries"]),(1,1,1))


if __name__=="__main__":
    unittest.main()
