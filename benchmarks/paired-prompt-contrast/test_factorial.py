import copy
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import factorial as g


def fixture(n=6):
    pairs = g.m.matrix()[:n]
    rules, parent = {}, []
    for i,p in enumerate(pairs):
        for j,(cell,prompt) in enumerate(g.texts(p).items()):
            prefix = (i+1,j+5)
            for suffix,added in (("", ()), (p.answer,(91,)), ("\n",(198,))):
                rules[prompt+suffix] = prefix+added
        prefixes,(a,b) = g.m.prepare_pair(p,rules.__getitem__)
        parent.append(dict(key=p.key,field=p.field,target=a,lf=b,prefixes=prefixes))
    return pairs,rules,g.m.digest(parent)


def witnesses(rows):
    result={}
    for row in rows:
        for c,margin in zip(g.CELLS,(4.,1.,-2.,-7.),strict=True):
            a,b=row["target"],row["lf"]
            s=dict(prompt_tokens=len(row["prefixes"][c]),target_id=a,alternative_id=b,
                   greedy_id=40 if c=="pipe_return" else a if margin>0 else b,
                   target_logit=margin,alternative_logit=0.,target_minus_alternative=margin)
            result[row["key"],row["field"],c]=dict(score=s,model="toy",source="new" if c in g.NEW else "cached",
                                                  prefix_sha256=g.m.digest(row["prefixes"][c]))
    return result


class FactorialTests(unittest.TestCase):
    def test_swap_exactly_one_line_and_leave_values_order_suffix_unchanged(self):
        pairs=g.m.matrix()
        self.assertEqual(Counter(g.new_call_order(p)[0] for p in pairs),{"pipe_return":48,"record_what":48})
        for p in pairs:
            t=g.texts(p)
            self.assertEqual([t[k] for k in g.CACHED.values()],[p.question,p.record])
            parts={k:v.split("\n") for k,v in t.items()}
            self.assertEqual({v[2] for v in parts.values()},{"Answer:"})
            for a,b in (("pipe_what","pipe_return"),("record_what","record_return")):
                self.assertEqual(parts[a][0],parts[b][0])
            for a,b in (("pipe_what","record_what"),("pipe_return","record_return")):
                self.assertEqual(parts[a][1],parts[b][1])
            self.assertEqual(len(set(t.values())),4)

    def test_admission_pins_old_diagonals_and_rejects_crossed_target_merges(self):
        pairs,rules,pin=fixture()
        self.assertEqual(len(g.admit(pairs,rules.__getitem__,pin)),6)
        with self.assertRaisesRegex(ValueError,"diagonals"):
            g.admit(pairs,rules.__getitem__,"x")
        p=pairs[1];t=g.texts(p)["record_what"]
        rules[t+p.answer]=(*rules[t],91,7)
        self.assertEqual(len(g.admit(pairs,rules.__getitem__,pin)),6)  # First token only.
        rules[t+p.answer]=(*rules[t],17)
        with self.assertRaisesRegex(ValueError,"candidate IDs"):
            g.admit(pairs,rules.__getitem__,pin)

    def test_effects_count_complete_records_and_all_global_winners(self):
        pairs,rules,pin=fixture(); rows=g.admit(pairs,rules.__getitem__,pin)
        out=g.summarize(rows,witnesses(rows),"toy")
        self.assertEqual((out["records"],out["field_pairs"]),(2,6))
        self.assertEqual(out["mean"],dict(presentation_at_what=-6.,presentation_at_return=-8.,request_with_pipe=-3.,
                                         request_with_record=-5.,presentation=-7.,request=-4.,interaction=-2.,diagonal=-11.))
        self.assertEqual(out["winners"]["pipe_return"],{"target":0,"LF":0,"other":6})
        self.assertEqual(out["transitions"]["presentation_at_return"]["other"]["LF"],6)

    def test_single_new_session_offset_cancels_main_effects_but_not_interaction(self):
        y=dict(zip(g.CELLS,(4.,1.,-2.,-7.),strict=True))
        first=g.effects(y); second=g.effects({k:v+(2 if k in g.NEW else 0) for k,v in y.items()})
        for k in ("presentation","request","diagonal"):
            self.assertEqual(first[k],second[k])
        self.assertEqual(second["interaction"]-first["interaction"],-4.)

    def test_no_diagonal_is_scored_and_bad_cached_tail_blocks_even_first_read(self):
        pairs,rules,pin=fixture(); rows=g.admit(pairs,rules.__getitem__,pin)
        known=witnesses(rows)
        cached={k:v for k,v in known.items() if k[-1] not in g.NEW}
        calls=[]
        def observe(p,c,ids,target,lf):
            self.assertIn(c,g.NEW)
            self.assertEqual((tuple(rules[g.texts(p)[c]]),91,198),(ids,target,lf))
            calls.append((p.key,p.field,c))
            return known[p.key,p.field,c]["score"]
        expected=g.summarize(rows,known,"toy")
        self.assertEqual(g.measure_crossed(pairs,rows,cached,"toy",observe),expected)
        self.assertEqual(calls,[(p.key,p.field,c) for p in pairs for c in g.new_call_order(p)])
        self.assertEqual(len(cached),2*len(pairs))
        calls.clear()
        cached[next(reversed(cached))]={**cached[next(reversed(cached))],"prefix_sha256":"wrong"}
        with self.assertRaisesRegex(ValueError,"prefix"):
            g.measure_crossed(pairs,rows,cached,"toy",observe)
        self.assertFalse(calls)

    def test_full_mode_observes_every_cell_and_rejects_cached_evidence(self):
        pairs,rules,pin=fixture(); rows=g.admit(pairs,rules.__getitem__,pin)
        known=witnesses(rows); calls=[]
        def observe(p,c,ids,a,b):
            self.assertEqual(ids, tuple(rules[g.texts(p)[c]]))
            calls.append((p.key,p.field,c))
            return known[p.key,p.field,c]["score"]
        with self.assertRaisesRegex(ValueError,"grid"):
            g.measure_all(pairs,rows,"toy",observe)
        self.assertFalse(calls)
        fresh=g.measure_all(pairs,rows,"toy",observe,grid_pin=g.m.digest(rows))
        self.assertEqual(len(set(calls)),4*len(rows))
        self.assertEqual(calls,[(p.key,p.field,c) for i,p in enumerate(pairs) for c in g.FULL_ORDERS[i%4]])
        self.assertEqual(fresh,g.summarize(rows,known,"toy"))
        with self.assertRaisesRegex(ValueError,"source"):
            g.summarize(rows,known,"toy",mode="all_measured")
        orders=[g.FULL_ORDERS[i%4] for i in range(96)]
        for pos in range(4):
            self.assertEqual(Counter(o[pos] for o in orders),dict.fromkeys(g.CELLS,24))
        steps=Counter((a,b) for o in orders for a,b in zip(o,o[1:]))
        self.assertEqual(steps,{(a,b):24 for a in g.CELLS for b in g.CELLS if a!=b})

    def test_reject_missing_and_misattributed_evidence(self):
        pairs,rules,pin=fixture(); rows=g.admit(pairs,rules.__getitem__,pin); original=witnesses(rows)
        key=next(iter(original))
        for change in ("missing","prefix","owner","source","winner","tie"):
            data=copy.deepcopy(original);r=data[key];s=r["score"]
            if change=="missing":del data[key]
            elif change=="prefix":r["prefix_sha256"]="wrong"
            elif change=="owner":r["model"]="foreign"
            elif change=="source":r["source"]="new"
            elif change=="winner":s["greedy_id"]=198
            elif change=="tie":s.update(greedy_id=198,target_logit=0.,target_minus_alternative=0.)
            with self.subTest(change=change),self.assertRaises(ValueError):
                g.summarize(rows,data,"toy")
        with self.assertRaisesRegex(ValueError,"triplet|per field"):
            g.summarize(rows[1:],original,"toy")


if __name__=="__main__":
    unittest.main()
