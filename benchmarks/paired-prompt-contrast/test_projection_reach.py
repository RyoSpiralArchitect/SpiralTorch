import copy
import math
import struct
import unittest

import entity_swap_task as task
import projection_reach as pr
from projection_reach_audit import central,decision


class ProjectionReachTests(unittest.TestCase):
    def test_only_old_training_pairs_enter_two_identity_balanced_folds(self):
        data=task.deck("train");folds=[[r for r in data if r["edge"]%2==p] for p in (0,1)]
        for j,rows in enumerate(folds):
            pr.fold_rows(rows,j)
            self.assertEqual({e for r in rows for e in r["entities"]},set(task.ENTITIES))
            self.assertEqual(sum(pr.sign(r) for r in rows),0)
        self.assertTrue({tuple(r["entities"]) for r in folds[0]}.isdisjoint(tuple(r["entities"]) for r in folds[1]))
        with self.assertRaises(ValueError):pr.fold_rows(folds[0],1)

    def test_coordinates_preserve_dense_row_major_diagonal_and_geometry_sign(self):
        w=32;matrix=list(map(float,range(w*w)));bias=[-1.]*w
        block=struct.pack("<1056d",*bias,*matrix)*8+struct.pack("<64f",*([1.]*64))+struct.pack("<64f",*([3.]*64))
        result=pr.decompose(block)
        self.assertEqual(result["1/output/diagonal"],tuple(bias+[matrix[j*(w+1)] for j in range(w)]))
        self.assertEqual(result["0/key/dense"][w+2*w+5],69.)
        self.assertEqual(result["geometry"],(-2.,)*64)
        self.assertEqual(set(pr.coordinates())-set(result),{"0/key/gauge","1/key/gauge"})
        v=pr.directions({k:[0.]*len(a) for k,a in result.items()})
        self.assertIsNone(v["geometry"])
        self.assertAlmostEqual(math.hypot(*v["0/key/gauge"]),1.)
        self.assertTrue(all(x==0. for x in v["0/key/gauge"][w:]))
        with self.assertRaises(ValueError):pr.decompose(block[:-4])

    def test_near_zero_agreement_is_not_resolved_and_sign_matters(self):
        tiny=central(3e-8,-3e-8,-1e-8,1e-8)
        self.assertTrue(tiny["agrees"]);self.assertFalse(tiny["resolved_ascent"])
        self.assertFalse(central(.1,-.1,-.1,.1)["agrees"])
        asym=central(.011,-.009,.011,-.009)
        self.assertAlmostEqual(asym["predicted"],.01);self.assertAlmostEqual(asym["bend"],.001)

    def test_construction_norm_alone_cannot_qualify_a_route(self):
        def cell(x):
            a=central(x,-x,x,-x)|dict(by_edge={str(i):x for i in range(4)},by_order=[x,x])
            return dict(fit_norm=x*10,radii={str(h):{f:copy.deepcopy(a) for f in ("fit","check")} for h in pr.RADII})
        lab="0/query/diagonal";rows={lab:cell(.003),"0/output/diagonal":cell(.001),"geometry":cell(.0001)}
        self.assertTrue(decision(rows,lab))
        rows[lab]["radii"]["0.02"]["check"]["by_order"][1]=-1.
        self.assertFalse(decision(rows,lab))
        rows[lab]=cell(.003);rows[lab]["radii"]["0.002"]["fit"]["agrees"]=False
        self.assertFalse(decision(rows,lab))
        rows[lab]=None;self.assertFalse(decision(rows,lab))
