from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from factorial_archive import load_cached
from test_factorial import fixture, witnesses
import factorial as g


def seal(root, name, value):
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    (root/name).write_bytes(raw)
    return sha256(raw).hexdigest()


def parent(root, mutate=None):
    pairs, rules, old_pin = fixture(3)
    rows = g.admit(pairs, rules.__getitem__, old_pin)
    models = [dict(name="toy", owners=["test-lora","test-geometry"])]
    evidence = witnesses(rows)
    events = []
    k = 0
    for p,row in zip(pairs,rows,strict=True):
        for form in (p.first_form, "record" if p.first_form == "question" else "question"):
            k += 1
            x = evidence[p.key,p.field,g.CACHED[form]]
            ident = dict(n=k, key=p.key, field=p.field, model="toy",form=form)
            events.extend([dict(**ident,event="start",target=row["target"],lf=row["lf"],prefix_sha256=x["prefix_sha256"]),
                           dict(**ident,event="returned",score=x["score"])])
    if mutate:
        mutate(events)
    trace = seal(root,"events.jsonl",b"".join(json.dumps(e).encode()+b"\n" for e in events))
    attempt = seal(root,"attempt.json",dict(models=models,wheel="wheel",native="native",base="base",archive="archive",source="code",calls=6,
                                          matrix=old_pin,new_updates=0))
    audit = seal(root,"audit.json",dict(models={"toy":dict(mean_delta=-11.,question={"target":3},record={"LF":3})},
                                       wheel="wheel",source="code",calls=6,input=old_pin,trace=trace,new_training_updates=0))
    pin = seal(root,"complete.json",dict(source="code",calls=6,attempts=1,model_updates=0,attempt=attempt,audit=audit))
    return pin,pairs,rows,models


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)

    def test_replay_copies_exact_saved_scores_without_crossed_cells(self):
        args=parent(self.root)
        scores=load_cached(self.root,*args,"wheel","native",base_id="base",archive_id="archive")["toy"]
        self.assertEqual(len(scores),6)
        original=witnesses(args[2])
        for k,v in scores.items():
            self.assertIn(k[-1],g.CACHED.values())
            self.assertEqual(original[k],v)

    def test_hashes_and_model_identity_cannot_be_substituted(self):
        pin,pairs,rows,models=parent(self.root)
        for expected,wheel in [([dict(name="toy",owners=[None,None])],"wheel"),(models,"another")]:
            with self.assertRaisesRegex(ValueError,"lineage"):
                load_cached(self.root,pin,pairs,rows,expected,wheel,"native",base_id="base",archive_id="archive")
        for base, archive in [("other", "archive"), ("base", "other")]:
            with self.assertRaisesRegex(ValueError,"lineage"):
                load_cached(self.root,pin,pairs,rows,models,"wheel","native",base_id=base,archive_id=archive)
        with (self.root/"events.jsonl").open("ab") as f:
            f.write(b" ")
        with self.assertRaisesRegex(ValueError,"archive member"):
            load_cached(self.root,pin,pairs,rows,models,"wheel","native",base_id="base",archive_id="archive")

    def test_even_self_consistently_rehashed_bad_events_are_rejected(self):
        for mutation in (lambda es:es[1].update(field="foreign"), lambda es:es[0].update(prefix_sha256="foreign"),
                         lambda es:es.pop(), lambda es:es.extend(es[:2])):
            args=parent(self.root,mutation)
            with self.assertRaises(ValueError):
                load_cached(self.root,*args,"wheel","native",base_id="base",archive_id="archive")


if __name__ == "__main__":
    unittest.main()
