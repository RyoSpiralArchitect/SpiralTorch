from collections import Counter
import json
import struct
from pathlib import Path
import tempfile
import unittest

import causal_key_lag_fit as study
from projection_language_fixture import weights


class CausalKeyLagFit(unittest.TestCase):
    def test_microbatches_keep_complete_sequences_and_the_original_exposure(self):
        rows = study.prev.decks()["train0"]
        logical = rows * 2
        self.assertEqual(study.PLAN["microbatch_rows"], study.MICROBATCH)
        chunks = [logical[i : i + study.MICROBATCH] for i in range(0, len(logical), study.MICROBATCH)]
        self.assertEqual([len(chunk) for chunk in chunks], [8] * 8)
        self.assertEqual([q for chunk in chunks for q in chunk], logical)
        self.assertLessEqual(max(sum(map(len, c)) * 32 for c in chunks), 4096)
        self.assertGreater(sum(map(len, rows)) * 32, 4096)

    def test_layout_does_not_change_labels_facts_or_prompt_exposure(self):
        rows = study.prev.decks()["train0"]
        for q in rows:
            last = study.present(q, "query")
            self.assertEqual(last[:7], q[:7])
            self.assertEqual(Counter(last), Counter(q))
            self.assertEqual(last[-1], q[7])
        ys = study.golds(rows, "aligned")
        shuffled = study.golds(rows, "reflected")
        self.assertEqual(Counter(ys), Counter(shuffled))
        self.assertEqual(ys[:32], shuffled[:32])
        self.assertTrue(all(a + b == 18 for a, b in zip(ys[32:], shuffled[32:], strict=True)))

    def test_rebinding_changes_only_source_identity(self):
        old = "sha256:" + "a" * 64
        cfg = dict(model_type="gpt2", n_layer=2)
        raw = study.pack(dict(base_model_identity=old, values=[0.5, 1]))
        with tempfile.TemporaryDirectory() as d:
            for lag in (0, 1, 2):
                md = Path(d) / str(lag)
                r = study.route(md, cfg, old, lag, old, raw, raw)
                self.assertEqual(r["base"] == old, lag == 0)
                self.assertEqual(
                    (md / "lora.json").read_bytes().replace(r["base"].encode(), old.encode()), raw
                )
                self.assertEqual(
                    json.loads((md / "config.json").read_bytes())["model_type"],
                    study.MODEL_TYPE if lag else "gpt2",
                )
        self.assertNotEqual(study.base_id(old, 1), study.base_id(old, 2))

    def test_transpose_and_splice_changes_exactly_one_expected_element(self):
        raw = weights()
        v = study.weight_bits(raw)
        self.assertEqual(study.merge(raw, v), raw)
        base, _, _ = study.pp.Archive(raw).layout["query"]
        v[37 * 32 + 4] = 0x3F500000
        expected = bytearray(raw)
        struct.pack_into("<I", expected, base + 4 * (4 * 96 + 37), v[37 * 32 + 4])
        self.assertEqual(study.merge(raw, v), expected)
        self.assertEqual(study.weight_bits(expected), v)
        v[0] = 0x7FC00000
        with self.assertRaises(ValueError):
            study.merge(raw, v)

    def test_checkpoint_rejects_short_payload_or_foreign_owner(self):
        with self.assertRaises(ValueError):
            study.checkpoint(bytes(3), {}, 1)
        with self.assertRaises(ValueError):
            study.checkpoint(struct.pack("<I", 1000000), {}, 1)
        header = dict(
            schema="spiraltorch.qkv_adamw.v1",
            steps=1,
            layer=0,
            width=32,
            base="a",
            geometry="g",
            lora="l",
            readout="f32_wide_backward",
            adam_bits=list(struct.unpack("<3I", struct.pack("<3f", 0.9, 0.999, 1e-8))),
        )
        h = study.pack(header)
        blob = struct.pack("<I", len(h)) + h + bytes(12 * 3072)
        self.assertEqual(len(study.checkpoint(blob, dict(base="a", lora="l", geometry="g"), 1)), 3072)
        with self.assertRaises(ValueError):
            study.checkpoint(blob, dict(base="b", lora="l", geometry="g"), 1)
