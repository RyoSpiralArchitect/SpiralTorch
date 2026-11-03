"""Device-free contracts for the resident, unique-input embedding probe."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import unittest

import spiraltorch as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from test_byte_corpus_study import fixture


class EmbeddingProbeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = fixture()[0]
        cls.study = st.nn.ByteCorpusStudy.from_json(cls.raw)

    def test_installed_export_is_opaque(self):
        exports = {}
        exec("from spiraltorch.nn import *", exports)
        self.assertIs(exports["ByteEmbeddingObservation"], st.nn.ByteEmbeddingObservation)
        with self.assertRaises(TypeError):
            st.nn.ByteEmbeddingObservation()

    def test_selection_binds_exact_input_and_coordinate(self):
        select = self.study.embedding_update_selection_json
        self.assertEqual(json.loads(select("geometry", 1, 2, 0)), {
            "request_sha256": hashlib.sha256(self.raw).hexdigest(),
            "case": "geometry", "update": 1, "token": 2, "feature": 0,
            "input_position": [0, 1],
            "rate_f32_bits": struct.unpack("<I", struct.pack("<f", 0.05))[0],
        })

    def test_invalid_selections_fail_before_runtime_acquisition(self):
        for select in (
            self.study.embedding_update_selection_json, self.study.run_wgpu_with_embedding_probe,
        ):
            for selection in [
                ("missing", 1, 2, 0), ("geometry", 0, 2, 0), ("geometry", 1, 3, 0),
                ("geometry", 2, 2, 0), ("geometry", 1, 2, 2), ("geometry", 1, 256, 0),
            ]:
                with self.subTest(selection=selection), self.assertRaises(ValueError):
                    select(*selection)
            for args in [(True, 2, 0), (1, 2.0, 0), (1, 2, False)]:
                with self.assertRaises(TypeError):
                    select("geometry", *args)
