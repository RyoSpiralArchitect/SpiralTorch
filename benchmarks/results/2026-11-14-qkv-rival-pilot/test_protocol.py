"""Model-free guard for the prospective QKV objective comparison."""

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("qkv_rival_pilot_protocol", HERE / "protocol.py")
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


class ProtocolTests(unittest.TestCase):
    def test_fresh_balanced_splits_and_distinct_orders(self):
        splits, receipt = protocol.freeze()
        self.assertEqual([len(splits[name]) for name in ("train", "eval")], [24, 48])
        self.assertEqual(receipt["recipe"]["arms"], ["ordinary_ce", "top_rival_ce"])
        self.assertEqual(len(set(receipt["orders_sha256"].values())), 3)
        self.assertEqual(len({row["key"] for rows in splits.values() for row in rows}), 72)
        self.assertEqual(receipt["excluded_keys_sha256"],
                         "0cb5d6c7b4e2a3da81d1175e5bec2a4896aad9082fcc1ab5837f7cfc15800c99")
        self.assertEqual(receipt["splits"]["train"]["text_sha256"],
                         "0cf02829405a9bd9d09c263a5003caae86038d9d3acd0a349416127a47fdea6c")
        self.assertEqual(receipt["splits"]["eval"]["text_sha256"],
                         "31d43c815c2e60deda038b75c153c2ec5e2409bd4ee9ef7005e78e00f32eab8f")
        self.assertEqual(receipt["orders_sha256"], {
            "order-31": "c437d2bc32630e3ab9c7cfaae2f567312d8f970c0f45516e4078d5253d917c8a",
            "order-47": "d93edcc6449edc9a3bf134bd615db7c2d719dbfdf36ceb59fce21ea3eb14a281",
            "order-59": "d772c60194a57e5781dc18b4e565c8972312f800f61d183b568795e9e628947a",
        })
        self.assertEqual(protocol.pin(receipt),
                         "5635ff6028119996ea01086b975c8f424ad0b2c58633bf7b02f002b0d5a6129b")
        self.assertEqual(protocol.freeze(), (splits, receipt))

    def test_unknown_order_fails(self):
        with self.assertRaises(ValueError):
            protocol.order_indices([], "chosen-after-scoring")


if __name__ == "__main__":
    unittest.main()
