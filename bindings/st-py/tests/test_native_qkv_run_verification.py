"""Model-free integrity checks for the native QKV file-run output protocol."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


SOURCE = Path(__file__).parents[1] / "spiraltorch" / "native_qkv_runs.py"
spec = importlib.util.spec_from_file_location("native_qkv_runs_under_test", SOURCE)
runs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runs)


def encode(header, payload=b""):
    data = json.dumps(header, separators=(",", ":")).encode()
    return len(data).to_bytes(4, "little") + data + payload


def pin(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def projection_id(base, weights):
    digest = hashlib.sha256(b"spiraltorch.qkv_projection.f32.v1\0")
    for value in (base, "", "", "f32"):
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    digest.update((0).to_bytes(8, "little"))
    digest.update(weights)
    return "sha256:" + digest.hexdigest()


class QkvRunVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {"context_length": 4, "batch_windows": 1, "epochs": 1}
        self.data_id = "sha256:" + "a" * 64
        self.base_id = "sha256:" + "b" * 64
        self.result = {
            "schema": "spiraltorch.native_qkv_run.v1",
            "data_id": self.data_id,
            "config": self.config,
            "initial_update": 0,
            "updates": 1,
            "total_updates": 1,
            "active_windows": 1,
            "exhausted": True,
            "checkpoint_index": "checkpoints.jsonl",
            "step_log": "steps.jsonl",
        }
        index = []
        for update in (0, 1):
            weights = struct.pack("<3f", float(update), 0.25, -0.5)
            inner = encode({
                "schema": "spiraltorch.qkv_adamw.v1", "base": self.base_id,
                "lora": None, "geometry": None, "readout": "f32", "layer": 0,
                "width": 1, "steps": update, "adam_bits": [0, 0, 0],
            }, weights + bytes(24))
            outer = encode({
                "schema": "spiraltorch.native_causal_lm_qkv_trainer.v1",
                "data_id": self.data_id,
                "config": self.config,
                "updates": update,
                "initial_steps": 0,
                "optimizer_id": pin(inner),
            }, inner)
            name = f"checkpoint-{update:09d}.bin"
            (self.root / name).write_bytes(outer)
            index.append({"file": name, "checkpoint_id": pin(outer), "updates": update})
        self.result["final_checkpoint"] = index[-1]
        self.result["qkv"] = {
            "file": "qkv.bin", "checkpoint_id": pin(inner),
            "optimizer_steps": 1, "layer": 0,
            "projection_id": projection_id(self.base_id, weights),
        }
        (self.root / "qkv.bin").write_bytes(inner)
        (self.root / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index))
        (self.root / "steps.jsonl").write_text('{"update":1}\n')
        self.write_result()

    def write_result(self):
        (self.root / "result.json").write_text(json.dumps(self.result) + "\n")

    def test_complete_run(self):
        self.assertEqual(runs.verify_qkv_run(self.root), self.result)

    def test_rejects_missing_truncated_or_extra_log(self):
        path = self.root / "steps.jsonl"
        for data in (b"", b'{"update":1}', b'{"update":1}\n{"update":2}\n'):
            with self.subTest(data=data):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    runs.verify_qkv_run(self.root)

    def test_rejects_incomplete_marker_or_index(self):
        marker = self.root / "result.json"
        marker.write_bytes(marker.read_bytes().rstrip(b"\n"))
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)
        self.write_result()
        (self.root / "checkpoints.jsonl").write_bytes(b"")
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)

    def test_rejects_result_or_file_changes(self):
        path = self.root / "qkv.bin"
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)
        path.unlink()
        with self.assertRaises(OSError):
            runs.verify_qkv_run(self.root)

    def test_rejects_resealed_export_that_differs_from_final_state(self):
        path = self.root / "qkv.bin"
        data = path.read_bytes() + b"different"
        path.write_bytes(data)
        self.result["qkv"]["checkpoint_id"] = pin(data)
        self.write_result()
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)

    def test_rejects_resealed_inconsistent_checkpoint(self):
        path = self.root / "checkpoint-000000001.bin"
        data = bytearray(path.read_bytes())
        length = int.from_bytes(data[:4], "little")
        header = json.loads(data[4:4 + length])
        header["data_id"] = "sha256:" + "b" * 64
        data = encode(header, data[4 + length:])
        path.write_bytes(data)
        self.result["final_checkpoint"]["checkpoint_id"] = pin(data)
        index = [json.loads(row) for row in (self.root / "checkpoints.jsonl").read_text().splitlines()]
        index[-1] = self.result["final_checkpoint"]
        (self.root / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index))
        self.write_result()
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)

    def test_rejects_path_substitution_and_duplicate_fields(self):
        self.result["qkv"]["file"] = "../other.bin"
        self.write_result()
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)
        self.result["qkv"]["file"] = "qkv.bin"
        self.write_result()
        (self.root / "steps.jsonl").write_text('{"update":1,"update":1}\n')
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)

    def test_rejects_nonfinite_result_and_symlinked_files(self):
        result = self.root / "result.json"
        result.write_text(result.read_text().replace('"total_updates": 1', '"total_updates": NaN'))
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)
        self.write_result()
        qkv = self.root / "qkv.bin"
        target = self.root / "export-copy.bin"
        qkv.rename(target)
        try:
            qkv.symlink_to(target)
        except OSError:
            self.skipTest("symlinks unavailable")
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)

    def test_rejects_overflowing_numeric_log_metric(self):
        (self.root / "steps.jsonl").write_text('{"update":1,"optimizer":{"objective_loss":1e999}}\n')
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)

    def test_rejects_rewritten_plan_and_completion_status(self):
        self.result["total_updates"] = 2
        self.result["exhausted"] = False
        self.write_result()
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)

    def test_rejects_resealed_optimizer_owner_splice(self):
        path = self.root / "checkpoint-000000000.bin"
        data = path.read_bytes()
        header_length = int.from_bytes(data[:4], "little")
        header = json.loads(data[4:4 + header_length])
        inner = data[4 + header_length:]
        optimizer_length = int.from_bytes(inner[:4], "little")
        optimizer = json.loads(inner[4:4 + optimizer_length])
        optimizer["base"] = "sha256:" + "c" * 64
        inner = encode(optimizer, inner[4 + optimizer_length:])
        header["optimizer_id"] = pin(inner)
        data = encode(header, inner)
        path.write_bytes(data)
        index_path = self.root / "checkpoints.jsonl"
        rows = [json.loads(row) for row in index_path.read_text().splitlines()]
        rows[0]["checkpoint_id"] = pin(data)
        index_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)

    def test_rejects_wrong_or_missing_projection_id(self):
        for value in ("sha256:" + "0" * 64, None):
            with self.subTest(value=value):
                self.result["qkv"]["projection_id"] = value
                self.write_result()
                with self.assertRaises(ValueError):
                    runs.verify_qkv_run(self.root)

    def test_rejects_changed_clock_origin_even_with_resealed_files(self):
        path = self.root / "checkpoint-000000001.bin"
        data = path.read_bytes()
        header_length = int.from_bytes(data[:4], "little")
        header = json.loads(data[4:4 + header_length])
        inner = data[4 + header_length:]
        optimizer_length = int.from_bytes(inner[:4], "little")
        optimizer = json.loads(inner[4:4 + optimizer_length])
        optimizer["steps"] = 2
        inner = encode(optimizer, inner[4 + optimizer_length:])
        header["initial_steps"] = 1
        header["optimizer_id"] = pin(inner)
        data = encode(header, inner)
        path.write_bytes(data)
        index_path = self.root / "checkpoints.jsonl"
        rows = [json.loads(row) for row in index_path.read_text().splitlines()]
        rows[-1]["checkpoint_id"] = pin(data)
        index_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        (self.root / "qkv.bin").write_bytes(inner)
        self.result["final_checkpoint"] = rows[-1]
        self.result["qkv"]["checkpoint_id"] = pin(inner)
        self.result["qkv"]["optimizer_steps"] = 2
        self.write_result()
        with self.assertRaises(ValueError):
            runs.verify_qkv_run(self.root)


if __name__ == "__main__":
    unittest.main()
