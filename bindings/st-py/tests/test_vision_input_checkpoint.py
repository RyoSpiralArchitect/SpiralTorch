"""Public input continuation API; run against a freshly built wheel."""
import hashlib
import json
import os
from pathlib import Path
import struct
import unittest

import spiraltorch as st


def pipeline():
    value = st.TransformPipeline(seed=29)
    value.add_horizontal_flip(0.5)
    value.add_normalize([0.5], [0.25])
    return value


def fixture():
    dataset = st.TensorVisionDataset("mnist")
    digest = hashlib.sha256()
    for index in range(11):
        values = [(index * 13 + j * 7) % 101 / 101 for j in range(16)]
        digest.update(struct.pack("<16fI", *values, index))
        dataset.push(st.ImageTensor(1, 4, 4, values), label=str(index))
    return dataset, digest.hexdigest()


def next_batch(loader):
    batch = loader.next_batch()
    if batch is None:
        loader.reset()
        batch = loader.next_batch()
    return batch.labels(), [image.data() for image in batch.images()]


class InputCheckpointTests(unittest.TestCase):
    def test_transform_restores_rng_but_rejects_different_configuration(self):
        source = pipeline()
        image = st.ImageTensor(1, 4, 4, list(range(16)))
        source.apply(image)
        state = source.checkpoint_json()
        expected = [source.apply(image).data() for _ in range(20)]
        if path := os.environ.get("SPIRALTORCH_INPUT_CHECKPOINT_HANDOFF"):
            with Path(path).open("x") as out:
                json.dump(dict(state=state, image=image.data(), expected=expected,
                               final_state=source.checkpoint_json()), out, allow_nan=False)
        resumed = pipeline()
        resumed.restore_checkpoint_json(state)
        self.assertEqual([resumed.apply(image).data() for _ in range(20)], expected)
        resumed.add_horizontal_flip(0.5)
        before = resumed.checkpoint_json()
        with self.assertRaises(ValueError):
            resumed.restore_checkpoint_json(state)
        self.assertEqual(before, resumed.checkpoint_json())

    def test_loader_resumes_across_epochs_and_rejects_wrong_identity_atomically(self):
        dataset, identity = fixture()
        make = lambda: dataset.dataloader(3, seed=17, pipeline=pipeline(), shuffle=True)
        source = make()
        for _ in range(37):
            next_batch(source)
        state = source.checkpoint_json(identity)
        expected = [next_batch(source) for _ in range(63)]
        final = source.checkpoint_json(identity)
        del source
        resumed = make()
        before = resumed.checkpoint_json(identity)
        with self.assertRaises(ValueError):
            resumed.restore_checkpoint_json("f" * 64, state)
        self.assertEqual(before, resumed.checkpoint_json(identity))
        resumed.restore_checkpoint_json(identity, state)
        self.assertEqual([next_batch(resumed) for _ in range(63)], expected)
        self.assertEqual(resumed.checkpoint_json(identity), final)
        invalid = json.loads(state)
        invalid["order"][1] = invalid["order"][0]
        with self.assertRaises(ValueError):
            resumed.restore_checkpoint_json(identity, json.dumps(invalid))
        self.assertEqual(resumed.checkpoint_json(identity), final)

    @unittest.skipUnless(os.environ.get("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS") == "1", "explicit GPU run required")
    def test_resident_pipeline_retains_device_and_replays_output(self):
        device = st.WgpuTensorDevice.create()
        self.assertNotEqual(device.adapter_info()["device_type"], "Cpu")
        source = pipeline()
        source.enable_wgpu()
        images = [st.ImageTensor(1, 4, 4, list(range(16)))]
        source.apply_resident_batch(images, device).snapshot().read_values()
        state = source.checkpoint_json()
        expected = source.apply_resident_batch(images, device).snapshot().read_values()
        resumed = pipeline()
        resumed.enable_wgpu()
        resumed.restore_checkpoint_json(state)
        self.assertTrue(resumed.has_gpu_dispatcher())
        self.assertEqual(resumed.apply_resident_batch(images, device).snapshot().read_values(), expected)


if __name__ == "__main__":
    unittest.main()
