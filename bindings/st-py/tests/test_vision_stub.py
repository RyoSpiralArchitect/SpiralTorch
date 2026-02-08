import pytest

import spiraltorch as st
from spiraltorch import SpiralTorchVision


def test_spiraltorchvision_zero_weight_no_update():
    vision = SpiralTorchVision(depth=2, height=2, width=2, alpha=0.5)

    first_frame = [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ]
    vision.accumulate(first_frame)
    baseline = vision.volume

    second_frame = [
        [[9.0, 9.0], [9.0, 9.0]],
        [[9.0, 9.0], [9.0, 9.0]],
    ]
    vision.accumulate(second_frame, weight=0.0)

    assert vision.volume == baseline


def test_spiraltorchvision_negative_weight_rejected():
    vision = SpiralTorchVision(depth=1, height=1, width=1, alpha=0.5)

    with pytest.raises(ValueError):
        vision.accumulate([[[1.0]]], weight=-0.1)


def test_stream_vision_training_with_trainer_updates_state():
    vision = SpiralTorchVision(depth=2, height=2, width=2, alpha=0.35, temporal=3)
    trainer = st.ZSpaceTrainer(z_dim=4, lr=5e-3)
    frames = [
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.2, 0.1], [0.4, 0.3]],
        ],
        [
            [[0.4, 0.3], [0.2, 0.1]],
            [[0.1, 0.3], [0.2, 0.4]],
        ],
    ]

    updates = st.vision.stream_vision_training(
        vision,
        frames,
        trainer=trainer,
        flush_every=1,
    )

    assert len(updates) == 2
    assert all(bool(update.get("applied")) for update in updates)
    assert updates[-1]["loss"] is not None
    assert isinstance(updates[-1]["z_state"], list)


def test_stream_vision_training_flush_every_with_custom_aggregator():
    class _EchoAggregator:
        def __init__(self) -> None:
            self._last = None

        def extend(self, frame):
            self._last = frame

        def to_streamed_volume(self):
            return self._last

    vision = SpiralTorchVision(depth=1, height=1, width=1, alpha=0.5)
    aggregator = _EchoAggregator()
    frames = [[[[1.0]]], [[[2.0]]], [[[3.0]]]]

    updates = st.vision.stream_vision_training(
        vision,
        frames,
        aggregator=aggregator,
        flush_every=2,
        final_flush=True,
    )

    assert [int(update["aggregated_frames"]) for update in updates] == [2, 1]
