import pytest

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
