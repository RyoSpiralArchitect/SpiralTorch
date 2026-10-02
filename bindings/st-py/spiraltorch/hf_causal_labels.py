"""HF transport adapter for the first label that causal shifting never predicts."""

from __future__ import annotations

from typing import Any, Callable, Mapping

__all__ = ["HfCausalLabelAlignmentCollator"]


class HfCausalLabelAlignmentCollator:
    """Mask the unused first label without changing causal prediction targets.

    Some Trainer versions count every non-ignored label for accumulated loss,
    even though causal model loss discards position zero. This adapter makes
    that count agree with the shifted targets. It does not make microbatch
    means token-weighted across unequal masks, and is not for pre-shifted labels
    or encoder-decoder/MLM tasks. Input tensors are not mutated.
    """

    def __init__(
        self, base_collator: Callable[[list[dict[str, Any]]], Mapping[str, Any]]
    ) -> None:
        self.base_collator = base_collator

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = dict(self.base_collator(features))
        labels = batch.get("labels")
        if getattr(labels, "ndim", None) != 2 or not callable(
            getattr(labels, "clone", None)
        ):
            raise TypeError("causal label alignment requires rank-2 tensor labels")
        if labels.shape[1] < 2:
            raise ValueError("causal training needs at least two label positions")
        labels = labels.clone()
        labels[:, 0] = -100
        batch["labels"] = labels
        return batch
