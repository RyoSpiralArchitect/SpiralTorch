"""Optional Torch host transport shared by Rust-owned geometry bridges."""

from functools import lru_cache
from typing import Any

try:
    import torch
except ImportError:
    torch = None


@lru_cache(maxsize=1)
def _buffer_transport_available() -> bool:
    if torch is None:
        return False
    try:
        torch.empty(0, dtype=torch.float32, device="cpu").numpy()
    except (ImportError, RuntimeError):
        return False
    return True


def _buffer_values(value: Any) -> Any:
    return value.detach().to(device="cpu").resolve_neg().contiguous().numpy()


def _transport_output(values: Any, like: Any, buffers: bool) -> Any:
    if buffers:
        if like.numel() == 0:
            # frombuffer rejects empty storage; empty rows are valid WaveGate inputs.
            return torch.empty_like(like)
        # Keep an export alive so a bytearray cannot resize under a CPU Tensor.
        return torch.frombuffer(memoryview(values), dtype=like.dtype).to(device=like.device).reshape_as(like)
    return torch.tensor(values, dtype=like.dtype, device=like.device).reshape_as(like)
