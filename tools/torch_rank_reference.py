"""Finite-f32 benchmark controls, not a SpiralTorch runtime implementation.

Admission is outside timing and bound to one immutable input fixture. All
input-dependent key construction and index repair run inside every operation.
"""
import math
from statistics import mean
import struct
from time import perf_counter


def float32(value):
    try:
        rounded = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, TypeError, struct.error) as error:
        raise ValueError("rank controls require finite f32 inputs") from error
    if not math.isfinite(rounded):
        raise ValueError("rank controls require finite f32 inputs")
    return rounded


def total_key(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return (~bits & 0xffffffff) if bits & 0x80000000 else bits ^ 0x80000000


def contract(request):
    kind = request["kind"]
    if kind not in ("topk", "midk", "bottomk"):
        raise ValueError("unknown rank kind")
    rows, cols, k = (request[name] for name in ("rows", "cols", "k"))
    if (any(type(value) is not int or value <= 0 for value in (rows, cols, k))
            or k > cols or cols >= 2**32 or len(request["input"]) != rows * cols):
        raise ValueError("invalid rank geometry or input cardinality")
    values = [float32(value) for value in request["input"]]
    topk_safe = repair_safe = kind != "midk"
    stable_safe = True
    start = (cols - k) // 2 if kind == "midk" else 0
    expected_values, expected_indices = [], []
    for row in range(rows):
        data = values[row * cols:(row + 1) * cols]
        zeros = {total_key(value) for value in data if value == 0}
        order = sorted(range(cols), key=lambda index: (
            -total_key(data[index]) if kind == "topk" else total_key(data[index]), index))
        ids = order[start:start + k]
        window_matches = True
        # Stable numeric order only diverges at mixed zeros; inspect the output, not the whole row.
        if len(zeros) == 2:
            numeric_order = sorted(range(cols), key=lambda index: -data[index] if kind == "topk" else data[index])
            window_matches = numeric_order[start:start + k] == ids
        stable_safe &= window_matches
        prefix = [data[index] for index in order[:min(k + 1, cols)]]
        topk_safe &= all(a != b for a, b in zip(prefix, prefix[1:]))
        cutoff_unique = k == cols or prefix[k - 1] != prefix[k]
        repair_safe &= cutoff_unique and window_matches
        expected_indices.extend(ids)
        expected_values.extend(data[index] for index in ids)
    operations = []
    if topk_safe:
        operations.append("topk")
    if repair_safe and not topk_safe:
        operations.append("topk_index_repair")
    if stable_safe:
        operations.append("stable_sort")
    operations.append("packed_topk")
    return dict(kind=kind, rows=rows, cols=cols, k=k, start=start, input=values,
                operations=operations, values=expected_values, indices=expected_indices)


class RankControl:
    """Preallocated eager-Torch operation for a contract-bound resident fixture."""

    def __init__(self, torch, device, admitted, operation):
        if operation not in admitted["operations"]:
            raise ValueError("CUDA rank operation is not admitted for this fixture")
        if (device.dtype != torch.float32 or device.requires_grad or not device.is_contiguous()
                or tuple(device.shape) != (admitted["rows"], admitted["cols"])):
            raise ValueError("rank control requires a contiguous inference-only f32 matrix")
        self.torch, self.device, self.operation = torch, device, operation
        self.k, self.start = admitted["k"], admitted["start"]
        self.largest = admitted["kind"] == "topk"
        rows, cols = device.shape
        self.values = torch.empty((rows, self.k), dtype=torch.float32, device=device.device)
        self.indices = torch.empty((rows, self.k), dtype=torch.int64, device=device.device)
        if operation == "stable_sort":
            self.sorted_values = torch.empty_like(device)
            self.sorted_indices = torch.empty_like(device, dtype=torch.int64)
            self.values = self.sorted_values[:, self.start:self.start + self.k]
            self.indices = self.sorted_indices[:, self.start:self.start + self.k]
        elif operation == "topk_index_repair":
            self.raw_values = torch.empty_like(self.values)
            self.raw_indices = torch.empty_like(self.indices)
            self.ordered_values = torch.empty_like(self.values)
            self.ordered_indices = torch.empty_like(self.indices)
            self.permutation = torch.empty_like(self.indices)
        elif operation == "packed_topk":
            self.bits = device.view(torch.int32)
            self.keys = torch.empty_like(device, dtype=torch.int64)
            self.mask = torch.empty_like(self.keys)
            # Geometry only: never cache value-dependent keys across operations.
            positions = torch.arange(cols, dtype=torch.int64, device=device.device)
            self.tie_break = cols - 1 - positions if self.largest else positions
            self.take = self.start + self.k
            self.selected_keys = torch.empty((rows, self.take), dtype=torch.int64, device=device.device)
            self.selected_indices = torch.empty_like(self.selected_keys)
            self.indices = self.selected_indices[:, self.start:self.start + self.k]

    def run(self):
        torch = self.torch
        if self.operation == "stable_sort":
            torch.sort(self.device, dim=1, descending=self.largest, stable=True,
                       out=(self.sorted_values, self.sorted_indices))
        elif self.operation == "topk":
            torch.topk(self.device, self.k, dim=1, largest=self.largest, sorted=True,
                       out=(self.values, self.indices))
        elif self.operation == "topk_index_repair":
            torch.topk(self.device, self.k, dim=1, largest=self.largest, sorted=False,
                       out=(self.raw_values, self.raw_indices))
            torch.sort(self.raw_indices, dim=1, out=(self.ordered_indices, self.permutation))
            torch.gather(self.device, 1, self.ordered_indices, out=self.ordered_values)
            torch.sort(self.ordered_values, dim=1, descending=self.largest, stable=True,
                       out=(self.values, self.permutation))
            torch.gather(self.ordered_indices, 1, self.permutation, out=self.indices)
        else:
            # Signed high word orders finite f32 bits, including -0 < +0.
            # The low word orders source indices, also across a tied cutoff.
            self.keys.copy_(self.bits)
            torch.bitwise_right_shift(self.keys, 63, out=self.mask)
            self.mask.bitwise_and_(0x7fffffff)
            self.keys.bitwise_xor_(self.mask)
            self.keys.bitwise_left_shift_(32)
            self.keys.bitwise_or_(self.tie_break)
            torch.topk(self.keys, self.take, dim=1, largest=self.largest, sorted=True,
                       out=(self.selected_keys, self.selected_indices))
            torch.gather(self.device, 1, self.indices, out=self.values)


def measure(request, device, torch, summarize):
    """Retain every eligible control; best_fixed is explicitly hindsight, not policy."""
    if device.device.type != "cuda":
        raise ValueError("matched rank timing requires CUDA, not a fallback")
    admitted = contract(request)
    host = torch.tensor(admitted["input"], dtype=torch.float32).reshape(device.shape)
    if not torch.equal(device.cpu().view(torch.int32), host.view(torch.int32)):
        raise ValueError("CUDA fixture differs from admitted f32 input bits")
    expected_values = torch.tensor(admitted["values"], dtype=torch.float32).reshape(admitted["rows"], admitted["k"])
    expected_indices = torch.tensor(admitted["indices"], dtype=torch.int64).reshape(expected_values.shape)
    controls = {name: RankControl(torch, device, admitted, name) for name in admitted["operations"]}
    samples = {name: [] for name in controls}
    pairs = []

    def check(control):
        if (not torch.equal(control.values.cpu().view(torch.int32), expected_values.view(torch.int32))
                or not torch.equal(control.indices.cpu(), expected_indices)):
            raise ValueError(f"CUDA {control.operation} differs from canonical value bits/source indices")

    for control in controls.values():
        control.run()
        check(control)
    names = list(controls)
    for block in range(14):
        offset = (block + request["seed"]) % len(names)
        order = names[offset:] + names[:offset]
        if (block // len(names)) % 2:
            order.reverse()
        pair = dict(order=order, per_op_ms={})
        for name in order:
            control = controls[name]
            torch.cuda.synchronize(device.device)
            began = perf_counter()
            for _ in range(16):
                control.run()
            torch.cuda.synchronize(device.device)
            elapsed = (perf_counter() - began) * 1000 / 16
            check(control)
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("invalid CUDA control timing")
            if block >= 2:
                samples[name].append(elapsed)
                pair["per_op_ms"][name] = elapsed
        if block >= 2:
            pairs.append(pair)
    best = min(names, key=lambda name: mean(samples[name]))
    return dict(schema="spiraltorch.cuda_rank_controls.v1", operations=names,
                repetitions=16, warmup_batches=2, retained_batches=12,
                samples=pairs, timings={name: summarize(values) for name, values in samples.items()},
                best_fixed=best, selection="hindsight lowest sample-mean latency among eligible controls; not an online policy",
                boundary="eager Torch enqueue plus completion; includes key encoding/index repair/gather on every operation; no cached value keys; preallocated buffers; validation readbacks outside timing")
