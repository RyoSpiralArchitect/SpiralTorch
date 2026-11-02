"""File-only diagnostic: did the two fixed branches train the same LoRA at every step?"""

import struct
from control_contract import SEEDS, digest, require, rows, sha


def comparison(a, b):
    require(len(a) == len(b) == 512, "need complete trajectories")

    def feature(row):
        return (row["post_lora_artifact_id"], struct.pack("<f", row["loss"]).hex())

    features = [[feature(r) for r in arm] for arm in (a, b)]
    out = {}
    for i, name in enumerate(("post_lora", "loss_f32_le")):
        aa, bb = ([v[i] for v in seq] for seq in features)
        diffs = [u for u, (l, r) in enumerate(zip(aa, bb, strict=True), 1) if l != r]
        out[name] = {
            "affine_sequence_sha256": digest(aa),
            "wave_sequence_sha256": digest(bb),
            "different_updates": len(diffs),
            "first_different_update": diffs[0] if diffs else None,
        }
    return out


def paired(run, stages):
    results = {}
    for seed in SEEDS:
        arms = []
        for arm in ("affine", "wave"):
            case = f"{seed}-{arm}"
            events = []
            for part in ("first", "second"):
                log = run / case / (part + ".jsonl")
                require(
                    sha(log) == stages[case][part]["steps_sha256"],
                    "different source trajectory",
                )
                events.extend(rows(log))
            require(
                [r["update"] for r in events] == list(range(1, 513)),
                "wrong trajectory order",
            )
            arms.append(events)
        results[str(seed)] = comparison(*arms)
    return results
