# Resident learner host diagnosis

The benchmark-only `learn_host_profile` native request and `learnHostProfile`
browser method execute the same custom quadratic/quartic objective as `learn`:
one forward, two independent VJPs and one weighted update. Plain SGD, Topos EMA
and clipped Topos EMA use the same Rust implementations as ordinary intervals.
They add host clocks, not a second optimizer or a production profiling API.

The eight fields separate forward enqueue, each seed and VJP enqueue, update
enqueue, owning receipt capture and receipt wait. They measure **host wall time**,
including resource preparation, encoding and submission. GPU work may overlap
any of these fields. They are not GPU kernel times, copy bandwidth measurements,
or proof that a specific allocation dominates end-to-end training.

Ordinary `learn` specializes the phase clocks out at compile time. It returns no
`host_profile` field. Throughput admission rejects instrumented results instead
of mixing them into an ordinary PyTorch comparison. No additional GPU waits or
timestamp-capable devices are introduced, even in the diagnostic path.

```sh
python tools/profile_resident_learner_host.py \
  --binary /absolute/path/to/frozen/resident_training_bench \
  --source IMMUTABLE_SOURCE_COMMIT \
  --output /absolute/path/to/new-host-profile.json
```

The collector requires a clean source-bound binary and harness. It runs all
three optimizer modes, three seeds, three graph shapes, two observation cadences
and ten alternating instrumented/control blocks (two warmups, eight retained).
Each interval resets weights outside timing and performs eight updates. Phase
counts and duration partitions are checked; every interval's terminal state
fingerprint must match its uninstrumented control. Full states are retained for
the first block. Independent numerical oracles and ordinary throughput runs
remain separate requirements for accepting an optimization.

These diagnostics retain stderr and partial reports on failure. A completed
host profile is not evidence of exclusive GPU ownership or a universal speedup.
