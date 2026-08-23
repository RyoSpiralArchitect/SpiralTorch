# HF blinded semantic review contract v1

This contract closes the qualitative-review path for held-out HF generation
comparisons without moving semantic ownership into an experiment script.

## Ownership

- `st-core::runtime::zspace_semantic_review` validates the packet commitment,
  score bounds, group coverage, response identity, sealed map, and deterministic
  arm/seed aggregation.
- PyO3 exposes those Rust operations without changing their meaning.
- Python owns terminal presentation and atomic draft persistence only.
- A group is the smallest persisted review unit. An interrupted group is not
  saved; every previously completed group remains resumable.
- `response_id` is absent until all packet groups have complete A/B/C scores and
  one preference. Unblinding rejects every incomplete draft.

## Alice replication

The current blinded packet is
`hf_periodic_baseline_replication_pythia70m_alice_semantic_review_packet_20260823.json`.
Its frozen identities are:

- Protocol: `sha256:79cc6f371f4245ffb4764eabbf7393cd8feb2c530c0c6f58bc8b3ae197f5bb38`
- Packet: `sha256:1da36f6fde782b246d5bd7518f4b6ec9e46955a9967edb7429ba9bd1adc136a5`
- Dimensions: 36 groups, 108 candidates, three training arms, seeds 53/59/61

The tracked packet commitment reproduces in Rust. The locally held sealed map
also completed an in-memory synthetic-neutral contract smoke across all 36
groups and 108 candidates. That smoke used score 3 and `tie` everywhere, was
not persisted, and is not semantic evidence.

Human blinded review remains pending. Until a human response is complete and
sealed, no semantic-quality result or superiority claim is ready. Even after
unblinding, the contract establishes structural integrity and deterministic
aggregation, not reviewer blindness, statistical significance, or broad model
efficacy.
