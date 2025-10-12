To inject Redis-derived soft rules ahead of solving:
- import `ability::unison_mediator::soft_from_redis`
- merge with DSL-derived soft rules before calling `solve_soft(...)`
This overlay ships the mediator; wiring depends on your existing `wgpu_heuristics.rs` structure.
