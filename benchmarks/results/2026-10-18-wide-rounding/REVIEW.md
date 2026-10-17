# Independent Review

A read-only local reviewer inspected the shared WGSL converter, integer oracle,
native/browser transport and analytical Attention expectations before the long
learning rerun. No actionable defect was found for retained normalized finite
components. The reviewer independently checked all 7,042 expectations with exact
rationals, 8,192 additional boundary-biased triples, 2,048 addition/TwoSum probes,
and all twelve Attention expectations. These were CPU checks, not GPU evidence.

Two qualifications remain explicit:

- The synthetic converter fixture bypasses normalization. Some triples are not
  canonical outputs of production normalization; the 582 baseline fixture
  mismatches are not 582 demonstrated reachable model failures. The four actual
  Attention midpoint failures separately establish production reachability.
- Earlier arithmetic can discard a decisive component. For example,
  `wide_normalize(0x3f7fffff, 0x33000000, 0x80000001, exponent=1)` halves a minimum
  subnormal tail to zero. The exact original sum rounds to `0x3fffffff`; the
  retained normalized sum rounds to `0x40000000`. This repair converts the
  retained representation correctly; it does not repair that prior loss.

The production change is limited to `wide_float`. No gradient masks, optimizer
policy changes, tolerance changes or full-model qualification are implied.
