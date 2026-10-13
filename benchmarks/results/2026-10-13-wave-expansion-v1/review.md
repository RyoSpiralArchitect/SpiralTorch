# Independent Read-Only Wave Follow-Up

Scope: current CPU stored-contraction expansion, WGSL scaled-scalar reduction,
rounding/capacity, and source exponent range. Not a fresh whole-PR review.
The reviewer did not edit files, compile, or execute GPU jobs. Runtime evidence
was collected separately by the parent.

Initial P2 finding: [1,20_000_000,2], decay=-80, phase=0, zero feature seeds and
nonzero terminal adjoint can wrap the unchecked i32 Wide exponent despite
otherwise valid dimensions and sufficiently large device limits.

Resolution: CausalWaveSpec rejects T > 1<<20 before allocation. Review found
512 exponent units per reverse transition conservative: two stored-f32
coefficient multiplications allow about150 downward units each; normalization
after rotation cancellation adds at most148. Chart-seed addition supplies a
fixed cancellation floor rather than another recurrent multiplier.
With loose fixed headroom H=65,536, M=512*2^20+H=536,936,448;
2M+1,024=1,073,873,920 remains below i32::MAX, covering signed differences and
product_parts shifts. Each recurrent product has one evolving adjoint
exponent, not two. Changes to Wide or recurrence dependencies reopen this bound.
This is a conservative engineering range argument, not exact real recurrence.

The follow-up found no new actionable issue in candidate collapse, retained
midpoint comparison, signed subnormal/overflow rounding, or scratch capacity.
Capacity: 4N + 4P(3BT+2) = 10N+8P words; two spare scalar slots allow both
rounding insertions. Temporary debug probe writes are not qualification.

Final source reviewed before commit8bc573ccf552ebc3e555ff2ad4a794d9d51b0088.
No actionable findings remained within this bounded review.
