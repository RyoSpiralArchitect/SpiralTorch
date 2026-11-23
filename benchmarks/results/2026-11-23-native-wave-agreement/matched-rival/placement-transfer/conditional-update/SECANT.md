# Post-study directional readback, not a new training condition

The primary comparison was completed and its gates evaluated before this
check. Readback SHA-256:
`ec081e8987bb56af70716de5227a9355f0326e419e0b06cb935567685bd917ff`.
On base 521/Wave a local derivative predicted the opposite sign of the
finite head-margin change for several pure head-trained endpoints. To
distinguish a bad local derivative from failure of the linear approximation
over a finite path, fix this diagnostic before examining intermediate logits.

Take order **17**, both schedules, both geometries, all three bases, pure
head teacher: 12 straight lines from immutable origin parameters to the
actual trained endpoint. No selection of endpoints by sign or size.
Restore the frozen decoder and canonical owner from saved bytes. Inspect
the same 16 already-scored test prompts: six shared origins and the 12
rays at t = -1/16, -1/64, -1/256, +1/256, +1/64, +1/16, 1/4, 1/2, 3/4,
1. Each coordinate is f32(origin+t*(end-origin)); all other weights stay
fixed. Exactly 2,016 ordinary forward calls, 0 VJPs/updates. Verify origin
and endpoint logits bit-for-bit against the first run and stop on mismatch.
One local CPU attempt, 120 seconds, no retry or retuning.

For every row use the archived origin VJP J=grad(z7-z8) and predict the
**half difference** of log odds at opposite t as J dot (actual f32
parameters at +t minus those at -t)/2. Use the same head-signed mean as
the training study. For all three scales report observed/predicted H and
the per-prompt RMS signal/error. Summarize the narrowest resolved H scale
where both |observed| and |predicted| exceed 1e-6 nats. Locally consistent
H means equal signs and error <=max(1e-6, 0.15*|prediction|). Independently
select the narrowest row-vector scale with RMS prediction >2e-6; it is
locally consistent if RMS error <=max(2e-6, 0.10*RMS prediction). Keep
unresolved cases distinct from consistent ones. Report complete intermediate
curves, not just the selected scale. These descriptive numerical checks
are not significance levels, guarantees about all decoder gradients or
evidence of transfer. If finite and analytic derivatives disagree even
locally, report that fact instead of attributing the endpoint sign to
curvature. No new real data, training, publication, GPU or free generation.
