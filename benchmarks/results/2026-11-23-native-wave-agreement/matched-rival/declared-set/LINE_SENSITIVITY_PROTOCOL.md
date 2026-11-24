# Can the geometry move other prompts without spending the entire cap on one?

Preregistered at the prompt-identity endpoint. The previous synthetic journal
SHA-256 is `20d430ff3734fb1440bcab0b38163e82112850e9c997b50fc00f52048e9d18bb`.
The matched largest/successor/source-prompt midpoints all refused WaveGate
alone at step one. Do not rerun a pretrained decoder, change weights outside
the selected owner, tune the interval, or choose a different target after
seeing the new outcomes.

Use the identical generated ordinary decoder, fixed nonzero LoRA, WaveGate
and three terminated-answer objectives. T=1/65536 nat, mean in [0.98T,T],
peak <=2T, parameter ceiling 1/8, probe h=1/256. Raw ray first, up to 20
trials. On a bounded miss only, probe the negative loss gradient masked to
each trainable tensor (block) or each trainable scalar (coordinate) at h.
Skip exactly zero source components explicitly. Record every probe's actual
ordered KLs; never search each candidate to convergence. Reuse the origin
bank. Only the single chosen direction gets one more bracket of <=20 trials.

- Score each measurable direction under KL(d) approx KL(h)*(d/h)^2. Choose
  the largest distance not exceeding the parameter ceiling, mean target, or
  per-prompt cap under this model. The envelope selector prefers directions
  predicted to reach the 0.98T floor, then greatest predicted linear loss
  decrease among feasible ones. If none is feasible, maximize the fraction
  of target reachable, then the predicted decrease. Stable lexical block /
  ascending coordinate order breaks ties. A zero KL row cannot predict a
  radius. The matched control takes the highest projected gradient norm
  without looking at KL, **after measuring the same set of probes**.
- Canonical loss order 0,1,2 for two epochs and bank 012. Cross three owners
  with two bases and two selectors (12 trajectories). Compare actual first
  raw misses/ordinary successes with the previous journal wherever they
  share the same state. Primary: does geometry alone take a measured
  first step, and how many of six opportunities finish before refusal?
  If it still stalls, retain the best measured cap-safe mean, the direction
  map and which constraint bound. Not an existence or impossibility proof.
- At the same initial weights start instead with task 1 or 2; geometry only,
  bank still 012, both bases and selectors, one attempt per cell (8). Raw
  success supplies no evidence about any selector. No adaptive extra arms.
- Independently reconstruct all probed geometry directions at the first
  origin from named parameters, gradients and f32 displacements; compare
  actual last-token distributions and rank without using optimizer
  selection as an oracle. Check accepted KL from logits, owner IDs, actual
  source descent on rescue, atomic refusals, and Python/WASM agreement.
  Stop a trajectory on its first refusal, the study on harness/engine error;
  no reattempting a failed study to turn it green. These are in-sample
  synthetic training directions, not independent language-quality samples.
