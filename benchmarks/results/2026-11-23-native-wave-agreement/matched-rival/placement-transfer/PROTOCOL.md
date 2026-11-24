# Which geometry site learns a conditional answer, at the same parameter distance?

Frozen before observing the role-task scores. This is a new experiment on one
generated two-layer, width-three, 11-token decoder, not the previous signed
bias probe or the real GPT-2 study. Keep originals. No pretrained model,
old evaluation file, download, GPU, external ML package or publication.

## Task and arms

- Prompt IDs: `[head noun, head number, 9, attractor noun, attractor number, 10]`.
  Nouns 3..6 form a four-cycle. Number 1 means singular, 2 plural.
  The answers are exactly `[7,0]` and `[8,0]` (0 is the terminator).
  Use all four number combinations on each ordered noun pair. Train on the
  eight adjacent ordered pairs; evaluate on the four opposite ordered pairs.
  No repeated noun within a prompt, no shared ordered pair between splits.
  Both numbers, each noun in each role, correct line length and label counts
  are balanced. These 32/16 symbolic prompts are not the earlier English data.
- Train on the **head** number or, as the matched control, on the **attractor**
  number. Only the gold index changes. The two rules disagree on exactly
  half of the examples. Objective: native complete-line menu loss, mix 0.5
  (gold full-vocabulary NLL per token and menu-conditional NLL). Max tile 11.
- Fixed generated base and nonzero attention-output LoRA from the existing
  ordinary/wave fixture. Promote LoRA into the base; compare attention output,
  MLP output and block output at each of two layers. One trainable WaveGate
  or one channel-affine adapter with the exact same gate/bias, width,
  residual strength and value limit. Six scalars in every geometry arm;
  no radius updates. Start each trajectory from its own immutable origin.
  Auxiliary reference: train the original 12-scalar LoRA **without** geometry.
  This reference is not a parameter-count-matched geometry comparison.
- Three order keys 17, 31, 73. Sort keys by SHA-256 of ASCII
  `<order>:<row-key>`, break ties by key; repeat the same permutation for two
  epochs. Each Rust update follows its current loss gradient with requested
  parameter L2 distance 1/256. Independently measure realized distance on
  canonical f32 parameters (relative error <= 1e-4). Check selected group,
  descent dot product, immutable companion and base. Endpoints need not be
  equidistant, nor need forward KL or runtime be equal. Record endpoint L2.
- 13 origins x 2 rules x 3 orders x 64 native updates = 4,992 updates.
  Score all 48 prompts only at each origin and endpoint (4,368 label-free
  complete-menu readouts). Check the first training prompt at each such
  snapshot against three independent full-vocabulary forwards: the two
  first-token costs and the two corresponding stop costs (3e-6 nats).
  A separate, disjoint tiny rehearsal may exercise the runner, not the
  primary score table. One formal CPU worker, 240 s, no retry or retuning;
  on any error stop, retain partial journals and do not rank incomplete arms.
  Archive final owner checkpoints and the complete realized step path.

## Interpretation fixed in advance

Let d be the complete singular line's log probability minus the plural line's
(including stop), s(number)=+1 for singular, -1 for plural. H=mean(s(head)*d),
A=mean(s(attractor)*d). Margins are in nats. A prompt-independent answer
bias cancels in both. Break H into content and termination contributions.
For each site/recipe/order, delta means after minus the **same origin**.
Primary evidence is on unseen noun pairs, not the training split.

Define head-specific effect B = delta H(head-trained) - delta H(attractor-trained).
Also measure S = ((delta H - delta A)(head-trained) -
(delta H - delta A)(attractor-trained))/2. Report head gold-NLL improvement
(before minus after, per full-vocabulary token), binary line wins/ties, and
how much of the gain is menu coverage versus within-menu choice. For a
**placement candidate** require in *every* order: delta H(head-trained), B,
S and head gold-NLL improvement each strictly > 0.001; also require positive
delta H in at least 3 of 4 evaluation noun pairs in each order. This 0.001
floor is a descriptive minimum, not a significance level. Label order keys
are not independent models, and four lexical pairs are not a language
benchmark. Report misses and all sites, not just the highest median.

Call a Wave-specific advantage only if the Wave site clears the candidate
gate and delta H **and** S exceed the corresponding affine gains by > 0.001
in every matched order. If no site clears either gate, close that claim
negatively *for this experiment*; do not lower the floor or choose a new
metric. A change in answer mass/termination alone is not role learning.
