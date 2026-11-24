# Transfer the decision, not just the terminator

Prospectively fixed after the [six-site experiment](../RESULTS.md), before
observing any outputs for these new artificial weights. No real model,
English examples, model download, GPU or publication.

- Reuse the *symbolic* task and the same 32/16 adjacent/opposite noun-pair
  split in [the parent protocol](../PROTOCOL.md). These prompts are reused
  across studies; they are untrained by the endpoints, not a fresh task.
  Keep its two teachers, order keys 17/31/73, two epochs and 1/256 parameter
  L2 per native update. Do not scan sites: use only block 1 (zero-based)
  MLP output. Train the six bias/gate values of either WaveGate or affine;
  original numeric LoRA, initial gate/bias and strength 0.5 are unchanged.
- Three fresh two-layer, width-three, vocabulary-11 bases (keys 101/211/307).
  Replace **every** f32 in the old fixture with a name/coordinate-keyed
  independent draw. For each scalar, convert the first four SHA-256 bytes
  (big endian) of ASCII `scope-v1:<key>:<tensor-name>:<index>` to
  `u = (word + 0.5)/2**31 - 1`. Embeddings: 0.25*u; layer norm gains:
  1+0.1*u; all biases: 0.04*u; projection matrices: 0.6*u/sqrt(fan_in).
  Round each output to f32. Pin all three archives before training; no model
  selection or adjustment after seeing scores. Rebind identical LoRA A/B
  and geometry parameters to each new identity, freeze the LoRA.
- Cross geometry x scope (answer vs through-stop) x teacher x order on
  **every** base. The two candidates are single tokens 7 and 8; stop=0.
  Loss in both scopes: 50% gold full-vocabulary NLL *per supervised token*
  + 50% NLL conditional on the two candidates. Answer scope uses only the
  predictor immediately preceding the gold token; through-stop uses the
  complete gold line and candidate-specific stop likelihood. No stop loss
  in the first condition, no change of denominator to fake equivalence.
  Same native f32 accumulator policy, same selected-geometry ray for both.
  Check all 4,608 individual displacements against the canonical f32
  parameters (relative error <=1e-4), descent alignment, base/owner pins,
  mask/target counts and unchanged frozen companion. Save owner data,
  gradient/step paths and losses. Check every cell's first loss against
  the independently recorded origin token costs (abs 3e-6).
- Six origins and 72 endpoints, once each on 48 prompts = 3,744 full-menu
  score records. On each snapshot compare the four costs on the first
  training prompt against three direct full-vocabulary forwards (234
  total; abs 3e-6). Separate disjoint rehearsal on base 409 and repeated
  nouns; verify the gradient against finite differences of the actual
  scoped costs there. One formal CPU worker, 240 seconds, stop at first
  error, no retry/retuning. A partial study is not ranked.

## Criteria

On each of the 16 held-out prompts let d be log P(token 7) minus log P(token
8) at the common prompt, **without stop**. H=mean(sign(head)*d);
A=mean(sign(attractor)*d); signs are +1 singular, -1 plural. Every change
is relative to the same base/geometry origin. For head training use h=delta H;
specificity S=((delta H - delta A) under head training minus that under
attractor training)/2. For each base/geometry require in **every** order:
h(answer) and S(answer) both >0.001 nats, their respective advantages over
through-stop both >0.001, and strictly positive h(answer) in at least 3/4
held-out pairs. Claim probability transfer for that geometry only if at least
two of three bases pass. Separately claim **decision transfer** only if two
such passing bases also gain at least one correct first-token choice versus
both origin and through-stop in every order (ties do not count as wins).
Claim a Wave-specific advantage only if on at least two Wave-passing bases
the paired scope advantages in **both** h and S exceed affine's by >0.001
in every order. Report all bases, misses, wins/ties and the answer/stop
contributions. No threshold changes, no pooling orders as independent bases.

Secondary diagnostics: on head-trained endpoints compare conditional
candidate odds, gold full-vocabulary answer NLL, the complete-line margin
and first-token switches. Relate the maximum observed first-token log-odds
movement to the minimum absolute initial margin (a sufficient, not necessary,
no-switch condition). Report the training split separately from the test
split. No claim about open generation, language quality or pretrained models.
