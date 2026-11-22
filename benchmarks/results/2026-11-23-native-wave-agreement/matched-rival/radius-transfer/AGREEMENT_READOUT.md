# Reading a balanced binary verb margin

This is a descriptive secondary readout, not another learning objective or
an amendment to the radius transfer gate. It needs neither a model nor a
tokenizer. The initial GPT-2 capture already existed when this was specified.

For a lexical quartet let s and a be subject and intervening noun number
(-1 singular, +1 plural). Write z = plural logit minus singular logit,
not the gold-oriented margin m = s*z. The four balanced cells determine

    z = B + S*s + A*a + I*s*a.

Compute each coefficient from the four signed logits for the **same noun
pair**, then average the quartets with equal weight. The original primary
mean gold margin is exactly S. Congruent and conflicting means are S+A
and S-A. These identities hold also for final minus each arm's own initial
state. Thus adding a context-independent verb bias B cannot pass the
primary gate by itself. At fixed attractor a, the plural-minus-singular
subject switch is 2*(S+a*I); positive means correct ordering, not two correct
choices. Report the observed gaps, number of correctly ordered pairs and
number with z(singular)<0<z(plural). Count reverse switches, same-verb
choices and any exact boundary ties separately. No thresholds are tuned.

`capture` admits only the prior checksummed training-margin archive. Its
greedy counts are the token IDs already recorded in the full-vocabulary
forward, not two-choice wins, generated sentences, or new model calls.
`completed` requires an already successful, bounded attempt and a separately
retained result digest **before it opens the plan**. It independently replays
the original numerical evidence and checks the original all/conflict/congruent
changes and between-arm gain against the factorial coefficients. It never
writes to the attempt or creates one. Neither mode revises the original
advance decision. The miniature rehearsal has no linguistic interpretation.

```sh
python3 -ISB test_agreement_readout.py -v
python3 -ISB agreement_readout.py capture "$CAPTURE" gpt2 "$NEW_TRAIN_JSON"
python3 -ISB agreement_readout.py completed "$TOY" rehearsal "$TOY_RESULT_SHA256" "$NEW_TOY_JSON"
```
