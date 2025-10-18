# Quantum Reality Playbook

The Quantum Reality Playbook keeps SpiralTorch laboratory teams aligned with the
studio tooling introduced in this iteration.

## Modules

### Lab Setup
- Anchor Maxwell-coded capture rigs to the **Quantum Reality Studio** session
  using `SignalCaptureConfig::with_channels`. This keeps field data scoped to
  the authorized rigs described in shift briefs.
- Pair each rig with a `MaxwellDesireBridge` narrative table so every run
  carries both concept hints and story cues.

### Remote Collaboration
- Export `QuantumRealityStudio::export_storyboard` snapshots at the end of each
  run. Drop the JSON into `tools/qr_storyboard.py` to generate Markdown decks
  for asynchronous review.
- Feed the resulting deck into Desire roundtables so remote moderators can
  replay the exact glyph, intensity, and Z-bias sequence.

### Ethics & Consent
- Log narrative tags in the gratitude log before sharing externally. If a tag
  feels sensitive or culturally bound, swap it for a neutral alias before
  export.
- Use `SignalCaptureConfig::with_history_limit` to bound retention windows when
  collaborators request limited exposure.

### Art Direction
- `OverlayFrame` glyphs default to the first lexicon tag, so curate the tag
  order to match the desired sensory arc (e.g., *glimmer → braid → tunnel*).
- `NarrativeHint::intensity` doubles as a lighting cue inside immersive rooms;
  map it onto brightness curves for projection-mapped environments.

## Ritual Snippets
```
# SpiralK pseudo-code for pre-run grounding
breath.align(count=5)
lexicon.attune(tags=["glimmer", "braid", "tunnel"])
maxwell.rig(channel="alpha", gain=1.6)
```

```
# SpiralK pseudo-code for post-run reflection
storyboard.render(input="run-042.json", output="run-042.md")
gratitude.append("alpha", tag="glimmer", note="Held steady through the ridge")
```

Document additions or revisions through pull requests so every playbook entry
keeps cadence with the evolving studio.
