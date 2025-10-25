# Interactive runtime handoff explorer

This directory collects standalone HTML explorations that visualise pieces of the
runtime diagrams from the main README. Each document can be opened directly in
a browser without a build step, and many now include guided story beats so you
can feel the handoff rhythm without memorising the sequence diagram.

Current prototypes:

- `return_handles_graph_node.html` – highlights how the "graph node"
  materialisation and "return handles" delivery flow through the dispatcher.
  Click the buttons at the top, launch the animated story tour, or poke
  individual nodes to explore the sequence.

## Getting started

Open the HTML files from your file browser or run a local web server:

```bash
python -m http.server --directory docs/interactive
```

Then navigate to <http://localhost:8000/return_handles_graph_node.html>.
You'll be greeted by the "Graph node & return handle handoff" experience,
complete with:

- a six-beat **Story Tour** that auto-plays the runtime journey;
- spotlight toggles for **Graph node materialisation** and **Return handles**;
- a **Storyboard timeline** you can click through at your own pace.
- a psychedelic **Aurora mode** that soaks the diagram in a shimmering gradient;
- glowing **Phase constellations** that group subsystems so you can trace each stage at a glance.

## Tips for exploration

- Toggle the "Return handles" and "Graph node" focus modes to jump between
  execution phases without losing context.
- Flip on **Aurora mode** whenever you want the psychedelic glow—the toggle lives
  next to the spotlight buttons and remembers its state until you tap again.
- Use the **Phase constellation** cards in the sidebar to highlight major runtime
  stages, or simply watch them light up as the story tour advances.
- Hover each node to reveal a concise description of the underlying runtime
  component and the data that flows through it.
- Start the Story Tour to see SpiralTorch narrate the handoff automatically, or
  use the "Next beat" / "Previous beat" controls to savour each move.
- Use the "Reset" button whenever you want to clear highlights and begin again
  from a blank slate.

## Extending the playground

Want to document another aspect of the runtime? Drop a new HTML file in this
folder and add it to the list above. The `return_handles_graph_node.html`
prototype demonstrates a lightweight pattern:

1. Define the nodes and edges in a small JSON payload.
2. Use the provided helper functions to animate state changes on click.
3. Keep descriptions short and action-oriented so the tooltip stays readable.

If you publish an additional explorer, please update this README so contributors
know what to expect.
