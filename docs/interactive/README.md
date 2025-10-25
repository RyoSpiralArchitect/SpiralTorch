# Interactive runtime handoff explorer

This directory collects standalone HTML explorations that visualise pieces of the
runtime diagrams from the main README. Each document can be opened directly in
a browser without a build step.

Current prototypes:

- `return_handles_graph_node.html` – highlights how the "graph node"
  materialisation and "return handles" delivery flow through the dispatcher.
  Click the buttons at the top or individual nodes to explore the sequence.

## Getting started

Open the HTML files from your file browser or run a local web server:

```bash
python -m http.server --directory docs/interactive
```

Then navigate to <http://localhost:8000/return_handles_graph_node.html>.

## Tips for exploration

- Toggle the "Return handles" and "Graph node" focus modes to jump between
  execution phases without losing context.
- Hover each node to reveal a concise description of the underlying runtime
  component and the data that flows through it.
- Use the "Replay full sequence" control to watch the diagram animate end to
  end whenever you want a refresher of the complete handoff choreography.

## Extending the playground

Want to document another aspect of the runtime? Drop a new HTML file in this
folder and add it to the list above. The `return_handles_graph_node.html`
prototype demonstrates a lightweight pattern:

1. Define the nodes and edges in a small JSON payload.
2. Use the provided helper functions to animate state changes on click.
3. Keep descriptions short and action-oriented so the tooltip stays readable.

If you publish an additional explorer, please update this README so contributors
know what to expect.
