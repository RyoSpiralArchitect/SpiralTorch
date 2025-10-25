# Interactive runtime handoff explorer

This directory collects standalone HTML explorations that visualise pieces of the
runtime diagrams from the main README. Each document can be opened directly in
a browser without a build step.

Current prototypes:

- `return_handles_graph_node.html` – highlights how the "graph node"
  materialisation and "return handles" delivery flow through the dispatcher.
  Click the buttons at the top or individual nodes to explore the sequence.

Open the HTML files from your file browser or run a local web server:

```bash
python -m http.server --directory docs/interactive
```

Then navigate to <http://localhost:8000/return_handles_graph_node.html>.
