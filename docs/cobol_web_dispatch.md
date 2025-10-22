# COBOL web dispatch quickstart

This note documents the lightweight envelope builder that ships with the
`spiraltorch-wasm` bindings.  It is meant to help browser runtimes stitch
SpiralTorch narrator parameters together and forward them into mainframe entry
points without asking the UI to understand COBOL data layouts directly.

The stack is intentionally small:

- A pure-Rust `CobolEnvelopeBuilder` that assembles the job metadata, narrator
  parameters, route selectors, and annotations.  The builder lives in
  `bindings/st-wasm/src/cobol.rs` so it can be unit-tested on native targets.
- A `CobolDispatchPlanner` wrapper exported through `wasm-bindgen` that exposes
  ergonomic methods to JavaScript and TypeScript callers.  This thin layer only
  performs conversions between JS typed arrays and the Rust builder.

Because the builder is regular Rust, teams can audit or extend the envelope
shape without touching WebAssembly.  The generated JSON is simple enough to be
consumed by batch upload jobs, MQ bridges, or CICS transactions.

## Envelope layout

An envelope contains five sections:

1. **Job shell** — `job_id`, `release_channel`, and an ISO-8601 `created_at`
   timestamp.
2. **Initiators** — a list describing the humans, models, or automation agents
   that collaborated on the narration request.
3. **Route** — optional MQ, CICS, and dataset selectors that inform the
   receiving COBOL code where to dispatch the payload.
4. **Narrator payload** — curvature, temperature, encoder identifier, locale,
   and coefficient buffer.
5. **Metadata** — tags, annotations, and an open `extra` field that accepts
   arbitrary JSON objects.

The builder applies basic sanitisation (trimmed strings, empty values dropped)
so browsers do not have to maintain that logic themselves.  It also tags each
fresh envelope with a `planner_initialized` annotation to make auditing easier.

## Using the builder from Rust

```rust
use spiraltorch_wasm::{
    make_initiator, CobolEnvelope, CobolEnvelopeBuilder, InitiatorKind,
};

let mut builder = CobolEnvelopeBuilder::new("demo-job");
builder.set_release_channel("shadow");
builder.set_narrator_config(0.6, 0.4, "spiraltorch.experimental", None);
builder.set_coefficients(vec![0.12, 0.34, 0.56]);
builder.add_initiator(make_initiator(
    InitiatorKind::Human,
    "Operator",
    Some("night-shift".into()),
    None,
    Some("ops@example".into()),
    Some("validated dataset selection".into()),
));
builder.set_mq_route("MQ1", "NARRATION.INBOUND", Some("sync".into()));
builder.add_tag("browser-ui");
let envelope = builder.snapshot();
let json = envelope.to_json_string()?;

// Reset the transport plan when retrying a job with different targets.
builder.clear_route();
builder.set_cics_route("CX12", Some("NARRDISP".into()), None);
```

The snippet above exercises the same code that the WebAssembly planner uses.
You can serialise the envelope with `to_json_string()` or `to_json_bytes()` and
ship it directly to batch tooling.

Existing envelopes can be loaded back into the builder for auditing or
rerouting:

```rust
let mut builder = CobolEnvelopeBuilder::from_envelope(
    CobolEnvelope::from_json_str(saved_json)?
);
builder.clear_route();
builder.set_mq_route("MQ2", "NARRATION.REROUTED", None);
```

`from_envelope` sanitises string fields and re-applies the
`planner_initialized` annotation so downstream tooling can distinguish between
fresh and imported envelopes.

## Calling the planner from JavaScript

```ts
import init, { CobolDispatchPlanner } from "spiraltorch-wasm";

await init();
const planner = new CobolDispatchPlanner("demo-job", "staging");
planner.setNarratorConfig(0.8, 0.35, "spiraltorch.default", null);
planner.setCoefficients(new Float32Array([0.25, 0.33, 0.48]));
planner.addHumanInitiator("Analyst", null, "analyst@example", "pilot run");
planner.setMqRoute("QM1", "SPIRALTORCH.INBOUND", "commit");
const jsonEnvelope = planner.toJson();
const bytes = planner.toUint8Array();

// Later in the session we can re-target the job without rebuilding the planner.
planner.clearRoute();
planner.clearInitiators();
planner.setCreatedAt(new Date().toISOString());
planner.setCicsRoute("CX45", "NARRDISP", null);

// Load a saved envelope and continue editing in place.
planner.loadJson(savedJson);
planner.clearInitiators();
planner.addAutomationInitiator("retry-bot", null, "retry job");

// Or spawn a planner pre-populated from an existing envelope.
const imported = CobolDispatchPlanner.fromJson(savedJson);
imported.setReleaseChannel("staging");
```

`toJson()` returns a pretty-printed string suited for debugging UIs while
`toUint8Array()` yields raw bytes ready for HTTP bridges or MQ payloads.

If the caller provides `null` or `undefined` metadata, the planner keeps the
existing structured metadata.  Passing an object merges keys into the `extra`
map, while any other JSON value replaces the `extra` field entirely.

The planner also exposes ergonomic state management methods.  `clearRoute()`
removes MQ, CICS, and dataset selectors in one call, while `clearInitiators()`
empties the collaboration roster.  When a workflow needs a fresh timestamp the
UI can call `setCreatedAt()` with an explicit value or `resetCreatedAt()` to
request a new server-side default.  These helpers keep the WebAssembly layer in
sync with the Rust builder’s semantics.

## Dispatching to mainframe bridges

The WebAssembly planner does not prescribe the bridge mechanism.  Most teams
wire it into a small HTTP service that performs three tasks:

1. Validate the release channel and enforce allow-lists.
2. Record the envelope JSON for audit purposes.
3. Forward the payload into MQ queues, CICS programs, or dataset writers.

A simple bridge request looks like this:

```http
POST /dispatch HTTP/1.1
Content-Type: application/json
X-SpiralTorch-Release: shadow

{ ...envelope json... }
```

MQ bridges typically forward the `Uint8Array` bytes as-is, while CICS bridges
may unpack the JSON to populate COMMAREAs or channels before invoking the COBOL
program.

## Testing philosophy

`CobolEnvelopeBuilder` ships with unit tests that run on native targets.  They
exercise the sanitisation logic and the metadata merge semantics so changes can
be validated with a plain `cargo check -p spiraltorch-wasm`.  This reduces the
need for ad-hoc manual QA when extending the planner.
