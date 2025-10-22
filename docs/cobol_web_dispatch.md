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
    make_initiator, CobolEnvelopeBuilder, InitiatorKind,
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
```

The snippet above exercises the same code that the WebAssembly planner uses.
You can serialise the envelope with `to_json_string()` or `to_json_bytes()` and
ship it directly to batch tooling.

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
```

`toJson()` returns a pretty-printed string suited for debugging UIs while
`toUint8Array()` yields raw bytes ready for HTTP bridges or MQ payloads.

If the caller provides `null` or `undefined` metadata, the planner keeps the
existing structured metadata.  Passing an object merges keys into the `extra`
map, while any other JSON value replaces the `extra` field entirely.

### Importing existing envelopes

Browsers that fetch historical envelopes can hydrate a planner directly from
JSON without re-entering every field manually:

```ts
const previousJson = await fetch("/api/envelope/job-200").then((res) => res.text());
const planner = CobolDispatchPlanner.fromJson(previousJson);
planner.setReleaseChannel("shadow");
planner.loadObject({ ...planner.toObject(), job_id: "job-201" });
```

`CobolDispatchPlanner.fromJson` and `fromObject` run the same sanitisation as
`CobolEnvelopeBuilder::new`, ensuring empty strings are trimmed away and the
`planner_initialized` annotation is present.  Instance methods `loadJson` and
`loadObject` can reset an existing planner, which is useful when rendering the
same component for multiple envelopes during a debugging session.

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
