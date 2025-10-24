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
   receiving COBOL code where to dispatch the payload. Dataset selectors can
   now carry richer metadata (PDS member, disposition, volume serial, DCB
   attributes, SMS class hints, SPACE allocations, DSNTYPE, LIKE templates,
   UNIT/AVGREC hints, retention periods, RLSE, and expiration dates) so batch
   writers or GDG loaders can stage the payload precisely without extra glue
   code in the bridge layer.
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
builder.set_dataset(Some("HLQ.DATA".into()));
builder.set_dataset_member(Some("NARRATE".into()));
builder.set_dataset_disposition(Some("SHR".into()));
builder.set_dataset_volume(Some("VOL001".into()));
builder.set_dataset_record_format(Some("FB".into()));
builder.set_dataset_record_length(Some(512));
builder.set_dataset_block_size(Some(6144));
builder.set_dataset_data_class(Some("NARRATE".into()));
builder.set_dataset_management_class(Some("GDG".into()));
builder.set_dataset_storage_class(Some("FASTIO".into()));
builder.set_dataset_space_primary(Some(15));
builder.set_dataset_space_secondary(Some(5));
builder.set_dataset_space_unit(Some("CYL".into()));
builder.set_dataset_directory_blocks(Some(30));
builder.set_dataset_type(Some("LIBRARY".into()));
builder.set_dataset_like(Some("ST.DATA.TEMPLATE".into()));
builder.set_dataset_organization(Some("PO".into()));
builder.set_dataset_key_length(Some(64));
builder.set_dataset_key_offset(Some(8));
builder.set_dataset_control_interval_size(Some(4096));
builder.set_dataset_share_options_cross_region(Some(3));
builder.set_dataset_share_options_cross_system(Some(3));
builder.set_dataset_reuse(Some(true));
builder.set_dataset_log(Some(true));
builder.set_dataset_unit(Some("SYSDA".into()));
builder.set_dataset_unit_count(Some(3));
builder.set_dataset_average_record_unit(Some("K".into()));
builder.set_dataset_catalog_behavior(Some("CATALOG".into()));
builder.set_dataset_retention_period(Some(45));
builder.set_dataset_release_space(Some(true));
builder.set_dataset_erase_on_delete(Some(true));
builder.set_dataset_expiration_date(Some("2025123".into()));
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
planner.setDataset("HLQ.DATA(+1)");
planner.setDatasetMember("NARRATE");
planner.setDatasetDisposition("SHR");
planner.setDatasetVolume("VOL001");
planner.setDatasetRecordFormat("FB");
planner.setDatasetRecordLength(512);
planner.setDatasetBlockSize(6144);
planner.setDatasetDataClass("NARRATE");
planner.setDatasetManagementClass("GDG");
planner.setDatasetStorageClass("FASTIO");
planner.setDatasetSpacePrimary(15);
planner.setDatasetSpaceSecondary(5);
planner.setDatasetSpaceUnit("CYL");
planner.setDatasetDirectoryBlocks(30);
planner.setDatasetType("LIBRARY");
planner.setDatasetLike("ST.DATA.TEMPLATE");
planner.setDatasetOrganization("PO");
planner.setDatasetKeyLength(64);
planner.setDatasetKeyOffset(8);
planner.setDatasetControlIntervalSize(4096);
planner.setDatasetShareOptionsCrossRegion(3);
planner.setDatasetShareOptionsCrossSystem(3);
planner.setDatasetReuse(true);
planner.setDatasetLog(false);
planner.setDatasetUnit("SYSDA");
planner.setDatasetUnitCount(3);
planner.setDatasetAverageRecordUnit("K");
planner.setDatasetCatalogBehavior("CATALOG");
planner.setDatasetRetentionPeriod(45);
planner.setDatasetReleaseSpace(true);
planner.setDatasetEraseOnDelete(true);
planner.setDatasetExpirationDate("2025123");
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

Passing `null`/`undefined` into `setDataset` or any of the dataset metadata
setters clears the previous value, which is useful when a UI allows operators to
remove fields without rebuilding the planner instance.

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

When it comes time to consume the metadata on z/OS, refer to
`examples/cobol/st_dataset_writer.cbl` for a concrete BPXWDYN allocation
example driven entirely by the planner's dataset hints.

## Validating envelopes

Before dispatching, both the Rust builder and the WebAssembly planner can audit
envelopes for common mistakes.  Call `builder.validation_issues()` to receive a
list of human-readable problems or `builder.is_valid()` when only a boolean is
needed.  Browser callers can mirror the same workflow with
`planner.validationIssues()` and `planner.isValid()`.  The checks flag missing
initiators, absent routes, narrator settings outside the 0–1 range, and jobs
that still rely on the default `job` placeholder identifier.
Dataset hints must also remain internally consistent: the planner warns when a
block size is not a clean multiple of the record length, when SPACE units lack
matching allocations, when secondary extents appear without a primary, when
directory blocks are provided for non-partitioned targets, when DSNTYPE/DSORG or
AVGREC fall outside the supported sets, when VSAM key metadata exceeds the
record length, when CI sizes are not BPXWDYN-friendly, when share options exceed
VSAM limits or appear without DSORG VS, when reuse/log hints are applied to
non-VSAM datasets, when unit counts omit a UNIT, when catalog directives are
unknown, when retention days exceed SMS limits, or when expiration dates are
malformed. These checks surface problems before SMS allocation commands reach
BPXWDYN.

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
