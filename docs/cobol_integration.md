# COBOL Interop Guide for Z-space Language Processing

This guide sketches how to couple legacy COBOL estates with SpiralTorch's
natural-language stack so existing transaction processors can narrate and
interpret Z-space telemetry without rewrites.  The pattern relies on
exporting a narrow C ABI from Rust, letting GnuCOBOL (or Enterprise COBOL
with LE/370) call into the runtime just like any other reentrant
subprogram.

## 1. Why COBOL participates

Many financial and logistics platforms already mine textual summaries to
triage incidents, but the surrounding mainframe code base cannot run
Rust or Python directly.  The `st-text` crate in this repository powers
resonance narrators such as [`TextResonator`] and `LanguageWaveEncoder`,
which already deliver smooth natural-language descriptions for the Z-space
signals SpiralTorch emits.  By wrapping those types behind a C ABI we can
expose the same capability to COBOL call sites without forcing a rewrite of
existing CICS or batch flows.【F:crates/st-text/src/lib.rs†L24-L91】

## 2. Architecture overview

The recommended bridge is intentionally small:

1. **Rust host shared library.** Build a `cdylib` crate that depends on
   `st-text`, owns the model initialisation, and surfaces flat C functions
   (`extern "C"`) for the operations COBOL needs (for example, initialising
   a narrator, requesting a textual summary, or synthesising an amplitude
   envelope).
2. **Thin C shim.** Provide a header and optional C helper that translate
   COBOL-friendly data (fixed-size arrays, pointers plus lengths) into the
   idiomatic Rust structures expected by the shared library.  This also
   hosts allocation helpers so COBOL can release buffers deterministically.
3. **COBOL procedure division.** Call the shim via `CALL "spiraltorch_text"`
   using `BY REFERENCE` parameters.  COBOL then distributes the returned
   sentences through MQ, VSAM, or whichever downstream interface already
   drives operator dashboards.

The sections below supply reference snippets for each stage so the bridge
remains reproducible across environments.

## 3. Exporting a Rust narrator behind a C ABI

Create `bindings/cobol_bridge/Cargo.toml` with `crate-type = ["cdylib"]`
and add `st-text` plus `st-tensor` as dependencies.  A minimal
`src/lib.rs` can expose two entry points: one to build a narrator and one
to turn raw resonance coefficients into prose.

```rust
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use st_tensor::DifferentialResonance;
use st_text::{LanguageWaveEncoder, TextResonator};

#[repr(C)]
pub struct ResonanceHandle {
    encoder: LanguageWaveEncoder,
}

#[no_mangle]
pub extern "C" fn st_cobol_new_resonator(
    curvature: c_float,
    temperature: c_float,
) -> *mut ResonanceHandle {
    let encoder = match LanguageWaveEncoder::new(curvature, temperature) {
        Ok(enc) => enc,
        Err(_) => return std::ptr::null_mut(),
    };
    let handle = ResonanceHandle { encoder };
    Box::into_raw(Box::new(handle))
}

#[no_mangle]
pub extern "C" fn st_cobol_describe(
    handle: *mut ResonanceHandle,
    values: *const c_float,
    len: c_int,
    out_summary: *mut *mut c_char,
) -> c_int {
    if handle.is_null() || values.is_null() || out_summary.is_null() {
        return -1;
    }
    let handle = unsafe { &mut *handle };
    let slice = unsafe { std::slice::from_raw_parts(values, len as usize) };
    let resonance = DifferentialResonance::from_coefficients(slice);
    let resonator = TextResonator::with_encoder(handle.encoder.clone());
    let narrative = resonator.describe_resonance(&resonance);
    let summary = CString::new(narrative.summary).unwrap();
    unsafe { *out_summary = summary.into_raw() };
    0
}
```

Production builds normally add allocation helpers (`st_cobol_free_string`)
and expose richer metrics, but the example highlights the ABI shape COBOL can
consume.

Build the shared library with:

```bash
cargo build -p cobol_bridge --release
```

The resulting `libcobol_bridge.so` (or `.dll`/`.dylib`) becomes the module
COBOL links against.

## 4. Translating outputs into COBOL

A companion header keeps COBOL calls stable:

```c
#ifndef SPIRALTORCH_COBOL_BRIDGE_H
#define SPIRALTORCH_COBOL_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void *st_cobol_new_resonator(float curvature, float temperature);
int32_t st_cobol_describe(void *handle,
                          const float *values,
                          int32_t len,
                          char **out_summary);
void st_cobol_free_string(char *ptr);

#ifdef __cplusplus
}
#endif

#endif /* SPIRALTORCH_COBOL_BRIDGE_H */
```

With the header installed, the COBOL side can look like:

```cobol
 identification division.
 program-id. nlp-bridge.
 environment division.
 configuration section.
 repository.
     function all intrinsic.
 data division.
 working-storage section.
 01  curvature          pic s9v9(3) comp-2 value 0.42.
 01  temperature        pic s9v9(3) comp-2 value 0.65.
 01  summary-pointer    usage pointer.
 01  status-code        pic s9(4) comp value zero.
 01  coeff-count        pic s9(4) comp value 128.
 01  coeff-table        pic s9v9(4) comp-2 occurs 128 value 0.
 procedure division.
     call "st_cobol_new_resonator" using by value curvature temperature
                                    returning summary-pointer.
     if summary-pointer = null
         display "initialisation failed" stop run
     end-if.
     call "st_cobol_describe" using summary-pointer
                                     by reference coeff-table
                                     by value coeff-count
                                     by reference summary-pointer
                             returning status-code.
     if status-code not = 0
         display "description failed" stop run
     end-if.
     call "st_cobol_free_string" using by value summary-pointer.
     stop run.
```

GnuCOBOL users can compile the snippet with

```bash
cobc -x nlp-bridge.cbl cobol_bridge.o -L target/release -lcobol_bridge
```

on z/OS the link step instead uses the LE binder against the PDSE that
hosts the shared library.

## 5. Operational guidance

- **Threading.** The exported ABI should mark every function as reentrant;
  COBOL tasks often reuse the same module across dozens of TCBs.  Using a
  per-handle `LanguageWaveEncoder` (as shown above) keeps state confined.
- **Buffer lifecycle.** Always pair any function returning an allocated
  pointer with an explicit free call so COBOL can release memory.  Enterprise
  COBOL does not automatically hand buffers back to the OS.
- **Batch throughput.** When smoothing natural-language responses for
  thousands of telemetry frames, drive the ABI in bulk by passing a slice of
  coefficients representing multiple frames, or create a light-weight worker
  pool on the Rust side that streams sentences over MQ to keep CICS regions
  responsive.
- **Testing.** Unit-test the shared library using `#[cfg(test)]` Rust tests,
  then add GnuCOBOL regression programs that check deterministic phrasing for
  known coefficient patterns before promoting the load module.

Following this pattern lets COBOL pipelines tap into SpiralTorch's language
stack while keeping existing operational envelopes intact.

## 6. Orchestrating narrations from the browser (WASM → mainframe)

To let humans and LM agents co-author dispatches without leaving the browser, the
`spiraltorch-wasm` crate now exposes a `CobolDispatchPlanner`. The helper collects narrator
parameters, provenance, and routing hints entirely in WebAssembly before serialising the
payload for COBOL.【F:bindings/st-wasm/src/cobol_bridge.rs†L1-L266】 Both people and models can
identify themselves through `addHumanInitiator`, `addModelInitiator`, or
`addAutomationInitiator`, keeping an auditable chain of custody inside the envelope.

```ts
import init, { CobolDispatchPlanner } from "spiraltorch-wasm";

await init();

const planner = new CobolDispatchPlanner("ops-lab-42", "shadow");
planner.setNarratorConfig(0.48, 0.67, "st-language.wave", "ja-JP");
planner.setCoefficients(new Float32Array(resonanceBuffer));
planner.addHumanInitiator("Operator Kana", null, null, "triage");
planner.addModelInitiator("lm://spiraltorch/language", "2025-05", "lm-agent", "autoprompt");
planner.setMqRoute("QM1", "ST.NARRATION.IN", "syncpoint");
planner.setCicsRoute("STXR", "STLANG", "NARRATE");
planner.setDataset("ST.DATA.NARRATION(+1)");

const bytes = planner.toUint8Array();      // JSON buffer ready for MQ/HTTP bridges
const leSlice = planner.coefficientsAsBytes(); // F32 little-endian view for pointer passing
```

On the mainframe side the JSON payload drops straight into the shared library created in the
previous sections: either push `bytes` onto MQ, invoke a CICS program that unwraps the JSON,
or write the blob into a dataset for batch jobs to consume. The COBOL preview exported by
`toCobolPreview()` mirrors the pointer layout used by `st_cobol_describe`, making it simple to
verify buffer sizing before the dispatch leaves the browser.【F:bindings/st-wasm/src/cobol_bridge.rs†L197-L266】

## 7. Web console for mixed human/model dispatch

The repository now ships a Vite-based sample console under
`bindings/st-wasm/examples/cobol-console/`. It surfaces the planner in a responsive UI so
operators, copilots, or autonomous agents can tweak curvature/temperature, seed coefficient
buffers, and route the payload into MQ or CICS straight from the browser.【F:bindings/st-wasm/examples/cobol-console/index.html†L1-L126】【F:bindings/st-wasm/examples/cobol-console/src/main.ts†L1-L180】

Run `npm run dev` inside the example after building the WASM bindings and point it at your
HTTP bridge. The UI renders the JSON envelope, a COBOL pointer summary, and the exact byte
count before letting you POST the payload downstream.【F:bindings/st-wasm/examples/cobol-console/README.md†L1-L62】 Humans can adjust
fields manually while model agents reuse the same surface via browser automations or direct
imports of `CobolDispatchPlanner`, providing a shared staging ground before the message enters
z/OS.
