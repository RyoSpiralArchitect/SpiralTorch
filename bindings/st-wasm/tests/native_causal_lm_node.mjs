import {readFileSync, writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {checkNativeCausalLm} from './native_causal_lm.mjs';
import {checkCachedSuffix} from './native_causal_lm_cache.mjs';
import {checkCachedLines} from './native_cached_lines.mjs';
import {checkLineGradients} from './native_line_gradients.mjs';
import {checkLineOwners} from './native_line_owners.mjs';
import {checkLineAnchorKl} from './native_line_anchor_kl.mjs';
import {checkLineEnvelope} from './native_line_envelope.mjs';
import {checkLineSession} from './native_line_session.mjs';

const glue = pathToFileURL(resolve(process.argv[2]));
const M = await import(glue);
await M.default({module_or_path: readFileSync(new URL(glue.href.replace(/\.js$/, '_bg.wasm')))});
const fixture = JSON.parse(readFileSync(new URL('./native_causal_lm_fixture.json', import.meta.url)));
const selected = [];
const report = checkNativeCausalLm(M, fixture, item => selected.push(item));
const models = JSON.parse(readFileSync(new URL('./native_causal_lm_cache_fixture.json', import.meta.url)));
report.cached = checkCachedSuffix(M, models);
const owners = JSON.parse(readFileSync(new URL('./native_line_owners_bridge.json',import.meta.url)));
report.line_envelope=checkLineEnvelope(M,models,owners,process.env.ST_SYNTHETIC_ENVELOPE_CUTS
  ? JSON.parse(readFileSync(process.env.ST_SYNTHETIC_ENVELOPE_CUTS)) : undefined);
report.line_anchor_kl=checkLineAnchorKl(M,models,owners,
  process.env.ST_SYNTHETIC_LINE_KL_BRIDGE ? JSON.parse(readFileSync(process.env.ST_SYNTHETIC_LINE_KL_BRIDGE)) : undefined);
const ray = JSON.parse(readFileSync(new URL('./native_line_ray_bridge.json',import.meta.url)));
report.line_ray = checkLineOwners(M, models, {...owners,...ray,expected_measurements:ray.measurements.ray},
  process.argv[6] ? value=>writeFileSync(process.argv[6],JSON.stringify(value),{flag:'wx'}) : undefined);
report.line_owners = checkLineOwners(M, models, {...owners,expected_measurements:ray.measurements.sgd},
  process.argv[5] ? value=>writeFileSync(process.argv[5],JSON.stringify(value),{flag:'wx'}) : undefined);
report.line_session=checkLineSession(M,models,readFileSync(new URL('./native_line_session_plan.json',import.meta.url),'utf8'),
  JSON.parse(readFileSync(new URL('./native_line_session_bridge.json',import.meta.url))),
  process.argv[4] ? payload=>writeFileSync(process.argv[4],JSON.stringify(payload),{flag:'wx'}) : undefined);
report.line_gradients = checkLineGradients(M, models, JSON.parse(readFileSync(new URL('./native_line_gradients_fixture.json', import.meta.url))));
report.lines = checkCachedLines(M, models, JSON.parse(readFileSync(new URL('./native_cached_lines_fixture.json', import.meta.url))));
if (process.argv[3]) writeFileSync(process.argv[3], JSON.stringify(selected), {flag: 'wx'});
console.log(JSON.stringify(report));
