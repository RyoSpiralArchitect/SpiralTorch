import {readFileSync, writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {checkNativeCausalLm} from './native_causal_lm.mjs';
import {checkCachedSuffix} from './native_causal_lm_cache.mjs';

const glue = pathToFileURL(resolve(process.argv[2]));
const M = await import(glue);
await M.default({module_or_path: readFileSync(new URL(glue.href.replace(/\.js$/, '_bg.wasm')))});
const fixture = JSON.parse(readFileSync(new URL('./native_causal_lm_fixture.json', import.meta.url)));
const selected = [];
const report = checkNativeCausalLm(M, fixture, item => selected.push(item));
report.cached = checkCachedSuffix(M, JSON.parse(readFileSync(new URL('./native_causal_lm_cache_fixture.json', import.meta.url))));
if (process.argv[3]) writeFileSync(process.argv[3], JSON.stringify(selected), {flag: 'wx'});
console.log(JSON.stringify(report));
