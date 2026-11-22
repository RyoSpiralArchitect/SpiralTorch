import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {checkLineMidpoint} from './native_line_midpoint.mjs';

const [path,raw,cuts]=process.argv.slice(2),url=pathToFileURL(resolve(path));
const M=await import(url);
await M.default({module_or_path:readFileSync(new URL(url.href.replace(/\.js$/,'_bg.wasm')))});
const read=p=>JSON.parse(readFileSync(p));
const fixture=read(new URL('./native_causal_lm_cache_fixture.json',import.meta.url));
const owners=read(new URL('./native_line_owners_bridge.json',import.meta.url));
const native=read(raw);
if(native.schema!=='spiraltorch.synthetic_line_midpoint.v1')throw Error('wrong native baseline');
console.log(JSON.stringify(checkLineMidpoint(M,fixture,owners,native,read(cuts))));
