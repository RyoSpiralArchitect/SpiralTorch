const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const generated = path.resolve(process.argv[2]);
const handwritten = path.resolve(__dirname, '../types/spiraltorch-wasm.d.ts');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'byte-selection-types-'));
try {
  for (const [label, module, declarations] of [
    ['generated', generated, []], ['handwritten', 'spiraltorch-wasm', [handwritten]],
  ]) {
    const source = path.join(temporary, label + '.ts');
    fs.writeFileSync(source, `
import {ByteCorpusStudy, ByteCorpusStudyCheckpoint, ByteCorpusCheckpointCandidates,
        ByteCorpusCheckpointSelection, ByteCorpusModelArtifact} from ${JSON.stringify(module)};
declare const study: ByteCorpusStudy;
declare const checkpoint: ByteCorpusStudyCheckpoint;
const candidates: ByteCorpusCheckpointCandidates = study.checkpointCandidates();
candidates.appendJson(checkpoint.toJson());
const selected: ByteCorpusCheckpointSelection = candidates.selectCheckpoint('plain', 'fixed_window_frozen.v1');
const artifact: ByteCorpusModelArtifact = selected.artifact;
const report: string = selected.reportJson;
const length: number = candidates.length;
const capacity: number = candidates.capacity;
// @ts-expect-error Checkpoint handles are not opaque JSON.
candidates.appendJson(checkpoint);
// @ts-expect-error JS sequences cannot become checkpoint collections.
candidates.appendJson([]);
// @ts-expect-error Candidate collections must own a study.
new ByteCorpusCheckpointCandidates();
// @ts-expect-error Selection results are not caller-constructible.
new ByteCorpusCheckpointSelection();
artifact.free(); selected.free(); candidates.free();
`);
    execFileSync('tsc', ['--noEmit', '--strict', '--skipLibCheck', '--target', 'es2022',
      '--module', 'esnext', '--moduleResolution', 'bundler',
      '--lib', 'es2022,dom,esnext.disposable', source, ...declarations], {stdio: 'inherit'});
    console.log('Byte checkpoint selection TypeScript contract passed: ' + label);
  }
} finally {
  fs.rmSync(temporary, {recursive: true, force: true});
}
