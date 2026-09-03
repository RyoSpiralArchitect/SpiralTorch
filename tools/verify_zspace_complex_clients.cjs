// SPDX-License-Identifier: AGPL-3.0-or-later
const fs = require('node:fs');
const assert = require('node:assert/strict');
const wasm = require(process.argv[2]);
const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
const results = cases.map(({request, python_receipt}) => {
    const receipt = JSON.parse(wasm.zspaceStochasticSchrodingerComplexStepJson(JSON.stringify(request)));
    assert.deepEqual(receipt, python_receipt);
    assert.deepEqual(JSON.parse(wasm.validateZspaceStochasticSchrodingerComplexJson(JSON.stringify(python_receipt))), receipt);
    const altered = JSON.parse(JSON.stringify(receipt));
    altered.gradient.grad_input_imaginary[0] += 1;
    assert.throws(() => wasm.validateZspaceStochasticSchrodingerComplexJson(JSON.stringify(altered)));
    return receipt;
});
process.stdout.write(JSON.stringify(results));
