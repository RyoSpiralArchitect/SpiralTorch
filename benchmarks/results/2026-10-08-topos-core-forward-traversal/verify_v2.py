"""Corrected offline verifier; the v1 archive and verifier remain frozen."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
V1_MANIFEST_SHA256 = "bcb636ebc879f657ca05e9a7453d4e19393df8ac8983dd1d3b6758507e673f86"
EXPECTED_PARITY = (
    "Every forward and both VJPs are bit-identical to the prepared native NN "
    "reference; both audits equal. Checked after each measured call."
)


def load_v1():
    if not __debug__:
        raise RuntimeError("The frozen v1 verifier requires assertions; do not use -O")
    manifest = (HERE / "SHA256SUMS").read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == V1_MANIFEST_SHA256
    expected = {"README.md", "measurements.json.gz", "verification.json", "verify.py"}
    seen = set()
    for line in manifest.decode().splitlines():
        digest, name = line.split("  ", 1)
        assert name in expected and name not in seen
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
        seen.add(name)
    assert seen == expected
    return runpy.run_path(str(HERE / "verify.py"))


def verify(data, witness, v1):
    v1["verify"](data, witness)
    for record in data["native_records"]:
        result = json.loads(record["raw_json"])
        if result.get("parity") != EXPECTED_PARITY:
            raise AssertionError("native parity missing or contradictory: " + record["file"])


def negative_controls(data, witness, v1):
    checked = 0
    for index in range(len(data["native_records"])):
        for missing in (False, True):
            bad, proof = copy.deepcopy(data), copy.deepcopy(witness)
            record = bad["native_records"][index]
            result = json.loads(record["raw_json"])
            if missing:
                del result["parity"]
            else:
                result["parity"] = "failed"
            raw = json.dumps(result)
            record["raw_json"] = raw
            proof["private_files"][record["file"]] = {
                "sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "bytes": len(raw.encode()),
            }
            # Rebinding proves rejection is semantic, not a stale inventory hash.
            if index == 0:
                v1["verify"](bad, proof)
            try:
                verify(bad, proof, v1)
            except AssertionError as error:
                assert str(error).startswith("native parity missing or contradictory:")
                checked += 1
            else:
                raise AssertionError("contradictory native parity accepted")
    assert checked == 72
    return checked


def main():
    v1 = load_v1()
    data = json.loads(gzip.decompress((HERE / "measurements.json.gz").read_bytes()))
    witness = json.loads((HERE / "verification.json").read_text())
    verify(data, witness, v1)
    v1["negative_controls"](data, witness)
    checked = negative_controls(data, witness, v1)
    print(f"v2 verified unchanged archive: 36 native parity assertions; {checked} "
          "rebound parity negative controls plus 4 v1 controls; no fresh execution")


if __name__ == "__main__":
    main()
