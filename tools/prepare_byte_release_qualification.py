#!/usr/bin/env python3
"""Freeze exact wheels/build receipts, or initialize the one canonical ledger.

CPU/files only. This neither builds, qualifies, nor publishes a package. Build
receipts must come from the three reviewed release builds, not guessed settings.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import byte_release_artifacts as artifacts
import byte_release_contract as contract

PLATFORMS = ("linux", "macos", "windows")


def freeze(root, wheels, receipts, candidate=1, prior=None, repair_review=None):
    source = contract.source_identity(root)
    inventories, builds = {}, {}
    if set(wheels) != set(PLATFORMS) or set(receipts) != set(PLATFORMS):
        raise ValueError("three platform wheels and original build receipts are required")
    for platform in PLATFORMS:
        inventory = artifacts.wheel_inventory(wheels[platform])
        if inventory["platform"] != platform:
            raise ValueError("wheel was assigned to another platform")
        before = artifacts.sha256(receipts[platform])
        receipt = contract.read(receipts[platform])
        if artifacts.sha256(receipts[platform]) != before:
            raise ValueError("build receipt changed during preparation")
        if type(receipt) is not dict:
            raise ValueError("build receipt must be an object")
        builds[platform] = {key: receipt.get(key) for key in ("source_commit", "wheel_sha256", "toolchain", "features")}
        builds[platform]["receipt_sha256"] = before
        inventories[platform] = inventory
    if contract.source_identity(root) != source:
        raise ValueError("source changed during preparation")
    value = dict(schema=contract.SCHEMA, decision=contract.DECISION, candidate=candidate,
                 version=inventories["macos"]["version"], source=source, wheels=inventories, builds=builds,
                 tests=[list(row) for row in contract.TESTS], rounds=3, budget_seconds=5400,
                 long_request_sha256=contract.long_resume.REQUEST_SHA256, release_ready=False,
                 prior_admission_sha256=contract.digest(contract.validate_admission(prior)) if prior is not None else None,
                 repair_review=repair_review)
    return contract.validate_admission(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init-ledger", help="one time only; never choose a new path to reset attempts")
    initialize.add_argument("path", type=Path)
    prepare = commands.add_parser("freeze", help="write an immutable admission, without executing it")
    for platform in PLATFORMS:
        prepare.add_argument("--" + platform + "-wheel", type=Path, required=True)
        prepare.add_argument("--" + platform + "-build-receipt", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--candidate", type=int, choices=(1, 2), default=1)
    prepare.add_argument("--prior-admission", type=Path)
    prepare.add_argument("--repair-review")
    args = parser.parse_args(argv)
    if args.command == "init-ledger":
        contract.initialize_ledger(args.path)
    else:
        value = freeze(ROOT, {p: getattr(args, p + "_wheel") for p in PLATFORMS},
                       {p: getattr(args, p + "_build_receipt") for p in PLATFORMS}, args.candidate,
                       contract.read(args.prior_admission) if args.prior_admission is not None else None,
                       args.repair_review)
        contract.save(args.output, value)
        print(contract.digest(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
