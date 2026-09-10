"""Verify the provenance-only public-release migration.

With ``--source-ref 10c6561`` this compares pre-public artifacts directly.  In
a public clone, omitting that option verifies the committed manifest and the
headline invariants without requiring the private archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_project.provenance import canonical_source_id  # noqa: E402


MANIFEST_PATH = ROOT / "provenance" / "migrations" / "public_release_v1.json"
SOURCE_MAP_FIELDS = {"source_hashes", "raw_source_hashes", "training_source_hashes", "evaluation_source_hashes", "target_source_hashes"}
ALLOWED_HASH_FIELDS = {
    "artifact_sha256", "final_config_sha256", "frozen_core_sha256",
    "frozen_calibration_core_sha256", "frozen_cross_subject_core_sha256",
    "configuration_sha256", "config_sha256", "reliability_config_sha256",
    "historical_prediction_sha256",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonicalize(value: Any) -> Any:
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if not isinstance(value, dict):
        return value
    output: dict[str, Any] = {}
    for key, item in value.items():
        if key == "source_file":
            output[key] = canonical_source_id(str(item))
        elif key == "source_files" and isinstance(item, list):
            output[key] = [canonical_source_id(str(source)) for source in item]
        elif key in SOURCE_MAP_FIELDS and isinstance(item, dict):
            converted: dict[str, Any] = {}
            for source, digest in item.items():
                try:
                    converted[canonical_source_id(str(source))] = canonicalize(digest)
                except ValueError:
                    converted[str(source)] = canonicalize(digest)
            output[key] = converted
        elif key in ALLOWED_HASH_FIELDS:
            output[key] = "<approved-integrity-contract-change>"
        elif key in {"sha256", "metadata_sha256"} and {"config", "metadata"}.issubset(value):
            # Explicit legacy bilateral-parent contract, not a generic hash rule.
            output[key] = "<approved-integrity-contract-change>"
        else:
            output[key] = canonicalize(item)
    return output


def compare_csv(old: bytes, current: Path) -> None:
    old_rows = list(csv.DictReader(old.decode("utf-8").splitlines()))
    with current.open(newline="", encoding="utf-8") as handle:
        new_rows = list(csv.DictReader(handle))
    if (list(old_rows[0]) if old_rows else []) != (list(new_rows[0]) if new_rows else []):
        raise AssertionError(f"CSV columns changed: {current}")
    if len(old_rows) != len(new_rows):
        raise AssertionError(f"CSV row count changed: {current}")
    for index, (before, after) in enumerate(zip(old_rows, new_rows, strict=True)):
        expected = dict(before)
        expected["source_file"] = canonical_source_id(expected["source_file"])
        if expected != after:
            raise AssertionError(f"CSV scientific cell changed at row {index}: {current}")


def compare_json(old: bytes, current: Path) -> None:
    before = canonicalize(json.loads(old))
    after = canonicalize(json.loads(current.read_text(encoding="utf-8")))
    if before != after:
        raise AssertionError(f"Unapproved JSON difference: {current}")


def verify_manifest(manifest: dict[str, Any]) -> int:
    migrated = manifest["migrated_artifacts"]
    if len(migrated) != 26:
        raise AssertionError("Expected exactly 26 migrated tracked artifacts.")
    for record in migrated:
        path = ROOT / record["path"]
        if sha256_bytes(path.read_bytes()) != record["post_migration_sha256"]:
            raise AssertionError(f"Manifest hash drift: {path}")
        if not record["scientific_equivalence_verified"]:
            raise AssertionError(f"Equivalence not recorded: {path}")
    summary = ROOT / "docs" / "assets" / "final_results_summary.csv"
    with summary.open(encoding="utf-8") as handle:
        rows = {row["result"]: row["value"] for row in csv.DictReader(handle)}
    if rows != manifest["headline_result_invariants"]:
        raise AssertionError("Headline frozen results differ from the public migration manifest.")
    return len(migrated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ref", help="Private pre-public commit for direct equivalence comparison.")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    count = verify_manifest(manifest)
    if args.source_ref:
        for record in manifest["migrated_artifacts"]:
            path = ROOT / record["path"]
            old = subprocess.check_output(["git", "show", f"{args.source_ref}:{record['path']}"])
            if sha256_bytes(old) != record["pre_migration_sha256"]:
                raise AssertionError(f"Private baseline hash mismatch: {path}")
            if path.suffix == ".csv":
                compare_csv(old, path)
            else:
                compare_json(old, path)
        print(f"Direct scientific equivalence verified for {count} migrated artifacts.")
    else:
        print(f"Public manifest and headline invariants verified for {count} migrated artifacts.")


if __name__ == "__main__":
    main()
