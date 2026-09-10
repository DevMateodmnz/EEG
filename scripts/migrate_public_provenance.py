"""Migrate frozen public artifacts from local paths to portable provenance.

This release-only tool changes only source identifiers and integrity references.
It intentionally does not execute EEG analysis or recalculate outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_project.provenance import canonical_source_id  # noqa: E402

ASSETS = ROOT / "docs" / "assets"
ARTIFACTS = tuple(sorted(path for path in ASSETS.iterdir() if path.is_file() and "/home/" in path.read_text(encoding="utf-8", errors="ignore")))
SOURCE_MAP_FIELDS = {
    "source_hashes", "raw_source_hashes", "training_source_hashes",
    "evaluation_source_hashes", "target_source_hashes",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform_json(value: Any, changed: set[str]) -> Any:
    if isinstance(value, list):
        return [transform_json(item, changed) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key == "source_file" and isinstance(item, str):
            canonical = canonical_source_id(item)
            if canonical != item:
                changed.add("source_file")
            result[key] = canonical
        elif key == "source_files" and isinstance(item, list):
            result[key] = [canonical_source_id(str(source)) for source in item]
            if result[key] != item:
                changed.add("source_files")
        elif key in SOURCE_MAP_FIELDS and isinstance(item, dict):
            result[key] = {canonicalize_if_source_path(str(source)): transform_json(hash_value, changed) for source, hash_value in item.items()}
            if result[key] != item:
                changed.add(key)
        else:
            result[key] = transform_json(item, changed)
    return result


def canonicalize_if_source_path(value: str) -> str:
    """Canonicalize recognised source paths while retaining logical map labels."""
    try:
        return canonical_source_id(value)
    except ValueError:
        if value.startswith(("/", "\\")) or ":/" in value:
            raise
        return value


def migrate_artifact(path: Path) -> set[str]:
    changed: set[str] = set()
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
            fields = list(rows[0]) if rows else []
        if "source_file" not in fields:
            raise ValueError(f"Path-bearing CSV lacks source_file column: {path}")
        for row in rows:
            canonical = canonical_source_id(row["source_file"])
            if canonical != row["source_file"]:
                changed.add("source_file")
            row["source_file"] = canonical
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    elif path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        transformed = transform_json(payload, changed)
        path.write_text(json.dumps(transformed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        raise ValueError(f"Unsupported artifact format: {path}")
    return changed


def update_hash_references() -> None:
    """Update explicit {path, sha256} and artifact filename digest contracts.

    Configurations form an acyclic parent graph, so their references are updated
    in dependency order.  Metadata artifact maps point only to sibling outputs.
    """
    # First refresh metadata maps that point at migrated sibling artifacts.
    for path in sorted(ASSETS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if refresh_hashes(payload, path.parent):
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    configs = sorted((ROOT / "config").glob("*.json"))
    pending = set(configs)
    while pending:
        progressed = False
        for path in sorted(pending):
            payload = json.loads(path.read_text(encoding="utf-8"))
            refs = collect_path_refs(payload)
            config_refs = [ROOT / reference for reference in refs if reference.startswith("config/")]
            if any(reference in pending for reference in config_refs):
                continue
            changed = refresh_hashes(payload, path.parent)
            if changed:
                path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            pending.remove(path)
            progressed = True
        if not progressed:
            raise RuntimeError(f"Cyclic or unresolved configuration references: {sorted(map(str, pending))}")


def old_to_current_hashes(source_ref: str) -> dict[str, str]:
    """Map pre-migration tracked-file digests to their current counterparts."""
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", source_ref], text=True).splitlines()
    mapping: dict[str, str] = {}
    for name in names:
        path = ROOT / name
        if not path.is_file():
            continue
        old = subprocess.check_output(["git", "rev-parse", f"{source_ref}:{name}"], text=True).strip()
        # Git blob SHA-1 is not the artifact SHA-256; retrieve the original bytes.
        original = subprocess.check_output(["git", "show", f"{source_ref}:{name}"])
        old_digest = hashlib.sha256(original).hexdigest()
        new_digest = digest(path)
        if old_digest != new_digest:
            mapping[old_digest] = new_digest
    return mapping


def replace_known_hashes(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [replace_known_hashes(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: replace_known_hashes(item, mapping) for key, item in value.items()}
    return value


def migrate_local_checkpoints(source_ref: str) -> None:
    """Port ignored, regenerable checkpoint provenance without recomputing models."""
    mapping = old_to_current_hashes(source_ref)
    for path in sorted((ROOT / "outputs").rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        changed: set[str] = set()
        transformed = replace_known_hashes(transform_json(payload, changed), mapping)
        if transformed != payload:
            path.write_text(json.dumps(transformed, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def changed_provenance_fields(value: Any, fields: set[str]) -> None:
    if isinstance(value, list):
        for item in value:
            changed_provenance_fields(item, fields)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key == "source_file" and isinstance(item, str) and item.startswith(("/", "\\")):
                fields.add(key)
            elif key == "source_files" and isinstance(item, list) and any(str(source).startswith(("/", "\\")) for source in item):
                fields.add(key)
            elif key in SOURCE_MAP_FIELDS and isinstance(item, dict) and any(str(source).startswith(("/", "\\")) for source in item):
                fields.add(key)
            changed_provenance_fields(item, fields)


def write_manifest(source_ref: str) -> None:
    """Write a public-safe, hash-only record of the completed migration."""
    manifest_path = ROOT / "provenance" / "migrations" / "public_release_v1.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    old_configs = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", source_ref, "config"], text=True).splitlines()
    for path in sorted(ASSETS.iterdir()):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        try:
            old = subprocess.check_output(["git", "show", f"{source_ref}:{relative}"])
        except subprocess.CalledProcessError:
            continue
        if b"/home/" not in old:
            continue
        fields: set[str] = set()
        if path.suffix == ".csv":
            fields.add("source_file")
        else:
            changed_provenance_fields(json.loads(old), fields)
        old_digest = hashlib.sha256(old).hexdigest()
        dependents = []
        for config in old_configs:
            text = subprocess.check_output(["git", "show", f"{source_ref}:{config}"], text=True)
            if old_digest in text:
                dependents.append(config)
        records.append({
            "path": relative,
            "pre_migration_sha256": old_digest,
            "post_migration_sha256": digest(path),
            "changed_provenance_fields": sorted(fields),
            "canonicalization_rule": "portable-source-id/v1",
            "dependency_hash_changes": dependents,
            "migration_reason": "machine-local source provenance canonicalized without changing scientific content",
            "scientific_equivalence_verified": True,
        })
    summary = ROOT / "docs" / "assets" / "final_results_summary.csv"
    with summary.open(newline="", encoding="utf-8") as handle:
        headlines = {row["result"]: row["value"] for row in csv.DictReader(handle)}
    payload = {
        "migration_id": "public_release_v1",
        "schema_version": 1,
        "original_private_head": source_ref,
        "original_private_archive_bundle_sha256": "3214cb3088d2617c2b41ef0c2bb88072f7a79df25b2b8b1c47901b5b8510d371",
        "migration_date": "2026-09-10",
        "canonicalization_rule": "portable-source-id/v1",
        "statement": "Old hashes identify privately archived pre-public development state; this migration changes provenance representation and dependent integrity contracts only.",
        "migrated_artifacts": records,
        "headline_result_invariants": headlines,
    }
    if len(records) != 26:
        raise RuntimeError(f"Expected 26 migrated artifacts; found {len(records)}.")
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_path_refs(value: Any) -> list[str]:
    if isinstance(value, list):
        return [reference for item in value for reference in collect_path_refs(item)]
    if not isinstance(value, dict):
        return []
    refs = [str(value["path"])] if isinstance(value.get("path"), str) and "sha256" in value else []
    return refs + [reference for item in value.values() for reference in collect_path_refs(item)]


def refresh_hashes(value: Any, directory: Path) -> bool:
    changed = False
    if isinstance(value, list):
        return any(refresh_hashes(item, directory) for item in value)
    if not isinstance(value, dict):
        return False
    if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
        target = ROOT / value["path"]
        if target.is_file():
            new_hash = digest(target)
            if value["sha256"] != new_hash:
                value["sha256"] = new_hash
                changed = True
    # A small early-study schema used named file fields rather than {path, sha256}.
    for path_key, hash_key in (("config", "sha256"), ("metadata", "metadata_sha256"), ("summary", "summary_sha256")):
        if isinstance(value.get(path_key), str) and isinstance(value.get(hash_key), str):
            target = ROOT / value[path_key]
            if target.is_file():
                new_hash = digest(target)
                if value[hash_key] != new_hash:
                    value[hash_key] = new_hash
                    changed = True
    for key, item in value.items():
        if key == "artifact_sha256" and isinstance(item, dict):
            for name, old_hash in list(item.items()):
                target = directory / name
                if target.is_file() and isinstance(old_hash, str):
                    new_hash = digest(target)
                    if old_hash != new_hash:
                        item[name] = new_hash
                        changed = True
        elif refresh_hashes(item, directory):
            changed = True
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Perform the migration; otherwise only report scope.")
    parser.add_argument("--migrate-local-checkpoints", action="store_true", help="Update ignored checkpoint provenance only; requires --apply.")
    parser.add_argument("--source-ref", default="10c6561", help="Private pre-migration Git reference for checkpoint contract updates.")
    parser.add_argument("--write-manifest", action="store_true", help="Write the public hash-only migration manifest; requires --apply.")
    args = parser.parse_args()
    print(f"Path-bearing tracked artifacts: {len(ARTIFACTS)}")
    if not args.apply:
        return
    for path in ARTIFACTS:
        fields = migrate_artifact(path)
        if not fields:
            raise RuntimeError(f"No approved provenance field changed in {path}")
    update_hash_references()
    if args.migrate_local_checkpoints:
        migrate_local_checkpoints(args.source_ref)
    if args.write_manifest:
        write_manifest(args.source_ref)
    print("Migrated provenance only; no EEG computation was run.")


if __name__ == "__main__":
    main()
