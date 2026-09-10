"""Public-clone checks for the provenance-only release migration."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import unittest

from eeg_project.provenance import canonical_source_id, resolve_source_id
from eeg_project.preprocessing import DEFAULT_DATA_DIRECTORY
from scripts.verify_public_provenance_migration import verify_manifest


ROOT = Path(__file__).resolve().parents[1]


class PublicProvenanceMigrationTests(unittest.TestCase):
    def test_manifest_and_headline_invariants_are_publicly_verifiable(self) -> None:
        manifest = json.loads((ROOT / "provenance" / "migrations" / "public_release_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(verify_manifest(manifest), 26)

    def test_canonical_physionet_identifier_round_trip(self) -> None:
        identifier = canonical_source_id(str(DEFAULT_DATA_DIRECTORY / "MNE-eegbci-data/files/eegmmidb/1.0.0/S001/S001R06.edf"))
        self.assertEqual(identifier, "physionet/eegmmidb/1.0.0/S001/S001R06.edf")
        self.assertEqual(resolve_source_id(identifier, DEFAULT_DATA_DIRECTORY).name, "S001R06.edf")

    def test_tracked_public_tree_has_no_machine_local_paths_or_raw_data(self) -> None:
        files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
        forbidden = re.compile(r"/(?:" + "home" + r"|Users)/[A-Za-z0-9_.-]+/|[A-Za-z]:\\Users\\")
        for name in files:
            if name == "AGENTS.md":
                continue
            self.assertIsNone(forbidden.search((ROOT / name).read_text(encoding="utf-8", errors="ignore")), name)
            self.assertFalse(name.startswith(("data/", "outputs/")), name)
