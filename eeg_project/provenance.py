"""Portable identifiers for files used as scientific provenance.

Physical filesystem locations are runtime details.  Frozen public artifacts use
these stable identifiers instead, while data-loading code resolves real paths
from its configured data root.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


PHYSIONET_MARKER = ("MNE-eegbci-data", "files", "eegmmidb", "1.0.0")
PHYSIONET_PREFIX = "physionet/eegmmidb/1.0.0"
LEE2019_PREFIX = "lee2019"


def canonical_source_id(value: str) -> str:
    """Return a deterministic public identifier for an approved EEG source.

    Existing identifiers are accepted unchanged.  Absolute paths are accepted
    only when they contain a known dataset marker; arbitrary machine paths fail
    loudly rather than leaking into an artifact.
    """
    text = value.replace("\\", "/")
    if text.startswith(f"{PHYSIONET_PREFIX}/") or text.startswith(f"{LEE2019_PREFIX}/"):
        return text
    parts = PurePosixPath(text).parts
    for index in range(len(parts) - len(PHYSIONET_MARKER) + 1):
        if parts[index : index + len(PHYSIONET_MARKER)] == PHYSIONET_MARKER:
            suffix = parts[index + len(PHYSIONET_MARKER) :]
            if not suffix:
                raise ValueError("PhysioNet provenance must identify a source file.")
            return "/".join((PHYSIONET_PREFIX, *suffix))
    raise ValueError(f"Source path is outside approved provenance roots: {value!r}")


def resolve_source_id(identifier: str, data_root: Path) -> Path:
    """Resolve a canonical identifier using caller-configured local data storage."""
    canonical = canonical_source_id(identifier)
    if canonical.startswith(f"{PHYSIONET_PREFIX}/"):
        suffix = canonical.removeprefix(f"{PHYSIONET_PREFIX}/")
        return data_root / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0" / suffix
    if canonical.startswith(f"{LEE2019_PREFIX}/"):
        return data_root / canonical
    raise ValueError(f"Unsupported canonical source identifier: {identifier!r}")
