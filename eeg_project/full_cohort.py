"""Frozen cohort partitions and provenance for the second EEG replication."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cohort import (
    ReplicationCohortConfig,
    audit_recording,
    load_replication_cohort_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FULL_COHORT_CONFIG_PATH = PROJECT_ROOT / "config" / "full_cohort_replication.json"
PROCESSING_IMPLEMENTATION_PATHS = (
    "eeg_project/preprocessing.py",
    "eeg_project/epoching.py",
    "eeg_project/time_frequency.py",
    "eeg_project/trial_quality.py",
    "eeg_project/cohort.py",
    "eeg_project/cohort_processing.py",
)


@dataclass(frozen=True)
class FrozenFile:
    """A scientific configuration path and its pre-outcome SHA-256."""

    path: str
    sha256: str


@dataclass(frozen=True)
class FullCohortConfig:
    """Chronological cohort identities and second-replication freeze evidence."""

    schema_version: int
    methodology_frozen_at_git_commit: str
    outcomes_inspected_at_freeze: bool
    development_subjects: tuple[int, ...]
    replication_1_subjects: tuple[int, ...]
    replication_2_subjects: tuple[int, ...]
    combined_replication_subjects: tuple[int, ...]
    runs: tuple[int, ...]
    frozen_methodology: dict[str, FrozenFile]
    primary_features_source: str
    replication_success_criteria_source: str
    technical_eligibility_source: str
    central_peak_procedure_source: str
    primary_direction: str
    cohort_2_separate_before_combining: bool
    full_cohort_parameter_refinement_forbidden: bool
    persist_full_tfr_arrays: bool
    processing_order: str
    resume_policy: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_full_cohort_config(
    path: Path = DEFAULT_FULL_COHORT_CONFIG_PATH,
) -> FullCohortConfig:
    """Load and validate the record frozen before cohort-2 outcome inspection."""
    with path.expanduser().resolve().open(encoding="utf-8") as config_file:
        payload: dict[str, Any] = json.load(config_file)
    cohorts = payload["cohorts"]
    config = FullCohortConfig(
        schema_version=int(payload["schema_version"]),
        methodology_frozen_at_git_commit=str(
            payload["methodology_frozen_at_git_commit"]
        ),
        outcomes_inspected_at_freeze=bool(
            payload["outcomes_from_subjects_21_109_inspected_at_freeze"]
        ),
        development_subjects=tuple(int(value) for value in cohorts["development"]),
        replication_1_subjects=tuple(
            int(value) for value in cohorts["replication_1"]
        ),
        replication_2_subjects=tuple(
            int(value) for value in cohorts["replication_2"]
        ),
        combined_replication_subjects=tuple(
            int(value) for value in cohorts["combined_replication"]
        ),
        runs=tuple(int(value) for value in payload["runs"]),
        frozen_methodology={
            str(name): FrozenFile(path=str(record["path"]), sha256=str(record["sha256"]))
            for name, record in payload["frozen_methodology"].items()
        },
        primary_features_source=str(payload["primary_features_source"]),
        replication_success_criteria_source=str(
            payload["replication_success_criteria_source"]
        ),
        technical_eligibility_source=str(payload["technical_eligibility_source"]),
        central_peak_procedure_source=str(payload["central_peak_procedure_source"]),
        primary_direction=str(payload["primary_direction"]),
        cohort_2_separate_before_combining=bool(
            payload["cohort_2_outcomes_must_be_analyzed_separately_before_combining"]
        ),
        full_cohort_parameter_refinement_forbidden=bool(
            payload["full_cohort_parameter_refinement_forbidden"]
        ),
        persist_full_tfr_arrays=bool(payload["persist_full_tfr_arrays"]),
        processing_order=str(payload["processing_order"]),
        resume_policy=str(payload["resume_policy"]),
    )
    validate_full_cohort_config(config)
    return config


def validate_full_cohort_config(config: FullCohortConfig) -> None:
    """Reject cohort overlap, scientific drift, and a false pre-outcome record."""
    if config.schema_version != 1:
        raise ValueError(f"Unsupported full-cohort schema: {config.schema_version}.")
    if config.methodology_frozen_at_git_commit != (
        "395df8ebc79b1658ff91ed96c90b41fd954d037e"
    ):
        raise ValueError("The second replication must remain frozen at commit 395df8e.")
    if config.outcomes_inspected_at_freeze:
        raise ValueError("The freeze record must precede cohort-2 outcome inspection.")
    if config.development_subjects != (1,):
        raise ValueError("Development cohort must contain only subject 1.")
    if config.replication_1_subjects != tuple(range(2, 21)):
        raise ValueError("Replication cohort 1 must contain subjects 2-20.")
    if config.replication_2_subjects != tuple(range(21, 110)):
        raise ValueError("Replication cohort 2 must contain subjects 21-109.")
    if config.combined_replication_subjects != tuple(range(2, 110)):
        raise ValueError("Combined replication must contain subjects 2-109.")
    cohort_sets = (
        set(config.development_subjects),
        set(config.replication_1_subjects),
        set(config.replication_2_subjects),
    )
    if any(cohort_sets[left] & cohort_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Chronological cohorts must not overlap.")
    if config.runs != (6, 10, 14):
        raise ValueError("Full-cohort runs must remain 6, 10, and 14.")
    if not config.cohort_2_separate_before_combining:
        raise ValueError("Cohort 2 must be evaluated separately before combination.")
    if not config.full_cohort_parameter_refinement_forbidden:
        raise ValueError("Outcome-driven full-cohort parameter refinement is forbidden.")
    if config.persist_full_tfr_arrays:
        raise ValueError("Full TFR arrays must not be persisted.")
    required_sources = {
        "preprocessing",
        "epoching",
        "event_related_spectral",
        "trial_quality",
        "replication_policy",
    }
    if set(config.frozen_methodology) != required_sources:
        raise ValueError("The full freeze record must cover every scientific policy.")
    for name, record in config.frozen_methodology.items():
        path = (PROJECT_ROOT / record.path).resolve()
        if not path.is_file() or _sha256(path) != record.sha256:
            raise RuntimeError(f"Frozen methodology hash mismatch: {name} ({path}).")


def load_frozen_replication_policy(
    config: FullCohortConfig,
) -> ReplicationCohortConfig:
    """Load the one existing source for features, eligibility, and success rules."""
    record = config.frozen_methodology["replication_policy"]
    return load_replication_cohort_config((PROJECT_ROOT / record.path).resolve())


def cohort_label(subject: int, config: FullCohortConfig) -> str:
    """Return the permanent chronological role for a requested subject."""
    if subject in config.development_subjects:
        return "development"
    if subject in config.replication_1_subjects:
        return "replication_1"
    if subject in config.replication_2_subjects:
        return "replication_2"
    raise ValueError(f"Subject {subject} is outside the frozen full cohort.")


def audit_requested_subjects(
    subjects: Sequence[int],
    runs: Sequence[int],
    replication_policy: ReplicationCohortConfig,
    *,
    data_directory: Path,
) -> list[dict[str, object]]:
    """Audit explicit subjects without using any event-related outcome."""
    requested = tuple(int(subject) for subject in subjects)
    if len(requested) != len(set(requested)):
        raise ValueError("Audit subjects must be unique.")
    if tuple(int(run) for run in runs) != replication_policy.runs:
        raise ValueError("Audit runs differ from the frozen replication policy.")
    return [
        audit_recording(
            subject,
            run,
            replication_policy,
            data_directory=data_directory,
        )
        for subject in requested
        for run in runs
    ]


def eligibility_for_requested_subjects(
    audit_rows: Sequence[Mapping[str, object]],
    subjects: Sequence[int],
    full_config: FullCohortConfig,
) -> list[dict[str, object]]:
    """Apply the frozen all-runs eligibility rule to explicit cohort subjects."""
    output: list[dict[str, object]] = []
    requested_runs = set(full_config.runs)
    for subject in subjects:
        rows = [row for row in audit_rows if int(row["subject"]) == int(subject)]
        observed_runs = {int(row["run"]) for row in rows}
        compatible_runs = {
            int(row["run"]) for row in rows if bool(row["protocol_compatible"])
        }
        failures = [
            f"run{row['run']}:{row['failure_reason']}"
            for row in rows
            if not bool(row["protocol_compatible"])
        ]
        missing_runs = sorted(requested_runs - observed_runs)
        failures.extend(f"run{run}:missing_audit_row" for run in missing_runs)
        eligible = compatible_runs == requested_runs
        output.append(
            {
                "subject": int(subject),
                "cohort": cohort_label(int(subject), full_config),
                "requested_runs": "/".join(str(run) for run in full_config.runs),
                "compatible_runs": "/".join(str(run) for run in sorted(compatible_runs)),
                "compatible_run_count": len(compatible_runs),
                "technical_eligibility": (
                    "eligible" if eligible else "technically_incompatible"
                ),
                "primary_analysis_included": eligible,
                "partial_run_analysis_used": False,
                "outcome_used_for_eligibility": False,
                "failure_reason": ";".join(failures),
            }
        )
    return output


def full_cohort_fingerprint(
    config: FullCohortConfig,
    config_path: Path = DEFAULT_FULL_COHORT_CONFIG_PATH,
) -> str:
    """Fingerprint cohort identities, frozen policies, and processing implementation."""
    digest = hashlib.sha256(config_path.expanduser().resolve().read_bytes())
    digest.update(config.methodology_frozen_at_git_commit.encode())
    for name, record in sorted(config.frozen_methodology.items()):
        digest.update(name.encode())
        digest.update(record.sha256.encode())
    for relative_path in PROCESSING_IMPLEMENTATION_PATHS:
        path = PROJECT_ROOT / relative_path
        digest.update(relative_path.encode())
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def checkpoint_payload(
    subject: int,
    cohort: str,
    result: Mapping[str, object],
    methodology_fingerprint: str,
    source_hashes: Mapping[str, str],
) -> dict[str, object]:
    """Create a deterministic compact checkpoint without full TFR arrays."""
    return _json_compatible(
        {
            "schema_version": 1,
            "subject": subject,
            "cohort": cohort,
            "methodology_fingerprint": methodology_fingerprint,
            "source_hashes": dict(sorted(source_hashes.items())),
            "result": result,
            "full_tfr_arrays_persisted": False,
        }
    )


def checkpoint_digest(payload: Mapping[str, object]) -> str:
    """Hash normalized checkpoint content for deterministic regeneration checks."""
    encoded = json.dumps(
        _json_compatible(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_subject_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write one configuration/source-validated subject checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(
            _json_compatible(payload),
            output_file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output_file.write("\n")
    temporary.replace(path)


def load_subject_checkpoint(
    path: Path,
    subject: int,
    cohort: str,
    methodology_fingerprint: str,
    source_hashes: Mapping[str, str],
) -> dict[str, object] | None:
    """Return a valid checkpoint or None; stale/partial files are never reused."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, json.JSONDecodeError):
        return None
    expected_hashes = dict(sorted(source_hashes.items()))
    if not (
        payload.get("schema_version") == 1
        and payload.get("subject") == subject
        and payload.get("cohort") == cohort
        and payload.get("methodology_fingerprint") == methodology_fingerprint
        and payload.get("source_hashes") == expected_hashes
        and payload.get("full_tfr_arrays_persisted") is False
        and isinstance(payload.get("result"), dict)
    ):
        return None
    result = payload["result"]
    required = {
        "trial_rows",
        "feature_rows",
        "run_rows",
        "peak_row",
        "qc_row",
        "task_tfr_shape",
        "rest_tfr_shape",
    }
    if set(result) != required:
        return None
    return payload
