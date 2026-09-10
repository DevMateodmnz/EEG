"""Resume-safe, source-only acquisition for the frozen Lee2019 study.

This command downloads no classifier artifacts and performs no evaluation.
It pins MOABB to its NEMAR mirror of the authors' original pre-BIDS MAT files,
then accepts a source only after its manifest size and SHA-256 match.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eeg_project.external_cross_session_lee2019 import (
    DEFAULT_DATA_DIRECTORY,
    LEE2019_NEMAR_ID,
    _moabb_dataset,
    lee2019_nemar_provider,
    lee2019_nemar_sources,
    load_study_config,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--retry-attempts", type=int, default=5)
    parser.add_argument("--retry-delay-seconds", type=float, default=60.0)
    return parser.parse_args()


def append_validation(journal: Path, records: list[dict[str, object]]) -> None:
    """Append only source records which have not already been journalled."""
    existing = set()
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("validation") == "sha256_valid":
                existing.add((record.get("subject"), record.get("session"), record.get("source_sha256")))
    with journal.open("a", encoding="utf-8") as handle:
        for record in records:
            key = (record["subject"], record["session"], record["source_sha256"])
            if key not in existing:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = arguments()
    config = load_study_config()
    subjects = args.subjects or config["external_dataset"]["candidate_subjects"]
    allowed = set(config["external_dataset"]["candidate_subjects"])
    if not subjects or any(subject not in allowed for subject in subjects):
        raise ValueError("Requested subjects must be non-empty members of the frozen 1..54 cohort.")
    if len(set(subjects)) != len(subjects):
        raise ValueError("Requested subjects must be unique so the acquisition journal is deterministic.")
    if args.retry_attempts < 1 or args.retry_delay_seconds < 0:
        raise ValueError("Retry attempts must be positive and retry delay must be non-negative.")

    journal = ROOT / "outputs" / "external_cross_session_lee2019" / "acquisition_validation.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    dataset = _moabb_dataset()
    with lee2019_nemar_provider(args.data_dir):
        for subject in subjects:
            try:
                records = lee2019_nemar_sources(subject, args.data_dir)
                append_validation(journal, records)
                print(f"already validated subject {subject}: {len(records)} NEMAR source files", flush=True)
                continue
            except (FileNotFoundError, RuntimeError, ValueError) as initial_error:
                if args.verify_only:
                    raise RuntimeError(f"Subject {subject} is not a validated local source set.") from initial_error
            for attempt in range(1, args.retry_attempts + 1):
                try:
                    # MOABB resolves only this missing participant from the
                    # NEMAR provenance manifest and retains partial transfers.
                    dataset.download(subject_list=[subject], path=args.data_dir, force_update=False, update_path=False, verbose=False)
                    records = lee2019_nemar_sources(subject, args.data_dir)
                    append_validation(journal, records)
                    print(f"validated subject {subject}: {len(records)} NEMAR source files", flush=True)
                    break
                except Exception as error:  # NEMAR wraps transport errors in provider-specific types.
                    print(f"subject {subject} attempt {attempt}/{args.retry_attempts} failed: {error}", flush=True)
                    if attempt == args.retry_attempts:
                        raise RuntimeError(f"NEMAR acquisition exhausted retries for subject {subject}.") from error
                    time.sleep(args.retry_delay_seconds)
    print(f"complete: {len(subjects)} participants, provider=nemar, deposit={LEE2019_NEMAR_ID}", flush=True)


if __name__ == "__main__":
    main()
