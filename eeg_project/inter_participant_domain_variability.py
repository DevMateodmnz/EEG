"""Frozen participant-level covariance geometry helpers.

This module deliberately contains no decoder fitting.  It maps class prototypes
to a tangent system already fitted on development participants only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from pyriemann.tangentspace import TangentSpace

from .riemannian_decoding import SubjectCovarianceData

RUNS = (6, 10, 14)


@dataclass(frozen=True)
class RunGeometry:
    subject: int
    run: int
    fists: np.ndarray
    feet: np.ndarray
    midpoint: np.ndarray
    direction: np.ndarray
    fists_trials: int
    feet_trials: int


def fit_development_tangent(datasets: Iterable[SubjectCovarianceData]) -> TangentSpace:
    """Fit exactly one non-updating tangent reference from development covariances."""
    values = list(datasets)
    subjects = [item.subject for item in values]
    if subjects != list(range(1, 21)):
        raise ValueError("The tangent reference must use ordered development subjects 1–20 only.")
    covariances = np.concatenate([item.covariances for item in values], axis=0)
    tangent = TangentSpace(metric="riemann", tsupdate=False)
    tangent.fit(covariances)
    return tangent


def _mean_covariance(covariances: np.ndarray) -> np.ndarray:
    """Return the Riemannian mean via pyRiemann's fitted reference implementation."""
    if covariances.shape[0] < 2:
        raise ValueError("A class prototype needs at least two trials.")
    return TangentSpace(metric="riemann", tsupdate=False).fit(covariances).reference_


def subject_run_geometry(data: SubjectCovarianceData, tangent: TangentSpace, qc_sensitivity: bool = False) -> list[RunGeometry]:
    """Create label-checked class prototypes while preserving each run identity."""
    keep = ~data.qc_candidates if qc_sensitivity else np.ones(data.labels.size, dtype=bool)
    rows: list[RunGeometry] = []
    for run in RUNS:
        run_mask = keep & (data.runs == run)
        prototypes = []
        counts = []
        for label in (0, 1):
            selected = data.covariances[run_mask & (data.labels == label)]
            prototypes.append(_mean_covariance(selected)); counts.append(selected.shape[0])
        fists, feet = tangent.transform(np.stack(prototypes, axis=0))
        rows.append(RunGeometry(data.subject, run, fists, feet, (fists + feet) / 2.0, fists - feet, counts[0], counts[1]))
    return rows


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def participant_geometry(rows: list[RunGeometry]) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight the three run vectors for one participant."""
    if len(rows) != 3 or {row.run for row in rows} != set(RUNS):
        raise ValueError("Participant geometry requires exactly runs 6, 10, and 14.")
    return np.mean([row.midpoint for row in rows], axis=0), np.mean([row.direction for row in rows], axis=0)
