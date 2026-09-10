# Repository guide

| Location | Purpose |
| --- | --- |
| `eeg_project/` | Reusable analysis, preprocessing, cohort, and decoder modules |
| `config/` | Versioned frozen policies and method settings |
| `scripts/` | Research entry points: acquisition, preprocessing, analysis, evaluation |
| `tests/` | Unit, integrity, artifact, and leakage-regression tests |
| `docs/` | Scientific reports, methods, and public-facing navigation |
| `docs/assets/` | Frozen tables, figures, provenance, and test-checked result artifacts |
| `data/` | Downloaded raw EEG/cache; ignored and never committed |
| `outputs/` | Local checkpoints, smoke outputs, and acquisition journals; ignored |

Start with [`README.md`](../README.md), then the study-specific report relevant to the question. Do not modify a frozen config or artifact in place to create a new analysis: create a clearly separated, versioned study and preserve parent hashes. The Lee2019 commands are acquisition/protocol tooling only until the source gate completes.
