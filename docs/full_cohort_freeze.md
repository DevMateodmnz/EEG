# Full-cohort replication freeze record

This record was created before inspecting spectral outcomes from subjects 21–109.
It defines the chronological scientific roles and locks the second-replication
questions to the verified repository state at commit
`395df8ebc79b1658ff91ed96c90b41fd954d037e`.

## Sequential evidence

| Cohort | Subjects | Scientific role |
| --- | --- | --- |
| Development | 1 | Constructed and refined the analysis |
| Replication 1 | 2–20 | First independent application of the frozen subject-1 method |
| Replication 2 | 21–109 | New independent application; outcomes unseen at this freeze |
| Combined replication | 2–109 | Descriptive combined evidence, calculated only after replication 2 |

Subjects 2–20 cannot become new data again when more subjects are added. Subject 1
plus subjects 2–109 may be displayed descriptively, but it is not 109 untouched
confirmatory participants because subject 1 shaped the methodology and the first
replication cohort has already been observed.

## Frozen primary questions

For each technically eligible subject in replication cohort 2, calculate one
subject-level median paired-rest percent change for both-fists imagery in the fixed
12–13 Hz band:

1. C4, predicted direction below zero.
2. C3, predicted direction below zero.

The primary features, technical eligibility, group inference, central-peak procedure,
and strong/mixed/failed criteria are reused directly from
[`replication_cohort.json`](../config/replication_cohort.json). They are not copied
into a second potentially divergent policy.

## Frozen configuration evidence

| Configuration | SHA-256 at freeze |
| --- | --- |
| `preprocessing.json` | `c330847fafe0e9713c8c4f1b64723f81311334a0070fa4f182111216d195cbd2` |
| `epoching.json` | `5aea5629ec751a1e4a45b52cee1e62f029821996f554ddbcd202a27ffc3f076c` |
| `event_related_spectral.json` | `4134542bd823a2105f26f536ba5a63b675484f2f7ec8468e5fe1766d389846eb` |
| `trial_quality.json` | `3215274385453251d6f5036107dec3181393ff81b4f09d16860c6849a91fe453` |
| `replication_cohort.json` | `7ed2015650278bb46bd1d72c22c94c7251f082fe5e41f86534c8a34ec85f8d09` |

The machine-readable source is
[`full_cohort_replication.json`](../config/full_cohort_replication.json). It records
all cohort identities, the full commit, file paths and hashes, outcome-independent
eligibility source, and the prohibition on retrospective parameter refinement.

## Permanent rule

> Subjects 21–109 are a replication cohort. Their outcomes cannot be used to
> retroactively alter primary preprocessing, epoching, paired-rest normalization,
> spectral parameters, frequency bands, sensors, QC definitions, hypotheses, or
> success criteria and then be evaluated as if they were still independent.

If a genuine implementation bug is found, it must be described, fixed for every
affected cohort, followed by complete regeneration and an explicit assessment of
whether earlier conclusions changed.

## Outcome-independent cohort-2 audit

The audit was completed after the freeze and scalable-executor commits, but before
calculating any event-related spectral outcome from subjects 21–109. It requested 89
subjects × 3 runs = 267 EDFs. Every file was acquired and loaded; 258 runs and 86
subjects meet the frozen all-three-runs eligibility rule.

Subjects 88, 92, and 100 are technically incompatible for this fixed replication:

| Subject | Affected runs | Technical evidence | Analysis consequence |
| ---: | --- | --- | --- |
| 88 | 6, 10, 14 | 128 Hz rather than frozen 160 Hz; 19 `T0` and 19 task annotations per run (`10/9`, `10/9`, `9/10`) | Excluded from the primary replication; no replacement |
| 92 | 6, 10, 14 | 128 Hz rather than frozen 160 Hz; 19 `T0` and 19 task annotations per run (`10/9`) | Excluded from the primary replication; no replacement |
| 100 | 6, 10, 14 | 128 Hz rather than frozen 160 Hz; 12 `T0` and 12 task annotations per run (`6/6`) | Excluded from the primary replication; no replacement |

All nine files load successfully and have 64 standardizable channels, but resampling
and changing the frozen event-count protocol would be scientific adaptation rather
than cosmetic portability. No C3/C4 result, ERD direction, effect magnitude, or class
separation informed these decisions.

Among all 267 runs, the stored sample counts are 19,680 (188 runs), 20,000 (57),
19,840 (13), 15,872 (6), and 15,744 (3). The deterministic trailing-zero rule detects
0 samples in 191 runs, 80 in 57, 144 in 13, and 64 in 6. It crops only detected zeros.
Among the 258 compatible runs, the task counterbalance is `8/7` in 131 and `7/8` in
127. No confirmed flat or non-finite sensor was found.

Machine-readable evidence:

- [`replication2_audit.csv`](assets/subjects01-109_full_replication_replication2_audit.csv)
- [`replication2_eligibility.csv`](assets/subjects01-109_full_replication_replication2_eligibility.csv)

### Pairing assertion bug found during execution

The first subject-at-a-time pass stopped at subject 89 before forming that subject's
TFR. The pairing implementation required the EDF `T0` duration field to end at the
following task onset within `1e-10` seconds. Subject 89 retains the correct 160 Hz,
30 alternating annotations, and adjacent `T0→task` identities, but the final two
pairs in each selected run have ±0.05-second duration/onset discrepancies.

That equality was an undocumented implementation assertion, not part of the frozen
paired-rest scientific definition. The fixed analysis identifies a pair by adjacent
annotation order and uses an interior rest-analysis interval. The assertion was
therefore corrected to require that the adjacent `T0` begins before its following
task. A regression test covers rounded duration disagreement. The code-aware
checkpoint fingerprint invalidated all 68 partial pre-fix checkpoints, so every
subject is regenerated. Subject 1 and stored replication-1 values remain exact; no
configuration, window, band, sensor, QC threshold, or replication criterion changed.

The completed outcomes and their interpretation are reported separately in
[Full-cohort EEG replication](full_cohort_replication.md), preserving this file as the
chronological pre-outcome protocol and audit record.
