# Scientific decision log

This log distinguishes a decision made before outcomes from an observation made after analysis. It supplements frozen JSON configurations; it does not retroactively preregister results.

| Decision | Rationale and trade-off | Timing/status | Implementation |
| --- | --- | --- | --- |
| Inspect recordings and reject invalid tails | A shared constant tail is non-physiological; retaining it would contaminate filters and epochs. This is data-integrity handling, not artifact correction. | Preprocessing contract; **FROZEN** | [preprocessing](preprocessing.md), `config/preprocessing.json` |
| Use average reference and linear-phase FIR settings | A reproducible, explicit sensor-level transform with documented spectral behavior. It does not solve all volume-conduction or artifact concerns. | Pre-outcome for each frozen study | [preprocessing](preprocessing.md) |
| Analyze central C3/C4 task/rest power | These standard central sensors are plausible hand-MI locations, while claims remain sensor-level. | Primary physiological analysis; **FROZEN** | [full cohort](full_cohort_replication.md) |
| Record technical exclusions | Nonstandard sampling/annotation structure makes the frozen pipeline inapplicable. Performance is never an exclusion rule. | Cohort integrity decision; **FROZEN** | full-cohort metadata and tests |
| Use held-out runs for decoder evaluation | Separates fit from evaluation and makes leakage auditable. | Before decoder evaluation; **FROZEN** | fit-audit CSVs and decoder tests |
| Use participant as the inferential unit | Repeated runs/trials from one person are not independent people. | Pre-inference; **FROZEN** | cohort/decoder summaries |
| Test fixed bands before individualized peaks | Fixed bands give a stable reference; individualized estimates can be uncertain or unavailable. | Primary then secondary sequence | [individualized frequency](individual_mu_frequency.md) |
| Add Lee2019 cross-session validation | Held-out runs do not establish day/session transfer. The external study tests transfer without target calibration. | Pre-outcome; **FROZEN / IN PROGRESS** | `config/external_cross_session_lee2019.json` |
| Wait for all external sources | Partial-cohort outcome inspection could bias exclusions or continuation choices. | Pre-outcome gate; **FROZEN** | external acquisition/validation scripts |

Post-hoc observations—including heterogeneous performance or unsuccessful adaptations—are reported as results/limitations, not used to redefine completed protocols.
