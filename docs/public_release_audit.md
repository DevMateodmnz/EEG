# Public release audit note

This note records the public-release boundary discovered during documentation preparation.

## Positive checks

- Raw EEG, NEMAR sources, caches, temporary downloads, and `outputs/` are ignored by Git.
- No credentials, API tokens, private URLs, or environment files were found in tracked source/configuration during the release scan.
- The repository has no configured Git remote, so this preparation did not publish or push anything.

## Provenance migration completed

The path-bearing frozen artifacts were migrated to portable source identifiers and their dependent integrity contracts were refreshed without recomputing scientific outcomes. The machine-readable [migration manifest](../provenance/migrations/public_release_v1.json) and [public provenance note](public_provenance.md) record the exact boundary and verification procedure.

## Licensing

The repository's original code and original project documentation are released under [BSD-3-Clause](../LICENSE). Dataset licenses and terms remain those of their providers and must not be represented as a license for source EEG or other third-party material.
