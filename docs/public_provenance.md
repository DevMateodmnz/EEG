# Public provenance and history boundary

Development occurred in a private repository. Before the first public release, machine-local source identifiers were converted to the portable form `physionet/eegmmidb/1.0.0/S###/S###R##.edf`. At runtime, this identifier is resolved through the configured local data root; the physical root is not frozen in public provenance.

The public Git history intentionally begins at a sanitized reproducible baseline. The author retains an immutable private Git archive of the pre-public history. The public [migration manifest](../provenance/migrations/public_release_v1.json) records SHA-256 continuity, old/new artifact hashes, and the equivalence status without publishing prior path text.

This is a portability and privacy boundary, not an attempt to hide unsuccessful work. The public tree includes a concise [development history](DEVELOPMENT_HISTORY.md), frozen results, methods, configurations, and tests needed to reproduce the research record.

Run the public check with:

```bash
python scripts/verify_public_provenance_migration.py
```

The optional direct comparison requires access to the author's private archived baseline and is therefore not part of ordinary public reproduction.
