"""Regression guards for frozen participant-geometry artifacts."""
from __future__ import annotations
import csv, hashlib, json, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; ASSETS=ROOT/'docs/assets'; PREFIX='inter_participant_domain_variability_evaluation_'
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.meta=json.loads((ASSETS/f'{PREFIX}metadata.json').read_text())
 def test_reference_is_development_only_and_evaluation_complete(self):
  final=json.loads((ROOT/'config/inter_participant_domain_variability_final.json').read_text())
  self.assertEqual(final['development_reference']['subjects'],list(range(1,21)))
  self.assertFalse(self.meta['tangent_reference_fit_evaluation_subjects_used'])
  self.assertEqual(self.meta['evaluation_subject_count'],86); self.assertEqual(self.meta['checkpoint_reuse_count'],86)
 def test_participant_rows_preserve_runs_labels_and_algebra_outputs(self):
  with (ASSETS/f'{PREFIX}participant_geometry.csv').open() as f: rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),86); self.assertEqual({int(r['subject']) for r in rows},set(range(21,110))-{88,92,100})
  self.assertTrue(all(float(r['within_subject_different_task_distance'])>0 for r in rows))
  self.assertTrue(all(r['all_three_run_alignments_positive'] in ('True','False') for r in rows))
 def test_hashes_and_historical_scores_are_guarded(self):
  self.assertTrue(self.meta['historical_scores_unchanged'])
  for name,digest in self.meta['artifact_sha256'].items(): self.assertEqual(hashlib.sha256((ASSETS/name).read_bytes()).hexdigest(),digest)
 def test_primary_family_is_exactly_six_participant_level_tests(self):
  with (ASSETS/f'{PREFIX}primary_correlations.csv').open() as f: rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),6); self.assertTrue(all(r['observational_unit']=='participant' and r['n']=='86' for r in rows))
if __name__=='__main__': unittest.main()
