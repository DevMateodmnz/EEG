import csv,hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'docs/assets';P='domain_vs_direction_alignment_evaluation_'
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m=json.loads((A/f'{P}metadata.json').read_text())
 def test_complete_and_no_test_fitting(self):
  self.assertEqual(self.m['evaluation_subject_count'],86);self.assertEqual(self.m['source_model_fit_count'],1);self.assertFalse(self.m['test_EEG_used_for_transform_fit'])
  with (A/f'{P}scenario_scores.csv').open() as f: rows=list(csv.DictReader(f))
  self.assertEqual(len(rows),86*3*3)
  grouped={}
  for row in rows: grouped.setdefault((row['subject'],row['calibration_run']),set()).add(row['test_trial_count'])
  self.assertTrue(all(len(counts)==1 for counts in grouped.values()))
 def test_identical_test_trials_and_frozen_outputs(self):
  self.assertTrue(self.m['historical_artifacts_unchanged'])
  for n,h in self.m['artifact_sha256'].items():self.assertEqual(hashlib.sha256((A/n).read_bytes()).hexdigest(),h)
 def test_expected_three_participant_level_comparisons(self):
  with (A/f'{P}comparisons.csv').open() as f: rows=list(csv.DictReader(f))
  self.assertEqual([r['comparison'] for r in rows],['B-A','C-B','C-A']);self.assertTrue(all(int(r['improved'])+int(r['worsened'])+int(r['tied'])==86 for r in rows))
if __name__=='__main__':unittest.main()
