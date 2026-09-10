import csv,hashlib,json,unittest
from pathlib import Path
class Tests(unittest.TestCase):
 def test_primary_artifacts_are_complete_and_rest_only(self):
  root=Path(__file__).resolve().parents[1];assets=root/'docs/assets';m=json.loads((assets/'individual_mu_reliability_summary.json').read_text())
  self.assertEqual(m['schema_version'],2);self.assertEqual(m['eligible_subject_count'],105);self.assertEqual(m['availability_counts']['3'],105);self.assertEqual(m['pair_count'],315)
  self.assertEqual(m['bootstrap_median_95'],[.5,.75]);self.assertAlmostEqual(m['agreement']['same_grid_bin'],44/315)
  self.assertAlmostEqual(m['agreement']['0.25'],124/315)
  with (assets/'individual_mu_reliability_run_peaks.csv').open(newline='') as f: peaks=list(csv.DictReader(f))
  keys=[(int(x['subject']),int(x['run']),float(x['prominence_threshold_db'])) for x in peaks]
  self.assertEqual(len(peaks),945);self.assertEqual(keys,sorted(keys))
  self.assertTrue(all(x['estimator']=='v1_0_exact_single_run_rest' for x in peaks))
  self.assertTrue(all(x['peak_frequency_hz']=='' or 7<=float(x['peak_frequency_hz'])<=14 for x in peaks))
  with (assets/'individual_mu_reliability_subject_summary.csv').open(newline='') as f: subjects=list(csv.DictReader(f))
  self.assertEqual(len(subjects),105);self.assertEqual(subjects[0]['subject'],'2');self.assertIn('median_pairwise_abs_difference_hz',subjects[0])
  manifest=json.loads((assets/'individual_mu_reliability_provenance.json').read_text())
  self.assertEqual(manifest['parent_v1_0_config_sha256'],m['parent_v1_0_config_sha256']);self.assertEqual(len(manifest['raw_source_hashes']),315)
  self.assertEqual(hashlib.sha256((root/'config/individual_mu_reliability.json').read_bytes()).hexdigest(),m['config_sha256'])
 def test_historical_individualized_decision_and_config_are_unchanged(self):
  root=Path(__file__).resolve().parents[1];assets=root/'docs/assets';metadata=json.loads((assets/'individual_mu_heldout_evaluation_metadata.json').read_text())
  self.assertEqual(metadata['overall_method_category'],'no improvement')
  config=json.loads((root/'config/individual_mu_frequency.json').read_text())
  parent=json.loads((root/'config/individual_mu_reliability.json').read_text())['parent_v1_0']
  self.assertEqual(hashlib.sha256((root/parent['config']).read_bytes()).hexdigest(),parent['sha256'])
if __name__=='__main__':unittest.main()
