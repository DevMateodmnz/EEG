import csv,json,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1];A=R/'docs/assets';P='calibration_quantity_and_diversity_evaluation_'
class Tests(unittest.TestCase):
 def test_complete_participant_level_outputs(self):
  m=json.loads((A/f'{P}metadata.json').read_text());self.assertEqual(m['evaluation_subject_count'],86);self.assertFalse(m['test_EEG_used_for_training'])
  with (A/f'{P}fold_scores.csv').open() as f:r=list(csv.DictReader(f))
  self.assertEqual(len(r),86*3*3);self.assertTrue(all(x['variant_count'] in ('1','2') for x in r))
 def test_primary_comparisons_are_participant_counts(self):
  with (A/f'{P}comparisons.csv').open() as f:r=list(csv.DictReader(f))
  self.assertTrue(all(int(x['improved'])+int(x['worsened'])+int(x['tied'])==86 for x in r))
if __name__=='__main__':unittest.main()
