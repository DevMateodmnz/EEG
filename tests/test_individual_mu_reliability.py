import unittest

import numpy as np

from eeg_project.individual_mu_reliability import (
    agreement_proportions,
    availability_counts,
    bootstrap_median_pair_difference,
    bootstrap_median_pair_difference_samples,
    estimate_run_rest_peak,
    pairwise_rows,
    subject_summary,
)


class Tests(unittest.TestCase):
 def test_pairwise_metrics(self):
  rows=pairwise_rows({2:[(6,10.),(10,10.5),(14,11.)],3:[(6,9.)]})
  self.assertEqual(len(rows),3);self.assertEqual([r['abs_difference_hz'] for r in rows],[.5,1.,.5])
  s=subject_summary({2:[(6,10.),(10,10.5),(14,11.)],3:[(6,9.)],4:[]})
  self.assertEqual(s[0]['pairwise_difference_count'],3);self.assertEqual(s[1]['range_hz'],'')
  self.assertEqual(s[2]['min_peak_hz'],'');self.assertEqual(s[2]['median_pairwise_abs_difference_hz'],'')
  self.assertEqual(availability_counts({2:[(6,10.)],3:[],4:[(6,9.),(10,9.5)]},3),{'0':1,'1':1,'2':1,'3':0})
 def test_agreement_is_tolerance_safe_and_ordered(self):
  rows=pairwise_rows({2:[(6,10.),(10,10.25),(14,11.)]})
  agreement=agreement_proportions(rows,(.25,.5,1.0))
  self.assertEqual(agreement['same_grid_bin'],0.0);self.assertEqual(agreement['0.25'],1/3)
  self.assertEqual(agreement['0.5'],1/3);self.assertEqual(agreement['1.0'],1.0)
 def test_subject_bootstrap_is_deterministic(self):
  rows=pairwise_rows({2:[(6,10.),(10,10.5)],3:[(6,9.),(10,10.)]})
  self.assertEqual(bootstrap_median_pair_difference(rows,100,7),bootstrap_median_pair_difference(rows,100,7))
  self.assertFalse(np.array_equal(bootstrap_median_pair_difference_samples(rows,100,7),bootstrap_median_pair_difference_samples(rows,100,8)))
 def test_bootstrap_is_participant_aware_and_degenerate_case_is_exact(self):
  rows=pairwise_rows({2:[(6,10.),(10,10.),(14,10.)],3:[(6,9.),(10,9.),(14,9.)]})
  self.assertEqual(bootstrap_median_pair_difference(rows,100,7),(0.0,0.0))
  # Subject 2 has three retained rows, so it cannot be treated as just one pair row.
  samples=bootstrap_median_pair_difference_samples(pairwise_rows({2:[(6,10.),(10,12.),(14,12.)],3:[(6,9.),(10,9.)]}),20,4)
  self.assertTrue(np.isfinite(samples).all())
 def test_rest_only_single_run_helper_recovers_peak_and_rejects_flat_data(self):
  times=np.arange(640)/160.;signal=8e-6*np.sin(2*np.pi*10*times)
  data=np.tile(signal,(3,3,1));estimate=estimate_run_rest_peak(data,['C3','Cz','C4'],160.,['C3','Cz','C4'],4.,(7.,14.),1.)
  self.assertTrue(estimate.method_available);self.assertEqual(estimate.peak_frequency_hz,10.)
  flat=estimate_run_rest_peak(np.zeros((3,3,640)),['C3','Cz','C4'],160.,['C3','Cz','C4'],4.,(7.,14.),1.)
  self.assertFalse(flat.method_available);self.assertTrue(flat.failure_reason)
if __name__=='__main__':unittest.main()
