import unittest,numpy as np
from eeg_project.final_calibration_learning_curve import subsets
class Tests(unittest.TestCase):
 def test_nested_balanced_and_disjoint(self):
  y=np.tile([0]*8+[1]*8,3);r=np.repeat([6,10,14],16);prev=None
  for n in (14,18,22,26,28):
   vs,t=subsets(y,r,6,n);self.assertTrue(all(len(v)==n for v in vs));self.assertTrue(all(np.bincount(y[v],minlength=2).tolist()==[n//2,n//2] for v in vs));self.assertTrue(all(not set(v)&set(t) for v in vs))
if __name__=='__main__':unittest.main()
