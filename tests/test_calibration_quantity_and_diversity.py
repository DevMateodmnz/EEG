import unittest,numpy as np
from eeg_project.calibration_quantity_and_diversity import subsets
class Tests(unittest.TestCase):
 def test_subsets_are_balanced_deterministic_and_test_free(self):
  y=np.tile([0]*8+[1]*8,3);r=np.repeat([6,10,14],16);a,b,c,t=subsets(y,r,6)
  self.assertEqual([len(x) for x in a],[14,14]);self.assertEqual([len(x) for x in b],[14,14]);self.assertEqual(len(c),32);self.assertTrue(all(np.bincount(y[x],minlength=2).tolist()==[7,7] for x in a+b));self.assertTrue(all(not set(x)&set(t) for x in a+b+[c]))
if __name__=='__main__':unittest.main()
