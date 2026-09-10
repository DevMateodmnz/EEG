import unittest
import numpy as np
from eeg_project.domain_vs_direction_alignment import calibration_split,minimal_rotation,adapt_vectors
class AlignmentTests(unittest.TestCase):
 def test_rotation_is_orthogonal_and_maps_direction(self):
  a=np.array([1.,2.,0.]);b=np.array([0.,1.,3.]);r=minimal_rotation(a,b)
  self.assertTrue(np.allclose(r.T@r,np.eye(3),atol=1e-10));self.assertTrue(np.allclose(r@(a/np.linalg.norm(a)),b/np.linalg.norm(b),atol=1e-10))
 def test_translation_ignores_labels_and_split_is_disjoint(self):
  labels=np.array([0]*8+[1]*8+[0]*8+[1]*8+[0]*8+[1]*8);runs=np.repeat([6,10,14],16);cal,test=calibration_split(labels,runs,6)
  self.assertEqual(len(cal),14);self.assertEqual(len(test),32);self.assertEqual(len(np.intersect1d(cal,test)),0)
  x=np.arange(42.).reshape(14,3);z=np.ones((2,3));out,_,_=adapt_vectors(x,labels[cal],z,np.zeros(3),np.ones(3),False);self.assertTrue(np.allclose(out,z-np.mean(x,axis=0)))
if __name__=='__main__':unittest.main()
