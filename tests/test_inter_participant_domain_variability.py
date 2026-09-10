"""Unit guards for frozen participant geometry algebra and leakage boundary."""
from __future__ import annotations
import unittest
import numpy as np
from pyriemann.tangentspace import TangentSpace
from eeg_project.inter_participant_domain_variability import RunGeometry, participant_geometry, subject_run_geometry
from eeg_project.riemannian_decoding import SubjectCovarianceData

class GeometryTests(unittest.TestCase):
    def test_midpoint_and_direction_are_exact(self):
        fists = np.array([3.0, 5.0]); feet = np.array([1.0, 2.0])
        row = RunGeometry(21, 6, fists, feet, (fists + feet) / 2, fists - feet, 7, 8)
        self.assertTrue(np.array_equal(row.midpoint, np.array([2.0, 3.5])))
        self.assertTrue(np.array_equal(row.direction, np.array([2.0, 3.0])))
    def test_participant_aggregation_is_equal_run_weight(self):
        rows = [RunGeometry(21, run, np.zeros(2), np.zeros(2), np.full(2, value), np.full(2, value * 2), 7, 8) for run, value in zip((6,10,14), (1.,2.,3.))]
        midpoint, direction = participant_geometry(rows)
        self.assertTrue(np.array_equal(midpoint, np.array([2.,2.])))
        self.assertTrue(np.array_equal(direction, np.array([4.,4.])))
    def test_class_prototypes_keep_their_run_and_label_membership(self):
        covariances=[]; labels=[]; runs=[]
        for run in (6,10,14):
            for label, scale in ((0,1.0),(1,2.0)):
                covariances.extend([np.eye(64)*scale, np.eye(64)*scale]); labels.extend([label,label]); runs.extend([run,run])
        data=SubjectCovarianceData(21,np.stack(covariances),np.array(labels),np.array(runs),[],np.zeros(12,dtype=bool))
        tangent=TangentSpace(metric='riemann',tsupdate=False); tangent.fit(np.stack(covariances))
        rows=subject_run_geometry(data,tangent)
        self.assertEqual([row.run for row in rows],[6,10,14])
        self.assertTrue(all((row.fists_trials,row.feet_trials)==(2,2) for row in rows))
        self.assertTrue(all(np.linalg.norm(row.direction)>0 for row in rows))

if __name__ == "__main__": unittest.main()
