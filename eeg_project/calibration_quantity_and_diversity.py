"""Deterministic calibration subsets for fixed target-CSP comparisons."""
from __future__ import annotations
import numpy as np
def indices_for(labels,runs,available,counts):
 out=[]
 for label in (0,1):
  for run,n in zip(available,counts):
   found=np.flatnonzero((runs==run)&(labels==label))
   if found.size<n:raise ValueError('Insufficient retained calibration trials.')
   out.extend(found[:n])
 return np.array(sorted(out))
def subsets(labels,runs,test_run):
 available=tuple(run for run in (6,10,14) if run!=test_run); test=np.flatnonzero(runs==test_run)
 a=[indices_for(labels,runs,(run,),(7,)) for run in available]
 b=[indices_for(labels,runs,available,split) for split in ((4,3),(3,4))]
 c=np.flatnonzero(np.isin(runs,available))
 if any(np.intersect1d(x,test).size for x in a+b+[c]):raise RuntimeError('Test leakage')
 return a,b,c,test
