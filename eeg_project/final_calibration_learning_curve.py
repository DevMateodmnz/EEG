"""Nested deterministic two-run calibration allocations."""
from __future__ import annotations
import numpy as np
ALLOC={14:((4,3),(3,4)),18:((5,4),(4,5)),22:((6,5),(5,6)),26:((7,6),(6,7)),28:((7,7),)}
def subsets(labels,runs,test_run,size):
 available=tuple(x for x in (6,10,14) if x!=test_run);out=[]
 for allocation in ALLOC[size]:
  ids=[]
  for label in (0,1):
   for run,n in zip(available,allocation):
    x=np.flatnonzero((runs==run)&(labels==label));
    if len(x)<n:raise ValueError('insufficient trials')
    ids.extend(x[:n])
  out.append(np.array(sorted(ids)))
 test=np.flatnonzero(runs==test_run)
 if any(np.intersect1d(x,test).size for x in out):raise RuntimeError('test leakage')
 return out,test
