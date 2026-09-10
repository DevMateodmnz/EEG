"""Fixed translations and minimal rotations for the adaptation experiment."""
from __future__ import annotations
import numpy as np

def calibration_split(labels, runs, calibration_run):
    labels=np.asarray(labels); runs=np.asarray(runs); calibration=[]
    for label in (0,1):
        idx=np.flatnonzero((runs==calibration_run)&(labels==label));
        if idx.size<7: raise ValueError('Calibration run lacks seven trials per class.')
        calibration.extend(idx[:7])
    calibration=np.array(sorted(calibration)); test=np.flatnonzero(runs!=calibration_run)
    if np.intersect1d(calibration,test).size: raise RuntimeError('Calibration/test overlap.')
    return calibration,test

def minimal_rotation(source, target):
    """Orthogonal plane rotation mapping normalized ``source`` to ``target``."""
    a=np.asarray(source,float); b=np.asarray(target,float); a/=np.linalg.norm(a); b/=np.linalg.norm(b)
    c=float(np.clip(a@b,-1,1)); n=a.size; identity=np.eye(n)
    if c>1-1e-12:return identity
    if c<-1+1e-12:
        basis=np.zeros(n); basis[np.argmin(np.abs(a))]=1.; v=basis-a*(a@basis); v/=np.linalg.norm(v)
        return identity-2*np.outer(a,a)-2*np.outer(v,v)
    v=(b-c*a)/np.sqrt(1-c*c)
    return identity+(c-1)*(np.outer(a,a)+np.outer(v,v))+np.sqrt(1-c*c)*(np.outer(v,a)-np.outer(a,v))

def adapt_vectors(calibration_vectors, calibration_labels, test_vectors, source_midpoint, source_direction, direction=False):
    target_midpoint=np.mean(calibration_vectors,axis=0)
    translated=test_vectors-target_midpoint+source_midpoint
    if not direction:return translated,target_midpoint,None
    target_direction=np.mean(calibration_vectors[calibration_labels==0],axis=0)-np.mean(calibration_vectors[calibration_labels==1],axis=0)
    rotation=minimal_rotation(target_direction,source_direction)
    return source_midpoint+(test_vectors-target_midpoint)@rotation.T,target_midpoint,rotation
