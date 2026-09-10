from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np
from pyriemann.tangentspace import TangentSpace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eeg_project.calibration_quantity_and_diversity import subsets
from scripts.evaluate_spatial_personalization import load_validated_source_features
AS=ROOT/'docs/assets';SRC=ROOT/'outputs/target_calibration_checkpoints';COV=ROOT/'outputs/riemannian_decoder_checkpoints';P='calibration_quantity_and_diversity_evaluation_'
def cos(a,b):return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
def main():
 meta=json.loads((AS/'target_calibration_evaluation_metadata.json').read_text());ref=np.load(AS/'inter_participant_domain_variability_development_tangent_reference.npy');t=TangentSpace(metric='riemann',tsupdate=False);t.reference_=ref;t.n_features_in_=64;rows=[]
 for s in [x for x in range(21,110) if x not in (88,92,100)]:
  source,_,_=load_validated_source_features(SRC,s,meta);z=t.transform(np.load(COV/f'subject_{s:03d}_covariances.npy'))
  for tr in (6,10,14):
   a,b,c,test=subsets(source.labels,source.runs,tr);td=z[test][source.labels[test]==0].mean(0)-z[test][source.labels[test]==1].mean(0)
   for name,variants in [('A',a),('B',b),('C',[c])]:
    directions=[z[v][source.labels[v]==0].mean(0)-z[v][source.labels[v]==1].mean(0) for v in variants];d=np.mean(directions,0);rows.append({'subject':s,'test_run':tr,'condition':name,'calibration_test_direction_cosine':cos(d,td),'calibration_direction_magnitude':np.linalg.norm(d)})
 with (AS/f'{P}geometry_diagnostics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
