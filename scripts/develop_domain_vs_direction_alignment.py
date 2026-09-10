"""Development-only LOSO simulation of frozen tangent-space adaptations."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eeg_project.cross_subject_decoding import load_cross_subject_config,load_cross_subject_data,file_sha256
from eeg_project.riemannian_decoding import covariance_data_from_dataset,fit_riemannian_model,load_riemannian_config
from eeg_project.domain_vs_direction_alignment import calibration_split,adapt_vectors
ASSETS=ROOT/'docs/assets';PREFIX='domain_vs_direction_alignment_development_'
def write(rows,path):
 with path.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def predict(model,values): return model.pipeline.named_steps['classifier'].predict(model.pipeline.named_steps['scaler'].transform(values))
def main():
 cfg=load_cross_subject_config();rcfg=load_riemannian_config();data={s:covariance_data_from_dataset(load_cross_subject_data(s,cfg)) for s in range(1,21)};rows=[]
 for subject,target in data.items():
  model=fit_riemannian_model([data[s] for s in data if s!=subject],'shrinkage_lda',rcfg); tangent=model.pipeline.named_steps['tangent'];train=np.concatenate([data[s].covariances for s in data if s!=subject]);tv=tangent.transform(train);tl=np.concatenate([data[s].labels for s in data if s!=subject]);sm=tv.mean(0);sd=tv[tl==0].mean(0)-tv[tl==1].mean(0);z=tangent.transform(target.covariances)
  for run in (6,10,14):
   cal,test=calibration_split(target.labels,target.runs,run);cv=z[cal];cl=target.labels[cal];y=target.labels[test]
   for condition,v in [('A',z[test]),('B',adapt_vectors(cv,cl,z[test],sm,sd,False)[0]),('C',adapt_vectors(cv,cl,z[test],sm,sd,True)[0])]:rows.append({'subject':subject,'calibration_run':run,'condition':condition,'balanced_accuracy':balanced_accuracy_score(y,predict(model,v)),'test_trial_count':len(test)})
  print(f'Development S{subject:03d}')
 write(rows,ASSETS/f'{PREFIX}scenario_scores.csv');summary=[]
 for condition in 'ABC':
  scores=[np.mean([r['balanced_accuracy'] for r in rows if r['subject']==s and r['condition']==condition]) for s in range(1,21)];summary.append({'condition':condition,'median_BA':float(np.median(scores)),'q25':float(np.quantile(scores,.25)),'q75':float(np.quantile(scores,.75))})
 write(summary,ASSETS/f'{PREFIX}summary.csv');(ASSETS/f'{PREFIX}metadata.json').write_text(json.dumps({'development_subjects':list(range(1,21)),'evaluation_loaded':False,'artifact_sha256':{p.name:file_sha256(p) for p in [ASSETS/f'{PREFIX}scenario_scores.csv',ASSETS/f'{PREFIX}summary.csv']}},indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
