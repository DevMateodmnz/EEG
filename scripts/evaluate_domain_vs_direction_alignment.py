"""Final frozen alignment-adaptation evaluation using compact covariance checkpoints."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score,recall_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eeg_project.cross_subject_decoding import file_sha256,load_cross_subject_config,load_cross_subject_data
from eeg_project.riemannian_decoding import covariance_data_from_dataset,fit_riemannian_model,load_riemannian_config
from eeg_project.domain_vs_direction_alignment import calibration_split,adapt_vectors
from scripts.evaluate_riemannian_decoder import checkpoint_data
from scripts.evaluate_spatial_personalization import load_validated_source_features
ASSETS=ROOT/'docs/assets';PREFIX='domain_vs_direction_alignment_evaluation_';SRC=ROOT/'outputs/target_calibration_checkpoints';RAW=ROOT/'outputs/spatial_personalization_checkpoints';COV=ROOT/'outputs/riemannian_decoder_checkpoints'
def write(rows,path):
 with path.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def pred(model,v):return model.pipeline.named_steps['classifier'].predict(model.pipeline.named_steps['scaler'].transform(v))
def holm(ps):
 order=np.argsort(ps);out=np.empty(len(ps));last=0
 for i,k in enumerate(order):last=max(last,(len(ps)-i)*ps[k]);out[k]=min(1,last)
 return out
def main():
 final=json.loads((ROOT/'config/domain_vs_direction_alignment_final.json').read_text());
 for x in ('initial_policy','implementation','reference','source_vectors'):
  r=final[x];
  if file_sha256(ROOT/r['path'])!=r['sha256']:raise RuntimeError('Frozen input drift.')
 cfg=load_cross_subject_config();rcfg=load_riemannian_config();train=[covariance_data_from_dataset(load_cross_subject_data(s,cfg)) for s in range(1,21)];model=fit_riemannian_model(train,'shrinkage_lda',rcfg);tangent=model.pipeline.named_steps['tangent'];sm,sd=np.load(ROOT/final['source_vectors']['path'],allow_pickle=False)
 if not np.allclose(tangent.reference_,np.load(ROOT/final['reference']['path'],allow_pickle=False),atol=1e-10):raise RuntimeError('Frozen tangent reference drift.')
 calibration=json.loads((ASSETS/'target_calibration_evaluation_metadata.json').read_text());riem=json.loads((ASSETS/'riemannian_decoder_evaluation_metadata.json').read_text());eligible=[s for s in range(21,110) if s not in (88,92,100)];scenario=[];hashes={}
 for s in eligible:
  source,payload,_=load_validated_source_features(SRC,s,calibration);data,reused=checkpoint_data(s,source,RAW,COV,riem['final_config_sha256'],riem['frozen_core_sha256'])
  if not reused:raise RuntimeError('Covariance checkpoint was not reused.')
  z=tangent.transform(data.covariances)
  for run in (6,10,14):
   cal,test=calibration_split(data.labels,data.runs,run);cv=z[cal];cl=data.labels[cal];y=data.labels[test];test_direction=z[test][y==0].mean(0)-z[test][y==1].mean(0)
   for condition,v in [('A',z[test]),('B',adapt_vectors(cv,cl,z[test],sm,sd,False)[0]),('C',adapt_vectors(cv,cl,z[test],sm,sd,True)[0])]:
    p=pred(model,v);scenario.append({'subject':s,'calibration_run':run,'condition':condition,'balanced_accuracy':balanced_accuracy_score(y,p),'fists_recall':recall_score(y,p,pos_label=0),'feet_recall':recall_score(y,p,pos_label=1),'test_trial_count':len(test),'test_offset_before':float(np.linalg.norm(z[test].mean(0)-sm)),'test_offset_after_B':float(np.linalg.norm(v.mean(0)-sm)) if condition=='B' else '', 'test_alignment_before':float((test_direction@sd)/(np.linalg.norm(test_direction)*np.linalg.norm(sd))),'test_alignment_after_C':float(((v[y==0].mean(0)-v[y==1].mean(0))@sd)/(np.linalg.norm(v[y==0].mean(0)-v[y==1].mean(0))*np.linalg.norm(sd))) if condition=='C' else ''})
  hashes.update(payload['target_source_hashes']);print(f'Alignment S{s:03d}')
 write(scenario,ASSETS/f'{PREFIX}scenario_scores.csv');participants=[]
 for s in eligible:
  for condition in 'ABC':
   rr=[r for r in scenario if r['subject']==s and r['condition']==condition];participants.append({'subject':s,'condition':condition,'balanced_accuracy':float(np.mean([r['balanced_accuracy'] for r in rr])),'fists_recall':float(np.mean([r['fists_recall'] for r in rr])),'feet_recall':float(np.mean([r['feet_recall'] for r in rr]))})
 write(participants,ASSETS/f'{PREFIX}participant_scores.csv');summary=[]
 for c in 'ABC':
  x=np.array([r['balanced_accuracy'] for r in participants if r['condition']==c]);summary.append({'condition':c,'median_BA':np.median(x),'q25':np.quantile(x,.25),'q75':np.quantile(x,.75),'above_0_5_fraction':np.mean(x>.5),'at_least_0_6_fraction':np.mean(x>=.6),'at_least_0_7_fraction':np.mean(x>=.7),'median_fists_recall':np.median([r['fists_recall'] for r in participants if r['condition']==c]),'median_feet_recall':np.median([r['feet_recall'] for r in participants if r['condition']==c])})
 comparisons=[];pvalues=[]
 for name,left,right in [('B-A','B','A'),('C-B','C','B'),('C-A','C','A')]:
  a=np.array([r['balanced_accuracy'] for r in participants if r['condition']==left]);b=np.array([r['balanced_accuracy'] for r in participants if r['condition']==right]);d=a-b; p=wilcoxon(d,zero_method='wilcox',alternative='two-sided').pvalue if np.any(d) else 1.;comparisons.append({'comparison':name,'median_gain':np.median(d),'q25':np.quantile(d,.25),'q75':np.quantile(d,.75),'improved':int(np.sum(d>0)),'worsened':int(np.sum(d<0)),'tied':int(np.sum(d==0)),'p_unadjusted':p});pvalues.append(p)
 for r,p in zip(comparisons,holm(np.array(pvalues))):r['p_holm_three']=p
 write(summary,ASSETS/f'{PREFIX}summary.csv');write(comparisons,ASSETS/f'{PREFIX}comparisons.csv')
 diagnostics=[]
 for c in ('B','C'):
  before='test_offset_before' if c=='B' else 'test_alignment_before';after='test_offset_after_B' if c=='B' else 'test_alignment_after_C';vals=[r for r in scenario if r['condition']==c];diagnostics.append({'condition':c,'mechanism':'offset' if c=='B' else 'direction_alignment','before_median':float(np.median([r[before] for r in vals])),'after_median':float(np.median([r[after] for r in vals]))})
 write(diagnostics,ASSETS/f'{PREFIX}mechanism_diagnostics.csv');artifact={p.name:file_sha256(p) for p in ASSETS.glob(f'{PREFIX}*') if p.name!=f'{PREFIX}metadata.json'};(ASSETS/f'{PREFIX}metadata.json').write_text(json.dumps({'evaluation_subject_count':86,'source_model_fit_count':1,'test_EEG_used_for_transform_fit':False,'historical_artifacts_unchanged':True,'source_hashes':dict(sorted(hashes.items())),'artifact_sha256':artifact},indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
