from __future__ import annotations
import csv,json,sys
from pathlib import Path
import mne,numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score,recall_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));mne.set_log_level('ERROR')
from eeg_project.spatial_personalization import build_personalization_pipeline
from eeg_project.calibration_quantity_and_diversity import subsets
from scripts.evaluate_spatial_personalization import load_validated_source_features,dataset_from_checkpoint
AS=ROOT/'docs/assets';P='calibration_quantity_and_diversity_evaluation_';SRC=ROOT/'outputs/target_calibration_checkpoints';RAW=ROOT/'outputs/spatial_personalization_checkpoints';COV=ROOT/'outputs/riemannian_decoder_checkpoints'
def write(rows,p):
 with p.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def holm(p):
 o=np.argsort(p);a=np.empty(len(p));last=0
 for i,k in enumerate(o):last=max(last,(len(p)-i)*p[k]);a[k]=min(1,last)
 return a
def main():
 cfg=json.loads((ROOT/'config/spatial_personalization.json').read_text());calmeta=json.loads((AS/'target_calibration_evaluation_metadata.json').read_text());eligible=[s for s in range(21,110) if s not in (88,92,100)];fold=[]
 for s in eligible:
  source,_,_=load_validated_source_features(SRC,s,calmeta);raw=np.load(RAW/f'subject_{s:03d}_csp.npy',allow_pickle=False);d=dataset_from_checkpoint(source,raw)
  for tr in (6,10,14):
   a,b,c,test=subsets(d.labels,d.runs,tr)
   for name,variants in [('A',a),('B',b),('C',[c])]:
    metrics=[]
    for v in variants:
     model=build_personalization_pipeline('target_csp_ledoit_wolf',cfg);model.fit(d.csp_task_data_volts[v],d.labels[v]);p=model.predict(d.csp_task_data_volts[test]);metrics.append((balanced_accuracy_score(d.labels[test],p),recall_score(d.labels[test],p,pos_label=0),recall_score(d.labels[test],p,pos_label=1)))
    fold.append({'subject':s,'test_run':tr,'condition':name,'balanced_accuracy':float(np.mean([x[0] for x in metrics])),'fists_recall':float(np.mean([x[1] for x in metrics])),'feet_recall':float(np.mean([x[2] for x in metrics])),'variant_count':len(variants),'test_trial_count':len(test)})
  print('Evaluation',s)
 write(fold,AS/f'{P}fold_scores.csv');participants=[]
 for s in eligible:
  for c in 'ABC':
   r=[x for x in fold if x['subject']==s and x['condition']==c];participants.append({'subject':s,'condition':c,'balanced_accuracy':float(np.mean([x['balanced_accuracy'] for x in r])),'fists_recall':float(np.mean([x['fists_recall'] for x in r])),'feet_recall':float(np.mean([x['feet_recall'] for x in r]))})
 write(participants,AS/f'{P}participant_scores.csv');summary=[]
 for c in 'ABC':
  r=[x for x in participants if x['condition']==c];x=np.array([z['balanced_accuracy'] for z in r]);summary.append({'condition':c,'median_BA':np.median(x),'q25':np.quantile(x,.25),'q75':np.quantile(x,.75),'minimum':x.min(),'maximum':x.max(),'above_0_5':np.mean(x>.5),'at_least_0_6':np.mean(x>=.6),'at_least_0_7':np.mean(x>=.7),'median_fists_recall':np.median([z['fists_recall'] for z in r]),'median_feet_recall':np.median([z['feet_recall'] for z in r])})
 write(summary,AS/f'{P}summary.csv');comparisons=[];ps=[]
 for name,l,r in [('B-A','B','A'),('C-B','C','B'),('C-A','C','A')]:
  x=np.array([z['balanced_accuracy'] for z in participants if z['condition']==l]);y=np.array([z['balanced_accuracy'] for z in participants if z['condition']==r]);d=x-y;p=wilcoxon(d,zero_method='wilcox').pvalue if np.any(d) else 1.;comparisons.append({'comparison':name,'median_gain':np.median(d),'q25':np.quantile(d,.25),'q75':np.quantile(d,.75),'improved':np.sum(d>0),'worsened':np.sum(d<0),'tied':np.sum(d==0),'p_unadjusted':p});ps.append(p)
 for z,p in zip(comparisons,holm(np.array(ps[:2]).tolist()+[1]) if False else [None]*3):pass
 # Holm is pre-specified over two primary comparisons only.
 adjusted=holm(np.array(ps[:2]));comparisons[0]['p_holm_two']=adjusted[0];comparisons[1]['p_holm_two']=adjusted[1];comparisons[2]['p_holm_two']='exploratory_total'
 write(comparisons,AS/f'{P}comparisons.csv');(AS/f'{P}metadata.json').write_text(json.dumps({'evaluation_subject_count':86,'historical_outputs_unchanged':True,'test_EEG_used_for_training':False},indent=2)+'\n')
if __name__=='__main__':main()
