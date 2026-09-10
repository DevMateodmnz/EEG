from __future__ import annotations
import csv,json,sys
from pathlib import Path
import mne,numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score,recall_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));mne.set_log_level('ERROR')
from eeg_project.spatial_personalization import build_personalization_pipeline
from eeg_project.final_calibration_learning_curve import subsets
from scripts.evaluate_spatial_personalization import load_validated_source_features,dataset_from_checkpoint
A=ROOT/'docs/assets';P='final_calibration_learning_curve_';SRC=ROOT/'outputs/target_calibration_checkpoints';RAW=ROOT/'outputs/spatial_personalization_checkpoints'
def write(rows,p):
 with p.open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def holm(p):
 o=np.argsort(p);out=np.empty(len(p));last=0
 for i,k in enumerate(o):last=max(last,(len(p)-i)*p[k]);out[k]=min(1,last)
 return out
def main():
 cfg=json.loads((ROOT/'config/spatial_personalization.json').read_text());meta=json.loads((A/'target_calibration_evaluation_metadata.json').read_text());fold=[];subjects=[s for s in range(21,110) if s not in (88,92,100)]
 for s in subjects:
  src,_,_=load_validated_source_features(SRC,s,meta);d=dataset_from_checkpoint(src,np.load(RAW/f'subject_{s:03d}_csp.npy',allow_pickle=False))
  for tr in (6,10,14):
   for size in (14,18,22,26,28):
    variants,test=subsets(d.labels,d.runs,tr,size);ms=[]
    for v in variants:
     model=build_personalization_pipeline('target_csp_ledoit_wolf',cfg);model.fit(d.csp_task_data_volts[v],d.labels[v]);p=model.predict(d.csp_task_data_volts[test]);ms.append((balanced_accuracy_score(d.labels[test],p),recall_score(d.labels[test],p,pos_label=0),recall_score(d.labels[test],p,pos_label=1)))
    fold.append({'subject':s,'test_run':tr,'size':size,'balanced_accuracy':np.mean([x[0] for x in ms]),'fists_recall':np.mean([x[1] for x in ms]),'feet_recall':np.mean([x[2] for x in ms]),'variant_count':len(variants),'test_trial_count':len(test)})
  print('curve',s)
 write(fold,A/f'{P}fold_scores.csv');part=[]
 for s in subjects:
  for n in (14,18,22,26,28):
   r=[x for x in fold if x['subject']==s and x['size']==n];part.append({'subject':s,'size':n,'balanced_accuracy':np.mean([x['balanced_accuracy'] for x in r]),'fists_recall':np.mean([x['fists_recall'] for x in r]),'feet_recall':np.mean([x['feet_recall'] for x in r])})
 write(part,A/f'{P}participant_scores.csv');summary=[]
 for n in (14,18,22,26,28):
  r=[x for x in part if x['size']==n];x=np.array([z['balanced_accuracy'] for z in r]);summary.append({'size':n,'n':86,'median_BA':np.median(x),'q25':np.quantile(x,.25),'q75':np.quantile(x,.75),'minimum':x.min(),'maximum':x.max(),'above_0_5':np.mean(x>.5),'at_least_0_6':np.mean(x>=.6),'at_least_0_7':np.mean(x>=.7),'median_fists_recall':np.median([z['fists_recall'] for z in r]),'median_feet_recall':np.median([z['feet_recall'] for z in r])})
 write(summary,A/f'{P}summary.csv');comp=[];ps=[]
 for lo,hi in ((14,18),(18,22),(22,26),(26,28),(14,28)):
  x=np.array([z['balanced_accuracy'] for z in part if z['size']==hi]);y=np.array([z['balanced_accuracy'] for z in part if z['size']==lo]);d=x-y;p=wilcoxon(d,zero_method='wilcox').pvalue if np.any(d) else 1.;comp.append({'comparison':f'{hi}-{lo}','median_gain':np.median(d),'q25':np.quantile(d,.25),'q75':np.quantile(d,.75),'improved':np.sum(d>0),'worsened':np.sum(d<0),'tied':np.sum(d==0),'p_unadjusted':p});ps.append(p)
 adj=holm(np.array(ps[:4]));
 for i,x in enumerate(comp):x['p_holm_four']=adj[i] if i<4 else 'total_descriptive'
 write(comp,A/f'{P}comparisons.csv');(A/f'{P}metadata.json').write_text(json.dumps({'evaluation_subject_count':86,'test_EEG_used_for_training':False,'historical_outputs_unchanged':True},indent=2)+'\n')
if __name__=='__main__':main()
