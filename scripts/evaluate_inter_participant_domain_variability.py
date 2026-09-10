"""Evaluate frozen participant covariance geometry from validated checkpoints."""
from __future__ import annotations
import csv, hashlib, json, subprocess, sys
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
from pyriemann.tangentspace import TangentSpace
from scipy.stats import spearmanr

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from eeg_project.cross_subject_decoding import file_sha256
from eeg_project.inter_participant_domain_variability import cosine, participant_geometry, subject_run_geometry
from eeg_project.riemannian_decoding import SubjectCovarianceData
from scripts.evaluate_riemannian_decoder import checkpoint_data
from scripts.evaluate_spatial_personalization import load_validated_source_features

ASSETS=PROJECT_ROOT/'docs/assets'; PREFIX='inter_participant_domain_variability_evaluation_'
SOURCE_DIR=PROJECT_ROOT/'outputs/target_calibration_checkpoints'; RAW_DIR=PROJECT_ROOT/'outputs/spatial_personalization_checkpoints'; COV_DIR=PROJECT_ROOT/'outputs/riemannian_decoder_checkpoints'

def rows_csv(rows,path):
    fields=list(rows[0])
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def q(values):
    a=np.asarray(values,float); return {'median':float(np.median(a)),'q25':float(np.quantile(a,.25)),'q75':float(np.quantile(a,.75)),'minimum':float(a.min()),'maximum':float(a.max()),'n':int(a.size)}
def holm(pvalues):
    order=np.argsort(pvalues); adjusted=np.empty(len(pvalues)); running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(pvalues)-rank)*pvalues[index]); adjusted[index]=min(1.,running)
    return adjusted
def pairmean(vectors, other_vectors): return float(np.mean([np.linalg.norm(a-b) for a in vectors for b in other_vectors]))
def reference_tangent(reference):
    tangent=TangentSpace(metric='riemann',tsupdate=False); tangent.reference_=reference; tangent.n_features_in_=64; return tangent

def geometry(data,tangent):
    runs=subject_run_geometry(data,tangent)
    midpoint,direction=participant_geometry(runs)
    fists=np.mean([r.fists for r in runs],axis=0);feet=np.mean([r.feet for r in runs],axis=0)
    within=float(np.mean([np.linalg.norm(r.fists-r.feet) for r in runs]))
    same_run=[]
    for task in ('fists','feet'):
        same_run += [np.linalg.norm(getattr(a,task)-getattr(b,task)) for a,b in combinations(runs,2)]
    run_cos=[cosine(a.direction,b.direction) for a,b in combinations(runs,2)]
    return dict(runs=runs,midpoint=midpoint,direction=direction,fists=fists,feet=feet,within=within,same_run=float(np.mean(same_run)),run_alignment=float(np.mean(run_cos)),all_run_positive=bool(np.all(np.asarray(run_cos)>0)))

def main():
    if subprocess.run(['git','merge-base','--is-ancestor','259c6e7','HEAD'],cwd=PROJECT_ROOT).returncode: raise RuntimeError('Final geometry freeze is not in history.')
    mne.set_log_level('ERROR'); final=json.loads((PROJECT_ROOT/'config/inter_participant_domain_variability_final.json').read_text())
    for record in [final['initial_policy'],final['implementation'],*final['development_reference'].values(),*final['historical_artifacts'].values()]:
        if isinstance(record,dict) and 'path' in record and file_sha256(PROJECT_ROOT/record['path']) != record['sha256']: raise RuntimeError('Frozen input hash drift.')
    reference=np.load(PROJECT_ROOT/final['development_reference']['reference']['path'],allow_pickle=False); source=np.load(PROJECT_ROOT/final['development_reference']['source_vectors']['path'],allow_pickle=False)
    tangent=reference_tangent(reference); source_midpoint,source_direction=source
    calibration=json.loads((ASSETS/'target_calibration_evaluation_metadata.json').read_text()); riem=json.loads((ASSETS/'riemannian_decoder_evaluation_metadata.json').read_text())
    final_riem_hash=riem['final_config_sha256']; core_hash=riem['frozen_core_sha256']; subjects=final['evaluation']['eligible_subject_count']; geometries={}; source_hashes={}
    eligible=json.loads((PROJECT_ROOT/'config/inter_participant_domain_variability.json').read_text())['cohorts']['evaluation_eligible_subjects']
    for index,subject in enumerate(eligible,1):
        src,payload,_=load_validated_source_features(SOURCE_DIR,subject,calibration)
        data,reused=checkpoint_data(subject,src,RAW_DIR,COV_DIR,final_riem_hash,core_hash)
        if not reused: raise RuntimeError(f'Checkpoint S{subject:03d} was not validated/reused.')
        geometries[subject]=geometry(data,tangent); source_hashes.update(payload['target_source_hashes']); print(f'Geometry: S{subject:03d} ({index}/{len(eligible)})')
    # Frozen historical outcomes: one primary row per subject.
    csp={int(r['subject']):float(r['primary_mean_run_balanced_accuracy']) for r in read_csv(ASSETS/'cross_subject_decoder_evaluation_subject_scores.csv') if r['model']=='csp_4_empirical'}
    riem_scores={int(r['subject']):float(r['primary_mean_run_balanced_accuracy']) for r in read_csv(ASSETS/'riemannian_decoder_evaluation_subject_scores.csv') if r['model']=='shrinkage_lda' and r['qc_sensitivity']=='False'}
    participant=[]
    for subject in eligible:
        g=geometries[subject]
        participant.append({'subject':subject,'source_midpoint_offset':float(np.linalg.norm(g['midpoint']-source_midpoint)),'task_direction_magnitude':float(np.linalg.norm(g['direction'])),'source_direction_alignment':cosine(g['direction'],source_direction),'within_subject_different_task_distance':g['within'],'same_subject_same_task_across_runs_distance':g['same_run'],'run_direction_alignment':g['run_alignment'],'all_three_run_alignments_positive':g['all_run_positive'],'zero_shot_csp_ba':csp[subject],'zero_shot_riemannian_ba':riem_scores[subject],'poor_zero_shot_csp':csp[subject]<=.5})
    # Each person supplies one average distance to all other people: no pairwise pseudo-replication.
    for row in participant:
        s=row['subject']; others=[x for x in eligible if x!=s]; g=geometries[s]
        same_task = [np.linalg.norm(g['fists']-geometries[o]['fists']) + np.linalg.norm(g['feet']-geometries[o]['feet']) for o in others]
        different_task = [np.linalg.norm(g['fists']-geometries[o]['feet']) + np.linalg.norm(g['feet']-geometries[o]['fists']) for o in others]
        row['different_subject_same_task_distance']=float(np.mean(same_task) / 2)
        row['different_subject_different_task_distance']=float(np.mean(different_task) / 2)
        row['mean_pairwise_direction_alignment']=float(np.mean([cosine(g['direction'],geometries[o]['direction']) for o in others]))
    correlations=[]; ps=[]
    for predictor in ('source_midpoint_offset','source_direction_alignment','task_direction_magnitude'):
        for outcome in ('zero_shot_csp_ba','zero_shot_riemannian_ba'):
            result=spearmanr([r[predictor] for r in participant],[r[outcome] for r in participant]); correlations.append({'predictor':predictor,'outcome':outcome,'spearman_rho':float(result.statistic),'p_unadjusted':float(result.pvalue),'observational_unit':'participant','n':len(participant)});ps.append(float(result.pvalue))
    for row,p in zip(correlations,holm(np.array(ps))):row['p_holm_six_tests']=float(p)
    distance_rows=[]
    for key in ('within_subject_different_task_distance','different_subject_same_task_distance','different_subject_different_task_distance','same_subject_same_task_across_runs_distance'):
        distance_rows.append({'comparison':key,'observational_unit':'participant_mean',**q([r[key] for r in participant])})
    summary=[{'metric':'participant_count','value':len(participant)},*({'metric':f'{key}_{stat}','value':value} for key in ('task_direction_magnitude','source_direction_alignment','run_direction_alignment','mean_pairwise_direction_alignment') for stat,value in q([r[key] for r in participant]).items()),{'metric':'all_three_run_alignments_positive_fraction','value':float(np.mean([r['all_three_run_alignments_positive'] for r in participant]))}]
    poor=[r for r in participant if r['poor_zero_shot_csp']]; good=[r for r in participant if not r['poor_zero_shot_csp']]
    subgroup=[]
    for key in ('task_direction_magnitude','source_direction_alignment','run_direction_alignment','source_midpoint_offset'):
        subgroup += [{'group':'poor_CSP_<=0.50','metric':key,**q([r[key] for r in poor])},{'group':'other','metric':key,**q([r[key] for r in good])}]
    rows_csv(participant,ASSETS/f'{PREFIX}participant_geometry.csv'); rows_csv(distance_rows,ASSETS/f'{PREFIX}distance_summary.csv'); rows_csv(correlations,ASSETS/f'{PREFIX}primary_correlations.csv'); rows_csv(summary,ASSETS/f'{PREFIX}summary.csv'); rows_csv(subgroup,ASSETS/f'{PREFIX}poor_subgroup.csv')
    # Four pre-specified compact figures.
    a=np.array([r['source_midpoint_offset'] for r in participant]); b=np.array([r['task_direction_magnitude'] for r in participant]);
    fig,ax=plt.subplots(figsize=(5.5,4));ax.scatter(a,b,c='#4477aa');ax.set(xlabel='Offset from development midpoint',ylabel='Task-direction magnitude');fig.tight_layout();fig.savefig(ASSETS/f'{PREFIX}offset_vs_magnitude.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,4));ax.hist([r['source_direction_alignment'] for r in participant],bins=15,color='#4477aa');ax.axvline(0,color='0.3',ls='--');ax.set(xlabel='Cosine alignment with development direction',ylabel='Participants');fig.tight_layout();fig.savefig(ASSETS/f'{PREFIX}source_alignment_distribution.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,4));ax.hist([r['run_direction_alignment'] for r in participant],bins=15,color='#4477aa');ax.axvline(0,color='0.3',ls='--');ax.set(xlabel='Mean pairwise run-direction cosine',ylabel='Participants');fig.tight_layout();fig.savefig(ASSETS/f'{PREFIX}run_stability.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,4));ax.scatter([r['source_direction_alignment'] for r in participant],[r['zero_shot_csp_ba'] for r in participant],c='#4477aa',label='CSP');ax.scatter([r['source_direction_alignment'] for r in participant],[r['zero_shot_riemannian_ba'] for r in participant],c='#cc6677',alpha=.65,label='Riemannian');ax.legend();ax.set(xlabel='Source-direction alignment',ylabel='Frozen zero-shot balanced accuracy');fig.tight_layout();fig.savefig(ASSETS/f'{PREFIX}alignment_vs_zero_shot_accuracy.png',dpi=180);plt.close(fig)
    artifact={p.name:file_sha256(p) for p in ASSETS.glob(f'{PREFIX}*') if p.name!=f'{PREFIX}metadata.json'}
    metadata={'schema_version':1,'stage':'evaluation_geometry_complete','evaluation_subject_count':len(participant),'checkpoint_reuse_count':len(participant),'tangent_reference_fit_evaluation_subjects_used':False,'tangent_reference_sha256':file_sha256(PROJECT_ROOT/final['development_reference']['reference']['path']),'historical_scores_unchanged':True,'source_hashes':dict(sorted(source_hashes.items())),'artifact_sha256':artifact}
    (ASSETS/f'{PREFIX}metadata.json').write_text(json.dumps(metadata,indent=2,sort_keys=True,allow_nan=False)+'\n')
if __name__=='__main__':main()
