"""Show only matched shared-target comparisons for the two-second context follow-up."""
import os,json,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('outputs/mei_audio_repair_20261018');RES=ROOT/'full_context_review'
d=json.loads((RES/'summary.json').read_text());runs=d['runs'];assert len(runs)==2
summary={name:{k:float(np.mean([r[name]['macro'][k] for r in runs])) for k in runs[0][name]['macro']} for name in ['old_local','initial_full','final']}
control={k:float(np.mean([r['macro'][k] for r in d['controls']])) for k in d['controls'][0]['macro']}
public=dict(summary=summary,controls=control,protocol=d['protocol'],seed_results=[dict(seed=r['seed'],old_local=r['old_local']['macro'],initial_full=r['initial_full']['macro'],final=r['final']['macro'],final_extra=r['final']['extra']['macro']) for r in runs],scope=d['scope'])
paired={};names=sorted(r['recording'] for r in runs[0]['final']['per_record'])
for label in ['old_local','time_template']:
    deltas=[]
    for run,ctrl in zip(runs,d['controls']):
        final={r['recording']:r['audio_mae'] for r in run['final']['per_record']}
        baseline={r['recording']:r['audio_mae'] for r in run['old_local']['per_record']} if label=='old_local' else {r['recording']:r['time_template_audio_mae'] for r in ctrl['per_record']}
        deltas.append([final[n]-baseline[n] for n in names])
    delta=np.mean(deltas,axis=0);rng=np.random.default_rng(20261019);boot=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(1)
    paired[label]=dict(new_minus_baseline=float(delta.mean()),record_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),n_new_lower_error=int(np.sum(delta<0)),n_records=len(delta),limits='Conditional on reusedsame-personVAL/currentmodels; seedsaveraged, not newusers/sessions or modelselection uncertainty.')
public['paired_audio_deltas']=paired
public['source_hashes']={n:hashlib.sha256((Path('work')/n).read_bytes()).hexdigest() for n in ['export_mei_full_context.py','train_mei_full_context.py','pack_mei_full_context.py','report_mei_full_context.py']}
(ROOT/'full_context_summary.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
p=dict(np.load(RES/'full_42_plots.npz'));fig,axs=plt.subplots(6,4,figsize=(15,13),layout='constrained')
for c in range(4):
    arrays=[p['real_audio'][c],p['old_audio'][c],p['generated_audio'][c],p['time_template'][c],p['real_ACC'][c,:195].reshape(3,65,65).mean(0),p['cycle_ACC'][c,:195].reshape(3,65,65).mean(0)]
    for row,a in enumerate(arrays):axs[row,c].imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma',extent=[.08,.72,0,80 if row<4 else 500]);axs[row,c].set_xlabel('Time within original window(s)')
    axs[0,c].set_title(str(p['ids'][c]),fontsize=7)
for row,label in enumerate(['Real microphone','Old0.8s model','New2s-context model','No-ACC timing template','Real ACC XYZ mean','New2s-context cycle ACC']):axs[row,0].set_ylabel(label)
fig.suptitle('Same actual target/interiorframes and shared scales / fixedseed42 sources\nOffline2s futurecontext, completecycles; not a causal real-time or identity claim');fig.savefig(ROOT/'full_context_comparison.png',dpi=140);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained')
arrays=[('Real microphone',p['real_audio'][0]),('Old0.8s model',p['old_audio'][0]),('New2s context',p['generated_audio'][0]),('No-ACC timing template',p['time_template'][0]),('Real ACC',p['real_ACC'][0,:195].reshape(3,65,65).mean(0)),('New2s cycle ACC',p['cycle_ACC'][0,:195].reshape(3,65,65).mean(0))]
for ax,(label,a) in zip(axs.flat,arrays):ax.imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma');ax.set_title(label);ax.set_xlabel('Shared interior frame');ax.set_ylabel('Frequency bin')
fig.suptitle('Fixed first mei VAL example / exact same target / seed42 / completecycles');fig.savefig(ROOT/'full_context_compact.png',dpi=150);plt.close(fig)
w=dict(np.load(RES/'full_42_whole_record_plots.npz'));fig,axs=plt.subplots(6,4,figsize=(16,13),layout='constrained')
for c in range(4):
    arrays=[w['real_audio'][c],w['initial_full_audio'][c],w['generated_audio'][c],w['time_template'][c],w['real_ACC'][c,:195].reshape(3,65,201).mean(0),w['cycle_ACC'][c,:195].reshape(3,65,201).mean(0)]
    for row,a in enumerate(arrays):axs[row,c].imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma',extent=[0,2,0,80 if row<4 else 500]);axs[row,c].set_xlabel('Recording time(s)')
    axs[0,c].set_title(str(w['ids'][c]),fontsize=7)
for row,label in enumerate(['Real microphone','Initial weights on2s input','Adapted2s-context G','No-ACC timing template','Real ACC XYZ mean','Adapted2s cycle ACC']):axs[row,0].set_ylabel(label)
fig.suptitle('Whole2s held-out records, first2sources/task / seed42 / raw outputs and shared scale\nInitial/full use SAME2s input; not a real-time or identity result');fig.savefig(ROOT/'whole_record_comparison.png',dpi=140);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained')
arrays=[('Real microphone /2s',w['real_audio'][0]),('Initial weights /2s input',w['initial_full_audio'][0]),('Adapted full context /2s',w['generated_audio'][0]),('No-ACC timing template /2s',w['time_template'][0]),('Real ACC /2s',w['real_ACC'][0,:195].reshape(3,65,201).mean(0)),('Adapted cycle ACC /2s',w['cycle_ACC'][0,:195].reshape(3,65,201).mean(0))]
for ax,(label,a) in zip(axs.flat,arrays):ax.imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma',extent=[0,2,0,80 if a.shape[0]==80 else 500]);ax.set_title(label,fontsize=10);ax.set_xlabel('Time(s)')
fig.suptitle('Whole fixed first mei VAL recording / seed42 / shared scales / completecycles');fig.savefig(ROOT/'whole_record_compact.png',dpi=150);plt.close(fig)
print(json.dumps(public['summary'],indent=2));print('CONTROLS',json.dumps(control))
print('PAIRED',json.dumps(paired))
