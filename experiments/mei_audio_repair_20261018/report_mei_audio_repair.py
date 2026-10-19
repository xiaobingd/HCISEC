"""Compare raw final-cycle outputs with initial model and no-ACC timing template."""
import os,json,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('outputs/mei_audio_repair_20261018');RES=ROOT/'review'
ROUTES=['continue','objective_repair','detail_refiner','conditional_refiner','energy_detail','activity_detail']
LABELS=['Matched continuation','Objective repair','Full-resolution detail','Detail + paired discriminator','Conservative energy detail','Conservative + activity']

def summarize(rows):
    return {split:{k:dict(mean=float(np.mean([r[split]['macro'][k] for r in rows])),seed_std=float(np.std([r[split]['macro'][k] for r in rows]))) for k in rows[0][split]['macro']} for split in ['validation','extra_validation']}

def main():
    runs=json.loads((RES/'results.json').read_text());assert len(runs)==12
    controls=json.loads((RES/'controls.json').read_text());protocol=json.loads((RES/'protocol.json').read_text())
    summary={route:summarize([r for r in runs if r['route']==route]) for route in ROUTES};summary['initial']=summarize(controls['initial'])
    public=dict(summary=summary,time_template_extra=controls['time_template_extra']['macro'],time_template_audio_mae=controls['time_template_audio_mae'],numeric_cycle_controls=controls['perturbation'],protocol=protocol,seed_results=[dict(route=r['route'],seed=r['seed'],validation=r['validation']['macro'],extra_validation=r['extra_validation']['macro'],train=r['train']['macro'],seconds=r['seconds'],parameters=r['parameters']) for r in runs],limits='Single-person60repeatedexploratoryVAL, fixedendpoints/twooptimizerseeds; nohuman speechlabels/listening/WAVE/audioqualityoridentityclaim. FullcycleFonlygetsGoutput. References are inspiration, not reproduction.')
    public['conservative_protocol']=json.loads((RES/'conservative_protocol.json').read_text())
    public['source_hashes']={n:hashlib.sha256((Path('work')/n).read_bytes()).hexdigest() for n in ['train_mei_audio_repair.py','train_mei_conservative_detail.py','audit_pack_mei_audio_repair.py','report_mei_audio_repair.py']}
    (ROOT/'summary_results.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
    ref=dict(np.load(RES/'references.npz'));old=dict(np.load(RES/'initial_42_predictions.npz'));timepred=dict(np.load(RES/'time_template_predictions.npz'))
    preds={r:dict(np.load(RES/(r+'_42_predictions.npz'))) for r in ROUTES}
    for mode in ['audio','cycle']:
        fig,axs=plt.subplots(9 if mode=='audio' else 8,4,figsize=(15,20 if mode=='audio' else 18),layout='constrained')
        for c in range(4):
            key='generated_audio' if mode=='audio' else 'cycle_ACC'
            arrays=[ref['y' if mode=='audio' else 'x'][c],old[key][c]]+([timepred['generated_audio'][c]] if mode=='audio' else [])+[preds[r][key][c] for r in ROUTES]
            for row,a in enumerate(arrays):
                if mode=='cycle':a=a[:195].reshape(3,65,81).mean(0)
                axs[row,c].imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma',extent=[0,.8,0,80 if mode=='audio' else 500]);axs[row,c].set_xlabel('Time (s)')
            axs[0,c].set_title(str(ref['ids'][c]),fontsize=7)
        labels=['Real microphone Mel' if mode=='audio' else 'Real ACC XYZ mean','Initial model']+(['No-ACC timing template'] if mode=='audio' else [])+LABELS
        for row,label in enumerate(labels):axs[row,0].set_ylabel(label)
        fig.suptitle('mei / full bidirectional cycles / seed42 / shared scales and fixed held-out examples\nRaw G outputs, no cosmetic filtering; one person, no identity conclusion')
        fig.savefig(ROOT/(mode+'_comparison.png'),dpi=140);plt.close(fig)
    # Predetermined final route, never the best-looking VAL route/sample.
    final=preds['activity_detail'];fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    arrays=[('Real microphone',ref['y'][0]),('Initial model G',old['generated_audio'][0]),('Conservative + activity G',final['generated_audio'][0]),('No-ACC timing template',timepred['generated_audio'][0]),('Real ACC',ref['x'][0,:195].reshape(3,65,81).mean(0)),('Conservative + activity cycle ACC',final['cycle_ACC'][0,:195].reshape(3,65,81).mean(0))]
    for ax,(title,a) in zip(axs.flat,arrays):ax.imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma');ax.set_title(title);ax.set_xlabel('Time frame');ax.set_ylabel('Frequency bin')
    fig.suptitle('Fixed first mei VAL example / seed42 / complete cycles / same calibration');fig.savefig(ROOT/'compact_comparison.png',dpi=150);plt.close(fig)
    alo,ahi=protocol['normalization_audio'];energy=lambda y:np.log10(np.mean(10**((y-1)*(ahi-alo)/20),axis=0).clip(1e-12))
    fig,axs=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    for c,ax in enumerate(axs.flat):
        ax.plot(np.arange(81)*.01,energy(ref['y'][c]),'k',label='Real microphone')
        ax.plot(np.arange(81)*.01,energy(old['generated_audio'][c]),label='Initial model')
        ax.plot(np.arange(81)*.01,energy(final['generated_audio'][c]),label='Conservative + activity')
        ax.plot(np.arange(81)*.01,energy(timepred['generated_audio'][c]),'--',label='No-ACC timing')
        ax.axhline(protocol['audio_energy_threshold'],color='gray',ls=':',label='TRAIN-only energy threshold');ax.set_title(str(ref['ids'][c]),fontsize=8);ax.set_xlabel('Time (s)');ax.set_ylabel('Log physical Mel power / shared reference');ax.legend(fontsize=7)
    fig.savefig(ROOT/'activity_tracks.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained');names=['initial']+ROUTES;labels=['Initial']+LABELS
    for ax,key,title,split in [(axs[0],'audio_mae','Paired Mel MAE (lower better)','validation'),(axs[1],'local_spectral_contrast_corr','Local spectral contrast correlation','extra_validation'),(axs[2],'activity_false_positive_rate','Energy-activity false positive rate','extra_validation')]:
        ax.bar(np.arange(len(names)),[summary[r][split][key]['mean'] for r in names]);ax.set_xticks(np.arange(len(names)),labels,rotation=25,ha='right',fontsize=7);ax.set_title(title,fontsize=10)
        if key=='audio_mae':ax.axhline(public['time_template_audio_mae'],color='black',ls=':',label='No-ACC timing template');ax.legend(fontsize=7)
        elif key in public['time_template_extra']:ax.axhline(public['time_template_extra'][key],color='black',ls=':')
    fig.suptitle('Two-seed record-macro means; activity is an automatic TRAIN-energy proxy');fig.savefig(ROOT/'metrics.png',dpi=150);plt.close(fig)
    print(json.dumps({r:{**{k:summary[r]['validation'][k]['mean'] for k in ['audio_mae','audio_detail_corr','ACC_cycle_mae','ACC_detail_corr']},**{k:summary[r]['extra_validation'][k]['mean'] for k in ['active_audio_mae','inactive_audio_mae','activity_false_positive_rate','activity_F1','local_spectral_contrast_corr']}} for r in summary},indent=2))
if __name__=='__main__':main()
