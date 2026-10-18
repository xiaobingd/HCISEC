"""Independent-recording learning curve, fixed examples and paired-record bootstrap."""
import os,json
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('outputs/mei_size_cycle_20261018')
RES=ROOT/'review'
ROUTES=['personal20','mei20','mei70','mei180','mei180_long','mei20_long']

def main():
    runs=json.loads((RES/'results.json').read_text())
    assert len(runs)==12 and {(r['route'],r['seed']) for r in runs}=={(r,s) for r in ROUTES for s in [42,43]}
    summary={}
    for route in ROUTES:
        rr=[r for r in runs if r['route']==route]
        summary[route]={split:{k:dict(mean=float(np.mean([r[split]['macro'][k] for r in rr])),seed_std=float(np.std([r[split]['macro'][k] for r in rr]))) for k in rr[0][split]['macro']} for split in ['train','validation','common_small_train']}
        summary[route]['n_train_recordings']=rr[0]['n_train_recordings'];summary[route]['n_train_windows']=rr[0]['n_train_windows']
    # Average optimization seeds per independent validation source before resampling records.
    paired={}
    for key in ['audio_mae','ACC_cycle_mae','audio_detail_corr','wrong_ACC_input_audio_mae_increase']:
        values={}
        for route in ['mei20','mei70','mei180','mei180_long','mei20_long']:
            rr=[r for r in runs if r['route']==route]
            names=sorted(r['recording'] for r in rr[0]['validation']['per_record'])
            values[route]=np.mean([[{z['recording']:z[key] for z in r['validation']['per_record']}[n] for n in names] for r in rr],axis=0)
        paired[key]={}
        for target in ['mei180','mei180_long']:
            delta=values[target]-values['mei20'];rng=np.random.default_rng(20261018)
            boot=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(1)
            paired[key][target+'_minus_mei20']=dict(delta=float(delta.mean()),record_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),n_records=len(delta),limits='Repeatedly sampled same-person source recordings, seed-averaged. Conditional descriptive interval, not new users/sessions or all model/sampling uncertainty. Long comparison also changes optimization budget.')
        delta=values['mei180_long']-values['mei20_long'];rng=np.random.default_rng(20261018);boot=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(1)
        paired[key]['mei180_long_minus_mei20_long']=dict(delta=float(delta.mean()),record_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist(),n_records=len(delta),limits='Matched360+360 epoch budget. Seed-averaged per-record conditional bootstrap; same person/session, not all optimization/split uncertainty.')
    protocol=json.loads((RES/'protocol.json').read_text())
    # Keep source arrays/private detailed membership out of the public summary.
    data=protocol['data'];protocol['data']={k:v for k,v in data.items() if k not in ['source_audit','crops','nested_subsets','calibration_recordings']}
    protocol['data']['membership_reconstruction']='Original manifest + exporter seed20261018 and duplicate/reserve-key rules; detailed membership kept in private run protocol.'
    lp=json.loads((RES/'long_protocol.json').read_text());lp.pop('data',None)
    slp=json.loads((RES/'small_long_protocol.json').read_text());slp.pop('data',None)
    public=dict(summary=summary,paired_record_deltas=paired,protocol=protocol,long_budget_protocol=lp,small_long_budget_protocol=slp,duplicate_audit=json.loads((ROOT/'duplicate_audit.json').read_text()),scope='All final models full cycles. Primary within-mei clean20/70/180, same60validation source records and shared calibration/model/updates. Additional20long/180long provide a matched3x-budget comparison. Follow-ups were exploratory, not all predeclared. No identity/authentication claim. Test arrays not read this round; historical test evaluation exists.')
    (ROOT/'summary_results.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
    ref=dict(np.load(RES/'plot_references.npz'));pred={r:dict(np.load(RES/(r+'_42_plot_predictions.npz'))) for r in ROUTES}
    for mode in ['audio','cycle']:
        fig,axs=plt.subplots(6,4,figsize=(15,14),layout='constrained')
        for c in range(4):
            arrays=[ref['mei_y'][c] if mode=='audio' else ref['mei_x'][c]]+[pred[r]['generated_audio' if mode=='audio' else 'cycle_ACC'][c] for r in ['mei20','mei70','mei180','mei20_long','mei180_long']]
            for row,a in enumerate(arrays):
                if mode=='cycle':a=a[:195].reshape(3,65,81).mean(0)
                axs[row,c].imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma',extent=[0,.8,0,80 if mode=='audio' else 500]);axs[row,c].set_xlabel('Time (s)')
            axs[0,c].set_title(str(ref['mei_ids'][c]),fontsize=7)
        for row,label in enumerate(['Real microphone Mel' if mode=='audio' else 'Real ACC XYZ mean','20 training recordings','70 training recordings','180 training recordings','20 recordings / 3x updates','180 recordings / 3x updates']):axs[row,0].set_ylabel(label)
        fig.suptitle('Data0723 wangjingmei / raw model outputs / same shared scales / seed42\nFirst two held-out sources per task folder, first crop; complete bidirectional cycles, no identity claim')
        fig.savefig(ROOT/(mode+'_comparison.png'),dpi=140);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained');p=pred['mei180_long'];i=0
    arrays=[('Real microphone',ref['mei_y'][i]),('20 records / 3x updates G',pred['mei20_long']['generated_audio'][i]),('180 records / 3x updates G',p['generated_audio'][i]),('20 records / standard updates G',pred['mei20']['generated_audio'][i]),('Real ACC',ref['mei_x'][i,:195].reshape(3,65,81).mean(0)),('180 records / 3x cycle ACC',p['cycle_ACC'][i,:195].reshape(3,65,81).mean(0))]
    for ax,(label,a) in zip(axs.flat,arrays):ax.imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma');ax.set_title(label);ax.set_xlabel('Time frame');ax.set_ylabel('Frequency bin')
    fig.suptitle('Same mei held-out input / fixed first example / seed42 / shared scales');fig.savefig(ROOT/'compact_comparison.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained');nn=[20,70,180]
    for ax,key,title in [(axs[0],'audio_mae','Paired audio MAE (lower better)'),(axs[1],'ACC_cycle_mae','ACC-cycle MAE (lower better)'),(axs[2],'wrong_ACC_input_audio_mae_increase','Wrong-record input error increase (higher better)')]:
        for seed in [42,43]:
            v=[next(r for r in runs if r['route']=='mei'+str(n) and r['seed']==seed)['validation']['macro'][key] for n in nn]
            ax.plot(nn,v,'o-',label='VAL seed'+str(seed))
            long=next(r for r in runs if r['route']=='mei180_long' and r['seed']==seed)['validation']['macro'][key]
            small=next(r for r in runs if r['route']=='mei20_long' and r['seed']==seed)['validation']['macro'][key]
            ax.plot([20,180],[small,long],'*--',label='3x updates seed'+str(seed))
        if key=='audio_mae':
            ax.plot(nn,[summary['mei'+str(n)]['common_small_train'][key]['mean'] for n in nn],'s--',label='Common20 TRAIN mean')
            ax.axhline(summary['mei20']['validation']['audio_template_mae']['mean'],color='gray',ls=':',label='Fixed20 mean template')
        if key=='wrong_ACC_input_audio_mae_increase':ax.axhline(0,color='gray',ls=':')
        ax.set_title(title,fontsize=10);ax.set_xlabel('Independent training recordings');ax.set_xticks(nn);ax.legend(fontsize=7)
    fig.savefig(ROOT/'learning_curve.png',dpi=150);plt.close(fig)
    print(json.dumps({r:dict(train=summary[r]['train']['audio_mae']['mean'],common_train=summary[r]['common_small_train']['audio_mae']['mean'],val=summary[r]['validation']['audio_mae']['mean'],template=summary[r]['validation']['audio_template_mae']['mean'],cycle=summary[r]['validation']['ACC_cycle_mae']['mean'],detail=summary[r]['validation']['audio_detail_corr']['mean'],wrong_gap=summary[r]['validation']['wrong_ACC_input_audio_mae_increase']['mean']) for r in ROUTES},indent=2))
    print('PAIRED',json.dumps(paired,indent=2))
if __name__=='__main__':main()
