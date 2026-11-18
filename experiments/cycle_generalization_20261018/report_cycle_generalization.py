"""Report fixed samples and record-macro metrics; never select a model on external VAL."""
import os, json, hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('outputs/cycle_generalization_20261018')
RES=ROOT/'results'
ROUTES=['fixed_full','dense_full','dense_small','dense_small_feature']
LABELS=['Fixed crops / full','Expanded crops / full','Expanded crops / small','Small + feature cycle']

def main():
    runs=json.loads((RES/'results.json').read_text())
    assert len(runs)==8 and {(r['route'],r['seed']) for r in runs}=={(r,s) for r in ROUTES for s in [42,43]}
    summary={}
    for route in ROUTES:
        rr=[r for r in runs if r['route']==route]
        summary[route]={split:{k:dict(mean=float(np.mean([r[split]['macro'][k] for r in rr])),seed_std=float(np.std([r[split]['macro'][k] for r in rr]))) for k in rr[0][split]['macro']} for split in ['train','validation']}
        summary[route]['selected_cycle_epochs']=[r['selection']['best_cycle_epoch'] for r in rr]
        summary[route]['parameters']=rr[0]['refit']['parameters']
    protocol=json.loads((RES/'protocol.json').read_text())
    public=dict(summary=summary,protocol=protocol,runs=runs,limits='Two optimization seeds, one person; external4VAL reused for exploration; no independent authentication outcome. Internal selection uses gradient-held-out recordings but prior20TRAIN global frontend scales. Crops are overlapping, no new subjects/sessions. Raw signal plots/arrays/checkpoints private.')
    (ROOT/'summary_results.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
    with np.load('outputs/activity_alignment_20261018/data_val.npz') as d:va={k:d[k] for k in ['x','y','ids']}
    preds={r:dict(np.load(RES/(r+'_42_predictions.npz'))) for r in ROUTES}
    ug=np.unique(va['ids']);ix=[np.flatnonzero(va['ids']==g)[0] for g in ug]
    for mode in ['audio','cycle']:
        fig,axs=plt.subplots(5,4,figsize=(15,12),layout='constrained')
        for col,i in enumerate(ix):
            arrays=[va['y'][i] if mode=='audio' else va['x'][i]]+[preds[r]['generated_audio' if mode=='audio' else 'cycle_ACC'][i] for r in ROUTES]
            for row,a in enumerate(arrays):
                if mode=='cycle':a=a[:195].reshape(3,65,81).mean(0)
                axs[row,col].imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma',extent=[0,.8,0,80 if mode=='audio' else 500])
                axs[row,col].set_xlabel('Time (s)')
            axs[0,col].set_title(str(ug[col]),fontsize=9)
        for row,label in enumerate(['Real microphone Mel' if mode=='audio' else 'Real ACC XYZ mean']+LABELS):axs[row,0].set_ylabel(label)
        fig.suptitle('Full bidirectional cycles / seed42 / fixed first window of each exploratory VAL recording\nShared scales, raw model outputs; stop chosen on internal TRAIN recordings, no identity claim')
        fig.savefig(ROOT/(mode+'_comparison.png'),dpi=140);plt.close(fig)
    # Predetermined final route, not best-looking/best-scoring VAL sample or route.
    i=ix[0];p=preds['dense_small_feature'];fig,axs=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    arrays=[('Real microphone',va['y'][i]),('Fixed crops / full G',preds['fixed_full']['generated_audio'][i]),('Expanded / full G',preds['dense_full']['generated_audio'][i]),('Small + feature-cycle G',p['generated_audio'][i]),('Real ACC',va['x'][i,:195].reshape(3,65,81).mean(0)),('Small + feature-cycle ACC',p['cycle_ACC'][i,:195].reshape(3,65,81).mean(0))]
    for ax,(label,a) in zip(axs.flat,arrays):ax.imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma');ax.set_title(label);ax.set_xlabel('Time frame');ax.set_ylabel('Frequency bin')
    fig.suptitle(str(ug[0])+' / first fixed window / seed42 / complete cycles, no cosmetic filtering');fig.savefig(ROOT/'compact_comparison.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained');xx=np.arange(4)
    for ax,key,title in [(axs[0],'audio_mae','Paired Mel MAE'),(axs[1],'ACC_cycle_mae','Clean ACC-cycle pixel MAE')]:
        for split,off,color in [('train',-.18,'#3674a3'),('validation',.18,'#de8845')]:ax.bar(xx+off,[summary[r][split][key]['mean'] for r in ROUTES],width=.34,label=split,color=color)
        ax.set_title(title+' (lower better)');ax.set_xticks(xx,LABELS,rotation=20,ha='right',fontsize=8);ax.legend()
    for route,label in zip(ROUTES,LABELS):
        for seed in [42,43]:
            hist=json.loads((RES/(route+'_'+str(seed)+'_selection_history.json')).read_text());hh=[r for r in hist if 'selection_score' in r]
            axs[2].plot([r['epoch'] for r in hh],[r['selection_score'] for r in hh],label=label if seed==42 else None,alpha=.8 if seed==42 else .35)
    axs[2].set_title('Internal recording selection score');axs[2].set_xlabel('Full-cycle epoch');axs[2].legend(fontsize=7)
    fig.savefig(ROOT/'metrics.png',dpi=150);plt.close(fig)
    # TRAIN/VAL clips differ; use first recordings/windows regardless of reconstruction quality.
    ex=dict(np.load(RES/'dense_small_feature_42_train_examples.npz'));fig,axs=plt.subplots(4,4,figsize=(14,10),layout='constrained')
    for c,i in enumerate(ix):
        for row,a in enumerate([ex['real_audio'][c],ex['generated_audio'][c],va['y'][i],p['generated_audio'][i]]):axs[row,c].imshow(a,origin='lower',aspect='auto',vmin=-1,vmax=1,cmap='magma')
        axs[0,c].set_title('TRAIN '+str(ex['ids'][c]),fontsize=8);axs[2,c].set_title('VAL '+str(ug[c]),fontsize=8)
    for row,label in enumerate(['TRAIN real','TRAIN generated','VAL real','VAL generated']):axs[row,0].set_ylabel(label)
    fig.suptitle('Small + feature-cycle, seed42 / different TRAIN and VAL clips / shared scales');fig.savefig(ROOT/'train_val_comparison.png',dpi=150);plt.close(fig)
    print(json.dumps({r:dict(audio_train=summary[r]['train']['audio_mae']['mean'],audio_val=summary[r]['validation']['audio_mae']['mean'],cycle_val=summary[r]['validation']['ACC_cycle_mae']['mean'],detail=summary[r]['validation']['audio_detail_corr']['mean'],envelope=summary[r]['validation']['audio_envelope_corr']['mean'],epochs=summary[r]['selected_cycle_epochs']) for r in ROUTES},indent=2))
if __name__=='__main__':main()
