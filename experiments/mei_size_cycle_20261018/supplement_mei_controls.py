"""Add the stronger no-ACC timing control to the completed learning-curve report."""
import os,json,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('outputs/mei_size_cycle_20261018')
d=json.loads((root/'summary_results.json').read_text());control=json.loads((root/'time_template_controls.json').read_text())
public_control=dict(protocol=control['protocol'],templates={n:v['macro'] for n,v in control['templates'].items()},model_comparisons=control['model_comparisons'])
d['time_template_control']=public_control
files=['export_mei_size_cycle.py','train_mei_size_cycle.py','train_mei_size_long180_original.py','train_mei_size_long.py','audit_mei_duplicates.py','audit_mei_time_template.py','report_mei_size_cycle.py','pack_mei_size_review.py','supplement_mei_controls.py']
d['source_hashes']={name:hashlib.sha256((Path('work')/name).read_bytes()).hexdigest() for name in files}
assert d['source_hashes']['train_mei_size_cycle.py']=='4a2820f02b20e606e56225d3973089fa01f71c56eec72019f5f9b81ea8e6ea49'
assert d['source_hashes']['train_mei_size_long.py']=='2df5696013da6a0a89ae176a7642b008f1773155045d1ed58bcc5f65f71dec00'
d['conclusion']='More independent recordings mitigate overfit under adequate training; long180MAE.2771 vs long20.3303. However noACC crop-position TRAIN template MAE.2727/detailcorr.3226 exceeds model.2771/.2480. Therefore conversion/identity front-end success not established. Quantity alone is not the full explanation.'
(root/'summary_results.json').write_text(json.dumps(d,indent=2),encoding='utf-8');(root/'time_template_public.json').write_text(json.dumps(public_control,indent=2),encoding='utf-8')
runs=json.loads((root/'review/results.json').read_text());s=d['summary'];nn=[20,70,180]
fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for ax,key,title in [(axs[0],'audio_mae','Paired audio MAE (lower better)'),(axs[1],'ACC_cycle_mae','ACC-cycle MAE (lower better)'),(axs[2],'wrong_ACC_input_audio_mae_increase','Wrong-record input error increase')]:
    for seed in [42,43]:
        vv=[next(r for r in runs if r['route']=='mei'+str(n) and r['seed']==seed)['validation']['macro'][key] for n in nn]
        ax.plot(nn,vv,'o-',label='Standard updates / seed'+str(seed))
        vv=[next(r for r in runs if r['route']=='mei'+str(n)+'_long' and r['seed']==seed)['validation']['macro'][key] for n in [20,180]]
        ax.plot([20,180],vv,'*--',label='3x updates / seed'+str(seed))
    if key=='audio_mae':
        ax.plot(nn,[control['templates'][str(n)]['macro']['audio_mae'] for n in nn],'D:',color='black',label='No-ACC timing template')
    if key=='wrong_ACC_input_audio_mae_increase':ax.axhline(0,color='gray',ls=':')
    ax.set_title(title,fontsize=10);ax.set_xlabel('Independent training recordings');ax.set_xticks(nn);ax.legend(fontsize=7)
fig.suptitle('Clean mei sources, fixed shared calibration; model improvement still does not beat timing-template audio MAE',fontsize=11)
fig.savefig(root/'learning_curve.png',dpi=150);plt.close(fig)
print(json.dumps(public_control['templates'],indent=2))
