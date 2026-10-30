import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/autodl-fs/data/mei_identity_relation_kd_20261030');r=json.loads((root/'summary_public.json').read_text())
keys=['E0_ACC_CE','E1_concat_relation','E2_interaction_relation','E3_full','E4_ACC_stability','E5_full_shuffled','E6_self_relation']
names=['ACC CE','Concat relation','Interaction relation','Full method','Stability only','Full shuffled','ACC self relation']
fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
for ax,backend in zip(axes,['cosine','RBF']):
 values=[100*r['K20'][backend][k]['hter']['mean'] for k in keys];err=[100*r['K20'][backend][k]['hter']['std'] for k in keys]
 ax.bar(np.arange(len(keys)),values,yerr=err,capsize=3,color=['#576b7f','#94b5c8','#5a97b1','#d38351','#7aa486','#a8a1bc','#bbb59d'])
 ax.set_xticks(range(len(keys)),names,rotation=42,ha='right');ax.set_title(backend+' K=20');ax.axhline(values[0],ls='--',color='#576b7f',lw=1)
 for j,v in enumerate(values):ax.text(j,v+err[j]+.3,'%.2f'%v,ha='center',fontsize=8)
 ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_ylim(0,50)
axes[0].set_ylabel('HTER (%) - lower is better');fig.suptitle('Nine known identities, held-out words, 3 seeds\nACC-only deployment; 400 updates per student; all attack sources retained')
fig.tight_layout();fig.savefig(root/'comparison_public.png',dpi=180);plt.close(fig)

