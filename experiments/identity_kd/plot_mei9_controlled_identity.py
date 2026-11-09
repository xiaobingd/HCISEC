import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path('outputs/mei9_controlled_identity_20261109');a=json.loads((root/'ablation_summary_public.json').read_text());n=json.loads((root/'neural_summary_public.json').read_text());d=json.loads((root/'descriptive_public.json').read_text());plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(13,9));fig.suptitle('Local nine-user HMD study: information sources and controlled encoders',fontsize=15)
names=['All factors','Remove mean shape','Remove std','Remove timing','Remove amplitude'];keys=['all_factors','without_mean_shape','without_std','without_time','without_amplitude'];values=[a['results']['crossword'][k]['two_unseen_attackers']['auc'] for k in keys];ax=axes[0,0];ax.barh(names,values,color=['#187d8f','#d76b4c','#87b7c3','#87b7c3','#87b7c3']);ax.invert_yaxis();ax.set_xlim(.5,1);ax.set_xlabel('Unseen-attacker AUC');ax.set_title('A. Mean spectral shape supplies the strongest ranking cue')
for i,v in enumerate(values):ax.text(v+.004,i,f'{v:.3f}',va='center')
groups=['Mean shape','Std','Timing','Amplitude','Collection proxies'];ks=['shape','std','time','amplitude','quality'];x=np.arange(len(ks));ax=axes[0,1];ax.bar(x-.18,[d['component_variance'][k]['user_r2']*100 for k in ks],.36,label='User grouping',color='#187d8f');ax.bar(x+.18,[d['component_variance'][k]['date_r2']*100 for k in ks],.36,label='Date grouping',color='#d7a657');ax.set_xticks(x,groups,rotation=15);ax.set_ylim(0,75);ax.set_ylabel('Explained variation (%)');ax.legend(frameon=False);ax.set_title('B. User/date associations overlap; they are not causal shares')
ax=axes[1,0];stress=a['simulated_query_only_stress']['crossword'];v=[a['results']['crossword']['original_statistics']['two_unseen_attackers']['hter']]+[stress[k]['hter'] for k in ['gain_half','gain_double','axis_XY_swap']];ax.bar(['Native','Gain x0.5','Gain x2','Swap X/Y'],np.array(v)*100,color=['#187d8f','#d76b4c','#d76b4c','#d7a657']);ax.set_ylim(0,60);ax.set_ylabel('HTER (%)');ax.set_title('C. Query-only perturbations, fixed enrollment and threshold')
for i,z in enumerate(v):ax.text(i,z*100+1,f'{z*100:.1f}',ha='center')
ax=axes[1,1];methods=['original_statistics','statistics_pca192','stats_mlp','resnet18'];x=np.arange(4)
for j,(cond,label,color) in enumerate([('sameword','Same-word assumption','#87b7c3'),('crossword','Cross-word assumption','#187d8f')]):
    v=np.array([n['results'][cond][k]['two_unseen_attackers']['hter'] for k in methods])*100;ax.bar(x+(j-.5)*.34,v,.34,label=label,color=color)
ax.set_xticks(x,['Statistics720','PCA192','Stats MLP192','ResNet18 192'],rotation=12);ax.set_ylim(0,40);ax.set_ylabel('HTER (%)');ax.legend(frameon=False,fontsize=9);ax.set_title('D. Same frontend, identity folds and verification rules')
fig.text(.5,.015,'9 target identities; one training seed. Content labels inferred. One headset; no independent re-don/cross-day repeats.\nGain and axis tests are simulated sensitivity checks, not measured wearing robustness.',ha='center',fontsize=9,color='#555555');fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(root/'controlled_identity_diagnosis.png',dpi=170);fig.savefig(root/'controlled_identity_diagnosis.pdf');plt.close(fig);print(root/'controlled_identity_diagnosis.png')
