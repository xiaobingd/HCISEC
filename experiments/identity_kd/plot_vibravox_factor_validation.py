import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path('outputs/vibravox_factor_validation_20261109');a=json.loads((root/'ablation_summary_public.json').read_text());n=json.loads((root/'neural_summary_public.json').read_text());plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,8.4));fig.suptitle('VibraVox forehead accelerometer: independent dataset validation',fontsize=15)
ax=axes[0,0];keys=['all_factors','without_shape','without_std','without_time','without_amplitude'];labels=['All factors','Remove mean shape','Remove std','Remove timing','Remove amplitude'];v=[a['results'][k]['two_unseen_attackers']['auc'] for k in keys];ax.barh(labels,v,color=['#187d8f','#d76b4c','#87b7c3','#87b7c3','#87b7c3']);ax.invert_yaxis();ax.set_xlim(.5,1.04);ax.set_xlabel('AUC');ax.set_title('A. Mean spectral shape remains informative')
for i,z in enumerate(v):ax.text(z+.004,i,f'{z:.3f}',va='center')
ax=axes[0,1];keys=['original_statistics','statistics_pca192','stats_mlp','resnet18'];labels=['Stats240','PCA192','Stats MLP192','ResNet18'];v=[n['results'][k]['two_unseen_attackers']['auc'] for k in keys];ax.bar(labels,v,color=['#187d8f','#87b7c3','#d7a657','#d76b4c']);ax.set_ylim(.5,1.04);ax.set_ylabel('AUC');ax.set_title('B. Same frontend, identity split and verification')
for i,z in enumerate(v):ax.text(i,z+.01,f'{z:.3f}',ha='center')
ax=axes[1,0];s=a['simulated_gain_query_only'];x=np.arange(3);v1=[a['results']['original_statistics']['two_unseen_attackers']['hter'],s['gain_half']['hter'],s['gain_double']['hter']];v2=[a['results']['shape_only']['two_unseen_attackers']['hter'],s['shape_gain_half']['hter'],s['shape_gain_double']['hter']];ax.bar(x-.18,np.array(v1)*100,.36,label='Original statistics',color='#d7a657');ax.bar(x+.18,np.array(v2)*100,.36,label='Mean shape without level',color='#187d8f');ax.set_xticks(x,['Native','Gain x0.5','Gain x2']);ax.set_ylabel('HTER (%)');ax.set_ylim(0,45);ax.legend(frameon=False);ax.set_title('C. Simulated query gain; enrollment/threshold fixed')
ax=axes[1,1];v=[a['results'][k]['two_unseen_attackers']['hter']*100 for k in ['low_band_statistics','mid_band_statistics','original_statistics']];ax.bar(['10-300 Hz','50-1200 Hz','50-7600 Hz'],v,color=['#d76b4c','#87b7c3','#187d8f']);ax.set_ylim(0,32);ax.set_ylabel('HTER (%)');ax.set_title('D. A wideband sensor needs its own frequency check')
for i,z in enumerate(v):ax.text(i,z+.8,f'{z:.2f}',ha='center')
fig.text(.5,.015,'9 target identities; 1040 source utterances; one seed. Real sentence groups isolated, query content not matched across people.\nGain is a simulation; no measured re-don/session control. Models trained from scratch on four Base identities.',ha='center',fontsize=9,color='#555555');fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig(root/'vibravox_validation.png',dpi=170);fig.savefig(root/'vibravox_validation.pdf');plt.close(fig);print(root/'vibravox_validation.png')
