"""Fixed final recordings, common color scale; current frozen personal G/F200."""
import sys,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/root')
import run_nine_sameword_balanced_20261030 as run
old=run.old;ROOT=Path('/autodl-fs/data/mei_conversion_visuals_20261030')

def record(u,w,rep=7):
 s=old.M['sources'][str(u)];groups=np.asarray(s['content_groups']);i=np.arange(s['rows']);reps=np.where(i<100,i//10,(i-100)//20);ix=np.flatnonzero((groups==w)&(reps==rep));assert len(ix)==1
 j=int(ix[0]);assert j not in s['roles']['gan_fit']+s['roles']['gan_val'];return j

def render(samples,path,title):
 fig,axes=plt.subplots(4,len(samples),figsize=(3.2*len(samples),9.5),squeeze=False,constrained_layout=True)
 names=['Real ACC','Generated Audio: G(ACC)','Real paired Audio','Cycle ACC: F(G(ACC))']
 for col,s in enumerate(samples):
  for row,k in enumerate(['acc','generated','audio','cycle']):
   ax=axes[row,col];im=ax.imshow(s[k],origin='lower',aspect='auto',interpolation='nearest',cmap='magma',vmin=-1,vmax=1)
   if row==0:ax.set_title('U%d | %s | repeat8\nrow %d'%(s['user'],run.same.WORDS[s['group']],s['row']),fontsize=11)
   if col==0:ax.set_ylabel(names[row]+'\nfrequency bin')
   if row==3:ax.set_xlabel('time frame')
   ax.set_xticks([0,40,79]);ax.set_yticks([0,40,79])
 fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='stored normalized value (shared scale)')
 fig.suptitle(title+'\nFrozen epoch200 | held out from G/F training | no per-panel rescaling',fontsize=13)
 fig.savefig(str(path),dpi=160);plt.close(fig)

def main():
 ROOT.mkdir(exist_ok=True);old.records();all_samples=[];metrics=[]
 for u in range(1,10):
  model=old.load_gan(u);a=np.load(str(old.DATA/('acc.%d.npy'%u))).astype('float32')[:,:80,:80];m=np.load(str(old.DATA/('audio.%d.npy'%u))).astype('float32')[:,:80,:80]
  for w in ([5,20,29] if u in [4,5] else [5]):
   j=record(u,w)
   with torch.no_grad():
    generated=model.g_acc2audio(old.te.tensor(a[j:j+1]));cycle=model.g_audio2acc(generated);g=generated[0,0].cpu().numpy();c=cycle[0,0].cpu().numpy()
   s=dict(user=u,group=w,row=j,acc=a[j],audio=m[j],generated=g,cycle=c);all_samples.append(s);metrics.append(dict(user=u,group=w,row=j,paired_audio_l1=float(np.abs(g-m[j]).mean()),paired_audio_corr=float(np.corrcoef(g.ravel(),m[j].ravel())[0,1]),cycle_acc_l1=float(np.abs(c-a[j]).mean()),cycle_acc_corr=float(np.corrcoef(c.ravel(),a[j].ravel())[0,1])))
  del model;torch.cuda.empty_cache()
 render([s for s in all_samples if s['user'] in [1,4,5] and s['group']==5],ROOT/'U1_U4_U5_same_recording.png','Same-word conversion comparison (vocabulary label conditional on row mapping)')
 render([s for s in all_samples if s['group']==5],ROOT/'nine_users_same_recording.png','All nine owners: own ACC through own G/F (not an attacker identity test)')
 for u in [4,5]:render([s for s in all_samples if s['user']==u],ROOT/('U%d_three_words.png'%u),'U%d conversion across three fixed held-out words'%u)
 old.save(ROOT/'metrics_public.json',dict(selection='all users group5 repeat7;U4/U5 also groups20/29 repeat7;fixed before inference,not best samples',color_scale=[-1,1],model='original strict personal G/F epoch200;no new generator training',scope='paired spectral structure only;not evidence of identity preservation or audible speech quality',mapping='word labels conditional on historical10/20period andphase',rows=metrics))
 print('COMPLETE',flush=True)
if __name__=='__main__':main()

