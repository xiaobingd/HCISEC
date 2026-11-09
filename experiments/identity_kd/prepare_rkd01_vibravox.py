"""KD-RKD-01 P0/P1: metadata-first cohort, external frozen Audio teacher."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib,unicodedata,time,sys
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import signal
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,roc_curve

def canon(t):return ' '.join(unicodedata.normalize('NFC',t).casefold().split())
def h(t):return hashlib.sha256(t.encode()).hexdigest()
def textrole(t):
    v=int(h('vib-factor-text-v1|'+t)[:8],16)%100
    return 'fit' if v<50 else 'develop' if v<65 else 'registration' if v<80 else 'query'
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def savearray(p,x):
    p=Path(p);tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:np.save(f,x)
    tmp.replace(p)
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def main(a):
    root=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);started=time.time()
    rows=sum([json.loads(p.read_text()) for p in sorted(root.glob('paired_index_*.json'))],[]);pool=defaultdict(lambda:defaultdict(list));seen=set();dedup=0
    for r in sorted(rows,key=lambda r:r['file']):
        t=canon(r['normalized_text']);key=(r['speaker_id'],t)
        if key in seen:dedup+=1;continue
        seen.add(key)
        if r['sample_rate']==16000 and r['duration']>=2 and (root/'paired'/r['file']).exists():pool[r['speaker_id']][textrole(t)].append(dict(r,text=t))
    eligible=[u for u,rs in pool.items() if all(len(rs[k])>=n for k,n in [('fit',50),('develop',10),('registration',12),('query',24)])];eligible.sort(key=lambda u:h('KD-RKD-01-identity-v1|'+u));assert len(eligible)>=59,len(eligible)
    users=eligible[:59];records=[]
    for u,name in enumerate(users):
        group='base' if u<35 else 'cal' if u<47 else 'test'
        for role,cap in [('fit',64),('develop',16),('registration',12),('query',24)]:
            for r in sorted(pool[name][role],key=lambda r:h('KD-RKD-01-record-v1|'+r['text']))[:cap]:records.append(dict(r,user=u,identity_role=group,role=role))
    bytext=defaultdict(set)
    for r in records:bytext[r['text']].add(r['role'])
    assert all(len(v)==1 for v in bytext.values())
    audit=dict(state='metadata_audited',cache_records=len(rows),cache_users=len(pool),eligible_users=len(eligible),chosen_users=59,identity_roles=dict(base=35,cal=12,test=12),record_roles=dict(Counter(r['role'] for r in records)),records=len(records),duplicate_user_texts_removed=dedup,text_groups_crossing_roles=0,identity_selection='sha256 KD-RKD-01-identity-v1,first59 eligible;not authentication scores',frontend='centered2sec per original recording,DC remove;16kHz STFT4096/1024hop/16384FFT;40log-spaced50..7600Hz;logpower mean40,std40,relative4segmentmeans160,mean-centered spectralshape40 replaces absolute mean,log amplitude6 appended=246;all scaling Basefit only',teacher='official unmodified VoxCeleb ECAPA192,hashchecked;no VibraVox-finetuned weights',inference='Audio is read only for Base fit/develop;Cal/Test Audio never read;registration/query Audio never read',limits=['VibraVox texts not matched across different users;not ABCS same-prompt experiment','sensor is single-axis forehead accelerometer,not HMD tracking velocity','previous exploratory VibraVox work exists;new fixed protocol is not an independent data collection'])
    dump(out/'audit_public.json',audit);dump(out/'manifest_private.json',dict(users=users,records=records));print(json.dumps(audit),flush=True)
    if a.audit_only:return
    partial=out/'stats_partial_private.npy';features=list(np.load(partial)) if partial.exists() else [];meta=records[:len(features)];hashes={r['acc_sha256'] for r in meta};assert len(hashes)==len(meta)
    def process(r):
        with np.load(root/'paired'/r['file']) as pair:w=pair['acc'].astype('float32')
        dig=hashlib.sha256(w.tobytes()).hexdigest();assert dig==r['acc_sha256'];i=(len(w)-32000)//2;w=w[i:i+32000].astype(float);w-=w.mean()
        f,t,z=signal.stft(w,fs=16000,nperseg=4096,noverlap=3072,nfft=16384,boundary=None,padded=False);lp=np.log(abs(z)**2+1e-20);b=np.stack([np.interp(np.geomspace(50,7600,40),f,lp[:,j]) for j in range(lp.shape[1])],1);m=b.mean(-1);seg=np.stack([b[:,q].mean(-1)-m for q in np.array_split(np.arange(b.shape[1]),4)],0);amp=np.log(np.r_[np.sqrt(np.mean(w*w)+1e-20),abs(w).max(),np.quantile(abs(w),[.5,.9,.99])]+1e-20);v=np.r_[m-m.mean(),b.std(-1),seg.ravel(),m.mean(),amp];assert v.shape==(246,) and np.isfinite(v).all();return v,dig
    remaining=records[len(features):]
    with ThreadPoolExecutor(max_workers=8) as workers:
        for r,(v,dig) in zip(remaining,workers.map(process,remaining)):
            assert dig not in hashes;hashes.add(dig);features.append(v);meta.append(r)
            if len(meta)%250==0:
                savearray(partial,np.array(features,dtype='float32'));print('STATS',len(meta),len(records),flush=True)
    savearray(partial,np.array(features,dtype='float32'))
    d=dict(x=np.array(features,dtype='float32'),user=np.array([r['user'] for r in meta]),group=np.array([r['identity_role'] for r in meta]),role=np.array([r['role'] for r in meta]),text=np.array([r['text'] for r in meta]),sample_id=np.array([r['file'] for r in meta]));np.savez_compressed(out/'inputs_private.npz',**d)
    if a.stats_only:return
    # Import the audited external-model loader, never load local adapted weights.
    sys.path.insert(0,a.code);from run_ecapa_pilot import teacher_load
    import torch
    torch.set_num_threads(1)
    for n in ['model','deps']:
        p=out/n
        if not p.exists():p.symlink_to(root/n,target_is_directory=True)
    model=teacher_load(out);assert all(not p.requires_grad for p in model.parameters());device=next(model.parameters()).device;batchsize=16 if device.type=='cuda' else 4
    official=torch.load(out/'model'/'embedding_model.ckpt',map_location='cpu',weights_only=False);actual=model.mods.embedding_model.state_dict();assert set(official)==set(actual) and all(torch.equal(actual[k].cpu(),official[k]) for k in actual);del official,actual
    ix=np.flatnonzero((d['group']=='base')&np.isin(d['role'],['fit','develop']));dest=out/'audio_targets_private.npy';teacher=np.load(dest) if dest.exists() else np.zeros((len(meta),192),dtype='float32');finished=out/'audio_done_private.json';done=set(json.loads(finished.read_text())) if finished.exists() else set()
    for start in range(0,len(ix),batchsize):
        jj=[int(i) for i in ix[start:start+batchsize] if int(i) not in done]
        if not jj:continue
        waves=[]
        for j in jj:
            r=meta[j]
            with np.load(root/'paired'/r['file']) as p:w=p['audio'].astype('float32')
            assert hashlib.sha256(w.tobytes()).hexdigest()==r['audio_sha256'];i=(len(w)-32000)//2;w=w[i:i+32000].copy();w-=w.mean();waves.append(w)
        with torch.no_grad():teacher[jj]=norm(model.encode_batch(torch.tensor(np.stack(waves),device=device)).squeeze(1).cpu().numpy())
        done.update(jj)
        if start%(batchsize*10)==0 or start+batchsize>=len(ix):
            savearray(dest,teacher);dump(finished,sorted(done));dump(out/'status_public.json',dict(state='extracting_external_teacher',device=str(device),done=len(done),total=len(ix),elapsed_seconds=time.time()-started));print('AUDIO_TARGETS',len(done),len(ix),flush=True)
    assert np.all(np.linalg.norm(teacher[ix],axis=1)>.99);assert np.all(teacher[~np.isin(np.arange(len(meta)),ix)]==0)
    dev=np.flatnonzero((d['group']=='base')&(d['role']=='develop'));z=teacher[dev];y=d['user'][dev];upper=np.triu(np.ones((len(dev),len(dev)),bool),1);same=y[:,None]==y[None,:];sim=z@z.T;labels=same[upper];vals=sim[upper];fp,tp,_=roc_curve(labels,vals);eer=float((fp[np.argmin(abs(fp-(1-tp)))]+1-tp[np.argmin(abs(fp-(1-tp)))])/2)
    # Identity-centroid geometry on Base development only.
    from sklearn.preprocessing import StandardScaler
    fit=(d['group']=='base')&(d['role']=='fit');sc=StandardScaler().fit(d['x'][fit]);sx=norm(sc.transform(d['x'][dev]));ap=norm(np.stack([z[y==u].mean(0) for u in range(35)]));sp=norm(np.stack([sx[y==u].mean(0) for u in range(35)]));uu=np.triu_indices(35,1);rho=float(spearmanr((ap@ap.T)[uu],(sp@sp.T)[uu]).statistic);same_text=upper&~same&(d['text'][dev,None]==d['text'][None,dev])
    bins=np.linspace(-1,1,41);dump(out/'similarity_histograms_public.json',dict(bins=bins.tolist(),same_user_different_text=np.histogram(vals[labels],bins)[0].tolist(),different_user=np.histogram(vals[~labels],bins)[0].tolist(),different_user_same_text=np.histogram(sim[same_text],bins)[0].tolist(),scope='Base development only;pair counts are dependent,not independent subjects'))
    result=dict(state='complete',teacher='external VoxCeleb ECAPA frozen;verified official five asset hashes;loaded embedding state bit-exact to original checkpoint',base_dev_auc=float(roc_auc_score(labels,vals)),base_dev_eer=eer,same_user_cross_text_mean=float(vals[labels].mean()),different_user_mean=float(vals[~labels].mean()),different_user_same_text_pairs=int(same_text.sum()),different_user_same_text_mean=float(sim[same_text].mean()) if same_text.any() else None,identity_geometry_spearman=rho,teacher_records=len(ix),cal_test_audio_reads=0,stats_hashes_verified=len(meta),exact_acc_duplicates=0,teacher_gate='pass' if roc_auc_score(labels,vals)>.8 else 'fail',gate_rule='predeclared Base-dev pair AUC>0.8;not Test-selected;low-level identity diagnostic,not proof of student transfer')
    dump(out/'teacher_diagnostic_public.json',result);dump(out/'status_public.json',dict(state='complete',elapsed_seconds=time.time()-started));print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--code',default='/root');p.add_argument('--audit-only',action='store_true');p.add_argument('--stats-only',action='store_true');main(p.parse_args())
