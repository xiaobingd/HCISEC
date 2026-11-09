"""Audit cached raw VibraVox forehead signals and prepare a metadata-selected cohort."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,time,unicodedata
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy import signal

ROOT=Path('/autodl-fs/data/vibravox_dual_teacher_20261105');OUT=Path('/autodl-fs/data/vibravox_factor_validation_20261109')
def canon(t):return ' '.join(unicodedata.normalize('NFC',t).casefold().split())
def h(t):return hashlib.sha256(t.encode()).hexdigest()
def role(t):
    k=int(h('vib-factor-text-v1|'+t)[:8],16)%100
    return 'fit' if k<50 else 'develop' if k<65 else 'registration' if k<80 else 'query'
def main():
    OUT.mkdir(exist_ok=True,parents=True);start=time.time();rows=sum([json.loads(p.read_text()) for p in sorted(ROOT.glob('paired_index_*.json'))],[]);pool=defaultdict(lambda:defaultdict(list));seen_text={};duplicates=0
    for r in sorted(rows,key=lambda r:r['file']):
        text=canon(r['normalized_text']);key=(r['speaker_id'],text)
        if key in seen_text:duplicates+=1;continue
        seen_text[key]=True
        if r['sample_rate']==16000 and r['duration']>=2.0 and (ROOT/'paired'/r['file']).exists():pool[r['speaker_id']][role(text)].append(dict(r,canonical_text=text))
    eligible=[u for u,rs in pool.items() if all(len(rs[k])>=n for k,n in [('fit',50),('develop',10),('registration',12),('query',24)])];eligible.sort(key=lambda u:h('vib-factor-speaker-v1|'+u));assert len(eligible)>=9,('eligible',len(eligible));speakers=eligible[:9];selected=[]
    for u,name in enumerate(speakers,1):
        for k,cap in [('fit',64),('develop',16),('registration',12),('query',24)]:
            rr=sorted(pool[name][k],key=lambda r:h('vib-factor-record-v1|'+r['canonical_text']))[:cap]
            for r in rr:selected.append(dict(r,user=u,new_role=k))
    text_roles=defaultdict(set);bytext=defaultdict(set)
    for r in selected:text_roles[r['canonical_text']].add(r['new_role']);bytext[r['canonical_text']].add(r['user'])
    assert all(len(v)==1 for v in text_roles.values());source_hashes=set();x=[];low=[];mid=[];amplitude=[];meta=[];peaks=[]
    def spectrum(w,centers):
        f,t,z=signal.stft(w,fs=16000,nperseg=4096,noverlap=3072,nfft=16384,boundary=None,padded=False);p=np.log(np.abs(z)**2+1e-20);return np.stack([np.interp(centers,f,p[:,j]) for j in range(p.shape[1])],axis=1).astype('float32')
    for r in selected:
        a=np.load(ROOT/'paired'/r['file']);w=a['acc'].astype('float32');assert np.isfinite(w).all() and w.ndim==1 and len(w)>=32000
        digest=hashlib.sha256(w.tobytes()).hexdigest();assert digest==r['acc_sha256'];assert digest not in source_hashes;source_hashes.add(digest)
        i=(len(w)-32000)//2;w=w[i:i+32000].astype(float);w-=w.mean();bb=spectrum(w,np.geomspace(50,7600,40));assert bb.shape==(40,28);x.append(bb[None]);low.append(spectrum(w,np.geomspace(10,300,40))[None]);mid.append(spectrum(w,np.geomspace(50,1200,40))[None]);amplitude.append(np.log(np.r_[np.sqrt(np.mean(w*w)+1e-20),np.max(np.abs(w)),np.quantile(np.abs(w),[.5,.9,.99])]+1e-20));peaks.append(float(np.max(np.abs(w))));meta.append(dict(user=r['user'],role=r['new_role'],text=r['canonical_text'],source_file=r['file'],source_speaker=r['speaker_id'],sentence_id=r['sentence_id'],original_duration=r['duration']))
        if len(meta)%100==0:print('PREPARED',len(meta),flush=True)
    d=dict(spectrum=np.stack(x),low_spectrum=np.stack(low),mid_spectrum=np.stack(mid),amplitude=np.array(amplitude,dtype='float32'),user=np.array([r['user'] for r in meta]),role=np.array([r['role'] for r in meta]),text=np.array([r['text'] for r in meta]),sample_id=np.array([r['source_file'] for r in meta]))
    np.savez_compressed(OUT/'inputs_private.npz',**d);(OUT/'source_manifest_private.json').write_text(json.dumps(dict(speakers=speakers,records=meta),ensure_ascii=False))
    public=dict(state='complete',source='Cnam-LMSSC/vibravox speech_clean existing paired extraction',source_sensor='audio.forehead_accelerometer,one-axis physical accelerometer;not HMD vVelocity',available_records=len(rows),available_speakers=len(pool),eligible_speakers=len(eligible),selected_speakers=9,selected_records=len(meta),role_counts=dict(Counter(d['role'])),source_payload_hashes_verified=len(meta),exact_payload_duplicates=0,same_user_duplicate_texts_deduplicated=duplicates,text_groups_crossing_roles=0,common_texts_all9=sum(len(v)==9 for v in bytext.values()),max_speakers_sharing_one_text=max(map(len,bytext.values())),independent_same_text_repeats='one canonical utterance per user/text;do not claim matched same-text train/test',frontend='one centered2sec clip per original utterance,DC removal,16kHz,4096Hann/1024hop/16384FFT,1x40x28;primary50..7600Hz;raw-stat controls10..300 and50..1200;no perrecord amplitude normalization',original_cache='extraction inspected:decodefloat32+resample_poly to16kHz,without perrecord gain normalization;new hash verifies this cache,not uninspected original archive',peak_quantiles=np.quantile(peaks,[0,.1,.5,.9,1]).tolist(),selection='hash of real normalized text defines global50/15/15/20fit/dev/register/query;metadata-count eligible;hashspeakerorder,not auth scores',limits=['no verified redon/session/device timestamps in cached metadata','different query texts across identities;not matched same-content attack test','one-axis physical acceleration differs from local 3-axis tracking velocity','nine-user exploratory subset,single seed;not fullVibraVoxbenchmark','primarywideband respects sensor bandwidth;not a numerically identical band to localHMD'],elapsed_seconds=time.time()-start)
    (OUT/'preprocessing_public.json').write_text(json.dumps(public,indent=2));print('COMPLETE',json.dumps(public),flush=True)
if __name__=='__main__':main()
