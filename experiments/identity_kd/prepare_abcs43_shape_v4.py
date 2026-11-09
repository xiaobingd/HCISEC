"""Metadata-fixed ABCS43 23/8/12 partition with matched evaluation prompts."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,hashlib,zlib,wave,unicodedata
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import signal
def h(t):return hashlib.sha256(t.encode()).hexdigest()
def canon(t):return ' '.join(unicodedata.normalize('NFC',t).casefold().split())
def main(a):
    audit=Path(a.audit);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);mapping=json.loads((audit/'transcript_mapping_private.json').read_text(encoding='utf8'));pool=defaultdict(dict)
    for r in sorted(mapping,key=lambda r:r['relative_wav']):
        if r['present']:pool[r['speaker_id']].setdefault(canon(r['normalized_text']),r)
    users=sorted(pool,key=lambda u:h('abcs43-v1|'+u));assert len(users)==43;base=users[:23];cal=users[23:31];test=users[31:];texts={u:set(pool[u]) for u in users};bc=set.intersection(*(texts[u] for u in base));cc=set.intersection(*(texts[u] for u in cal));tc=set.intersection(*(texts[u] for u in test));order=lambda s:sorted(s,key=lambda t:h('abcs43-text-v1|'+t))
    replay=sorted(bc,key=lambda t:(t in tc,h('abcs43-replay-v1|'+t)))[:4];assert len(replay)==4
    registration=order((cc&tc)-set(replay))[:12];assert len(registration)==12
    test_query=(order(tc-cc-set(replay)-set(registration))+order((tc&cc)-set(replay)-set(registration)))[:24];assert len(test_query)==24
    cal_query=order(cc-set(registration)-set(test_query)-set(replay))[:24];assert len(cal_query)==24
    held=set(registration+test_query+cal_query);assert len(held)==60 and not held&set(replay);selected=[];wordids={t:j for j,t in enumerate(sorted(set.union(*texts.values())))}
    for u,name in enumerate(users):
        if name in base:
            available=texts[name]-held-set(replay);fit=order({t for t in available if int(h('abcs43-fit-dev|'+t)[:8],16)%5!=0})[:56]+replay;dev=order({t for t in available if int(h('abcs43-fit-dev|'+t)[:8],16)%5==0})[:15];assert len(fit)==60 and len(dev)==15
            roles=[('fit',fit),('develop',dev)]
        elif name in cal:roles=[('registration',registration),('calibration',cal_query)]
        else:roles=[('registration',registration),('query',test_query)]
        for role,tt in roles:
            for t in tt:selected.append(dict(pool[name][t],user=u,identity_role='base' if u<23 else 'cal' if u<31 else 'test',role=role,text=t,text_index=wordids[t],replay=t in replay))
    assert len(selected)==2445;group_roles=defaultdict(set)
    for r in selected:group_roles[r['text']].add(r['role'])
    assert all(len(v)==1 for v in group_roles.values());archive={r['name']:r for r in json.loads((audit/'archive_central_private.json').read_text())}
    def process(r):
        path=Path(a.root)/r['relative_wav'];raw=path.read_bytes();entry=archive['ABCS_database/'+r['relative_wav']];assert len(raw)==entry['size'] and zlib.crc32(raw)==entry['crc32']
        with wave.open(str(path),'rb') as f:
            assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(16000,2,2);pcm=f.readframes(f.getnframes())
        pair=np.frombuffer(pcm,dtype='<i2').reshape(-1,2).astype('float64')/32768.;assert len(pair)>=4096;x=pair[:,0];x=x-x.mean();ff,_,zz=signal.stft(x,fs=16000,nperseg=4096,noverlap=3072,nfft=16384,boundary=None,padded=False);lp=np.log(abs(zz)**2+1e-20);b=np.stack([np.interp(np.geomspace(50,7600,40),ff,lp[:,j]) for j in range(lp.shape[1])],1);b=np.stack([np.interp(np.linspace(0,b.shape[1]-1,16),np.arange(b.shape[1]),row) for row in b]).astype('float32');m=b.mean(1);std=b.std(1);segs=np.stack([b[:,q].mean(1) for q in np.array_split(np.arange(16),4)]);amp=np.log(np.r_[np.sqrt(np.mean(x*x)+1e-20),abs(x).max(),np.quantile(abs(x),[.5,.9,.99])]+1e-20);feat=np.r_[m-m.mean(),std,(segs-m).ravel(),m.mean(),amp].astype('float32');stats=np.r_[m,std,segs.ravel()].astype('float32');teacher=None;length=None
        if r['replay'] or r['role']=='registration':
            v=pair[:,1];n=min(len(v),32000);start=max((len(v)-32000)//2,0);v=v[start:start+n].copy();v-=v.mean();teacher=np.pad(v,(0,32000-n)).astype('float32');length=n/32000
        return feat,stats,teacher,length,hashlib.sha256(pcm).hexdigest()
    xx=[];vv=[];aa=[];ll=[];ii=[];seen=set()
    with ThreadPoolExecutor(max_workers=8) as workers:
        for i,(r,result) in enumerate(zip(selected,workers.map(process,selected))):
            x,v,w,l,digest=result;assert digest not in seen;seen.add(digest);xx.append(x);vv.append(v)
            if w is not None:ii.append(i);aa.append(w);ll.append(l)
            if (i+1)%250==0:print('PREPARED',i+1,2445,flush=True)
    assert len(ii)==332;np.savez_compressed(out/'inputs_private.npz',x=np.stack(xx),statistics=np.stack(vv),user=np.array([r['user'] for r in selected]),group=np.array([r['identity_role'] for r in selected]),role=np.array([r['role'] for r in selected]),text_index=np.array([r['text_index'] for r in selected]),replay=np.array([r['replay'] for r in selected]),sample_id=np.array([f'ABCS43-row{i:04}' for i in range(len(selected))]));np.savez_compressed(out/'audio_fit_register_private.npz',indices=np.array(ii),waves=np.stack(aa),lengths=np.array(ll,dtype='float32'))
    (out/'manifest_private.json').write_text(json.dumps(dict(users=users,records=selected,registration=registration,test_query=test_query,cal_query=cal_query,replay=replay),ensure_ascii=False),encoding='utf8');summary=dict(state='prepared',total_users=43,base_users=23,cal_users=8,test_users=12,records=2445,teacher_audio_records=332,query_audio_forward_candidates=0,role_counts=dict(Counter(r['role'] for r in selected)),shared_test_query_texts=24,shared_registration_texts=12,shared_cal_query_texts=24,text_role_overlap=0,crc_checks=2445,exact_pcm_duplicates=0,selection='fixed identity sha256 abcs43-v1,metadata coverage only;all43included;no auth-score selection',channels='ch0 provisional vibration,ch1 provisional Audio;physical mapping unresolved',files={p.name:p.stat().st_size for p in [out/'inputs_private.npz',out/'audio_fit_register_private.npz']});(out/'preparation_public.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--audit',required=True);p.add_argument('--out',required=True);main(p.parse_args())
