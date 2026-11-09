"""Prepare ABCS channel0 student/channel1 auxiliary targets; training-text-only bands."""
import argparse,json,hashlib,wave,zlib,time
from pathlib import Path
import numpy as np
from scipy import signal,ndimage

def main(a):
    root=Path(a.root);audit=Path(a.audit);out=Path(a.out);out.mkdir(exist_ok=True,parents=True);start=time.time()
    cov=json.loads((audit/'available_text_coverage_private.json').read_text(encoding='utf-8'))['1'];users=cov['cohort'];texts=sorted(cov['texts'],key=lambda t:hashlib.sha256(('abcs-split-v1|'+t).encode()).hexdigest())
    assert len(users)==8 and len(texts)==125
    mapping={(r['speaker_id'],r['normalized_text']):r for r in json.loads((audit/'transcript_mapping_private.json').read_text(encoding='utf-8')) if r['present'] and r['speaker_id'] in users and r['normalized_text'] in texts}
    archive={x['name']:x for x in json.loads((audit/'archive_central_private.json').read_text())};bins=np.fft.rfftfreq(2048,1/16000);centers=np.geomspace(60,1200,32)
    def spec(v):
        v=v.astype(float)/32768.;v-=v.mean();frames=np.lib.stride_tricks.sliding_window_view(v,1024)[::256].copy();frames-=frames.mean(1,keepdims=True)
        p=np.abs(np.fft.rfft(frames*np.hanning(1024),n=2048,axis=1))**2
        return p,np.log(np.sqrt(np.mean(frames**2,1)+1e-12))
    def resize(v):return ndimage.zoom(v,(1,80/v.shape[1]),order=1).astype('float32')
    inputs=[];targets=[];labels=[];indices=[];roles=[];seen=set();previous={};cohs=[];nulls=[]
    for u,name in enumerate(users):
        for j,text in enumerate(texts):
            row=mapping[name,text];path=root/row['relative_wav'];raw=path.read_bytes();entry=archive['ABCS_database/'+row['relative_wav']];assert len(raw)==entry['size'] and zlib.crc32(raw)==entry['crc32']
            with wave.open(str(path),'rb') as f:
                assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(16000,2,2);pcm=f.readframes(f.getnframes())
            h=hashlib.sha256(pcm).hexdigest();assert h not in seen;seen.add(h);pair=np.frombuffer(pcm,dtype='<i2').reshape(-1,2)
            bp,_=spec(pair[:,0]);b=np.stack([np.interp(np.geomspace(50,1200,80),bins,np.log(t+1e-12)) for t in bp],axis=1);inputs.append(resize(b))
            target=np.zeros((33,80),dtype='float32')
            if j<75:
                ap,env=spec(pair[:,1]);bands=np.stack([np.interp(centers,bins,np.log(t+1e-12)) for t in ap],axis=1);bands-=bands.mean(0,keepdims=True);target=resize(np.vstack([bands,env]))
            if j<60:
                ff,c=signal.coherence(pair[:,0].astype(float),pair[:,1].astype(float),fs=16000,nperseg=2048,noverlap=1024);cohs.append(np.interp(centers,ff,c))
                if j in previous:
                    aa=previous[j];n=min(len(aa),len(pair));ff,c=signal.coherence(pair[:n,0].astype(float),aa[:n],fs=16000,nperseg=2048,noverlap=1024);nulls.append(np.interp(centers,ff,c))
                previous[j]=pair[:,1].astype(float)
            targets.append(target);labels.append(u);indices.append(j);roles.append('fit' if j<60 else 'dev' if j<75 else 'cal' if j<100 else 'query')
        print('PREPARED_USER',u+1,flush=True)
    # Training-only diagnostic ranking, not a claim of a statistically established passband.
    excess=np.mean(cohs,axis=0)-np.mean(nulls,axis=0);selected=np.sort(np.argsort(excess,kind='stable')[-17:]);x=np.stack(inputs);y=np.stack(targets)[:,np.r_[selected,32]];roles=np.array(roles);fit=roles=='fit';allowed=np.isin(roles,['fit','dev'])
    lo,hi=np.percentile(x[fit],[1,99]);x=np.clip(2*(x-lo)/(hi-lo)-1,-1,1).astype('float32');mean=y[fit].mean((0,2),keepdims=True);std=np.maximum(y[fit].std((0,2),keepdims=True),1e-4);y[allowed]=(y[allowed]-mean)/std;assert np.all(y[~allowed]==0)
    np.savez_compressed(out/'inputs_private.npz',x=x,audio=y,user=np.array(labels),index=np.array(indices),role=roles)
    p=dict(status='prepared',users=8,common_texts=125,roles=dict(fit_per_user=60,dev_per_user=15,cal_per_user=25,query_per_user=25),student_channel=0,teacher_channel=1,channel_evidence='user-proposed channel0 BC;official channel index not independently confirmed;retain provisional label',selected_centers=centers[selected].tolist(),selected_train_coherence_excess=excess[selected].tolist(),band_rule='top17of32centers60..1200Hz ranked only on fit paired-minus-different-user same-text coherence;diagnostic candidate ranking,not statistical passband proof',frontend='same80logfreq50..1200Hz x80time;whole utterance STFT64ms16mshop;no waveform warping;global fit-only1/99percentile',audio='17relative logpower targets+logRMS;fit/dev Audio only;cal/query auxiliary targets exactly zero',duplicate_pairs=0,crc_failures=0,elapsed_seconds=time.time()-start,split='exact previous SHA256 text order;original75fit split60fit15dev;25Cal25Query unchanged;allusers same texts perrole',limitations=['one known cohort,closed identity protocol','channel assumption unresolved','official transcription not independently listened','matched text wrong-user coherence null has different speech timing'])
    (out/'preprocessing_public.json').write_text(json.dumps(p,indent=2),encoding='utf-8');print('COMPLETE',json.dumps(p),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--audit',required=True);p.add_argument('--out',required=True);main(p.parse_args())
