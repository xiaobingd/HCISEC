"""Prepare anonymous ABCS V4 inputs and only fit/register teacher waveforms."""
import argparse,json,zlib,wave
from pathlib import Path
import numpy as np

def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);prior=Path(a.prior);d=dict(np.load(prior/'inputs_private.npz'));records=json.loads((prior/'manifest_private.json').read_text());assert len(records)==992 and d['spectrum'].shape==(992,1,40,16)
    groups=[set(d['text'][d['role']==r]) for r in ['fit','develop','registration','calibration','query']];assert all(not groups[i]&groups[j] for i in range(5) for j in range(i))
    b=d['spectrum'];m=b.mean(-1);level=m.mean(-1);seg=np.stack([b[:,:,:,q].mean(-1)-m for q in np.array_split(np.arange(16),4)],2);x=np.concatenate([(m-level[:,:,None]).reshape(992,40),b.std(-1).reshape(992,40),seg.reshape(992,160),level,d['amplitude']],1).astype('float32');assert x.shape==(992,246)
    stats=np.concatenate([m[:,:,None],b.std(-1)[:,:,None],seg+m[:,:,None]],2).reshape(992,240).astype('float32');selected=np.flatnonzero(((d['role']=='fit')&np.isin(d['text_index'],[0,15,30,45]))|(d['role']=='registration'));assert len(selected)==128
    archive={r['name']:r for r in json.loads((Path(a.audit)/'archive_central_private.json').read_text())};waves=[];lengths=[]
    for i in selected:
        r=records[int(i)];p=Path(a.root)/r['relative_wav'];raw=p.read_bytes();entry=archive['ABCS_database/'+r['relative_wav']];assert len(raw)==entry['size'] and zlib.crc32(raw)==entry['crc32']
        with wave.open(str(p),'rb') as f:
            assert (f.getframerate(),f.getnchannels(),f.getsampwidth())==(16000,2,2);pcm=f.readframes(f.getnframes())
        pair=np.frombuffer(pcm,dtype='<i2').reshape(-1,2);v=pair[:,1].astype('float32')/32768.;n=min(len(v),32000);start=max((len(v)-32000)//2,0);v=v[start:start+n].copy();v-=v.mean();v=np.pad(v,(0,32000-n));waves.append(v);lengths.append(n/32000)
    for role in ['fit','develop','registration','calibration','query']:
        ss=[set(d['text_index'][(d['user']==u)&(d['role']==role)]) for u in range(1,9)];assert all(s==ss[0] for s in ss)
    np.savez_compressed(out/'inputs_private.npz',x=x,statistics=stats,user=d['user'],role=d['role'],text_index=d['text_index'],sample_id=np.array([f'ABCS-row{i:04}' for i in range(992)]))
    np.savez_compressed(out/'audio_fit_register_private.npz',indices=selected,waves=np.stack(waves),lengths=np.array(lengths,dtype='float32'))
    summary=dict(state='prepared',identities=8,feature_records=992,teacher_audio_records=128,teacher_audio_fit_per_user=4,teacher_audio_registration_per_user=12,query_audio_records=0,query_audio_excluded=True,crc_checks=128,feature_dimension=246,shape_dimension=40,student_dimension=192,supplement_dimension=232,channels='ch0 provisional vibration,ch1 provisional Audio;not confirmed by official mapping',frontend='reuse whole-utterance ch0 50..7600Hz logpower40x16;ch1 teacher center<=2sec,padright ifshort,valid lengths supplied to frontend/ECAPA',short_teacher_clips=int(np.sum(np.array(lengths)<1)),roles=dict(fit=60,develop=15,registration=12,calibration=13,query=24),text_role_overlap=0,matched_real_text_across_users=True,privacy='anonymous row/user indices,no speaker names,transcripts,source paths or sample hashes in exported inputs',files={p.name:p.stat().st_size for p in [out/'inputs_private.npz',out/'audio_fit_register_private.npz']})
    (out/'preparation_public.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--audit',required=True);p.add_argument('--prior',required=True);p.add_argument('--out',required=True);main(p.parse_args())
