"""Full-record frozen Audio CTC posterior, mapped to IMU-origin training windows."""
import argparse,json,io,hashlib,time,platform,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
import torch
from threadpoolctl import threadpool_limits
from runtime import lexicon,teacher_model,MODEL,REVISION,WEIGHT_SHA
from prepare_acc import BASE,PRIVATE,OUT,OPT,DEV,ROOT
from common import ctc_posterior,smooth_to_acc,template
from clock_mapping import clock_regions,map_receptive_centers,regional_targets,POLICY

def greedy_collapse(logprob,blank=0):
    ids=logprob.argmax(1);return [int(v) for i,v in enumerate(ids) if v!=blank and (i==0 or v!=ids[i-1])]
def edit(a,b):
    row=np.arange(len(b)+1)
    for i,x in enumerate(a):
        nxt=np.zeros(len(b)+1,dtype=int);nxt[0]=i+1
        for j,y in enumerate(b):nxt[j+1]=min(nxt[j]+1,row[j+1]+1,row[j]+(x!=y))
        row=nxt
    return int(row[-1])

def main(limit=None):
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);cache=PRIVATE/'teacher_records';cache.mkdir(exist_ok=True)
    if limit is None and (PRIVATE/'teacher_private.npz').exists() and (OUT/'teacher_audit_public.json').exists():
        completed=json.loads((OUT/'teacher_audit_public.json').read_text(encoding='utf-8'))
        assert completed['status']=='PASS' and completed['clock_policy']==POLICY
        assert hashlib.sha256((PRIVATE/'teacher_private.npz').read_bytes()).hexdigest()==completed['teacher_cache_sha256']
        print('IMMUTABLE_COMPLETED_TEACHER_CACHE_REUSED',flush=True);return
    records=json.loads((PRIVATE/'recordings_private.json').read_text(encoding='utf-8'))
    selected=[r for r in records if r['word'] in OPT+DEV];assert len(selected)==160
    vocab=json.loads((OUT/'role_split_public.json').read_text(encoding='utf-8'))['vocabulary']
    lex,tokeninfo=lexicon({wo:vocab[str(wo)] for wo in OPT+DEV});assert any(v['ids'] for v in lex.values())
    (OUT/'lexicon_public.json').write_text(json.dumps(dict(lexicon=lex,tokenizer=tokeninfo),ensure_ascii=False,indent=2),encoding='utf-8')
    data=np.load(PRIVATE/'acc_private.npz');u,w=data['user'],data['word'];teacheridx=np.flatnonzero(np.isin(w,OPT+DEV));inverse=np.full(len(u),-1,dtype=int);inverse[teacheridx]=np.arange(len(teacheridx))
    targets=np.zeros((len(teacheridx),146,392),dtype='float32');targets[:,:,0]=1;valid=np.zeros((len(teacheridx),146),bool)
    framehost=data['frame_times'];model=None;extractor=None;modelinfo=None;audit=[];Audio_read=[];clock_read=[]
    startall=time.perf_counter()
    config=json.loads((MODEL/'config.json').read_text());rf,stride=1,1
    for kernel,step in zip(config['conv_kernel'],config['conv_stride']):rf+=(kernel-1)*stride;stride*=step
    for counter,rec in enumerate(selected[:limit] if limit else selected):
      user,word=int(rec['user']),int(rec['word']);assert word in OPT+DEV;stem=Path(rec['stem']);path=cache/f'u{user:02}_w{word:02}_private.npz'
      row=dict(user=user,word=word,role='optimize' if word in OPT else 'development',Audio_opened=False,Audio_timestamp_opened=False,cache_reused=False)
      if path.exists():
        with np.load(path) as c:payload={key:c[key] for key in c.files}
        assert str(payload['weight_sha'])==WEIGHT_SHA and np.array_equal(payload['word_tokens'],lex[word]['ids'])
        q=payload['posterior'];row.update(json.loads(str(payload['audit_json'])));row.update(cache_reused=True,Audio_opened=False,Audio_timestamp_opened=False)
        end,host=payload['clock_frame_end'],payload['clock_host'];nominal,segments,clockaudit=clock_regions(end,host)
        assert nominal==row['resampling_nominal_rate'],'Changed nominal requires fresh Audio inference,not reuse'
      else:
        meta=json.loads(stem.with_name(stem.name+'_meta.json').read_text(encoding='utf-8-sig'))
        assert meta['prompt_text'].strip().lower()==vocab[str(word)].strip().lower()
        clock_path=stem.with_name(stem.name+'_audio_timestamps.csv');clock=pd.read_csv(clock_path)
        end=(clock.frame_offset+clock.frames).to_numpy(dtype='float64');host=clock.host_callback_perf_counter_s.to_numpy(dtype='float64');good=np.isfinite(end)&np.isfinite(host)
        end,host=end[good],host[good];assert len(end)>3
        nominal,segments,clockaudit=clock_regions(end,host);clock_read.append((user,word));row['Audio_timestamp_opened']=True
        audio_path=stem.parent/meta['full_audio'];source=audio_path.read_bytes();Audio_read.append((user,word));header,pcm=wavfile.read(io.BytesIO(source));row['Audio_opened']=True
        assert pcm.dtype==np.int16;pcm=pcm.astype('float32')/32768
        channels=1 if pcm.ndim==1 else pcm.shape[1]
        if pcm.ndim==2:pcm=pcm[:,0]
        gcd=np.gcd(nominal,16000);wave=signal.resample_poly(pcm,16000//gcd,nominal//gcd).astype('float32')
        repeat_count=len(meta.get('segments',[]));tokens=lex[word]['ids']*repeat_count
        if not tokens:raise ValueError('No compatible known transcript phones/repeats')
        if model is None:
          print('LOADING_FROZEN_AUDIO_TEACHER',flush=True);model,extractor,modelinfo=teacher_model()
          versions={k:importlib.metadata.version(k) for k in ['torch','transformers','tokenizers','espeakng-loader']}
          (OUT/'teacher_model_public.json').write_text(json.dumps(dict(**modelinfo,tokenizer=tokeninfo,software=versions,platform=platform.platform()),ensure_ascii=False,indent=2),encoding='utf-8')
        before=time.perf_counter();inputs=extractor(wave,sampling_rate=16000,return_tensors='pt')
        with torch.inference_mode():logprob=model(**inputs).logits[0].log_softmax(-1).cpu().numpy()
        assert np.isfinite(logprob).all()
        q,logZ=ctc_posterior(logprob,tokens,blank=0)
        assert rf==modelinfo['receptive_field_samples'] and stride==modelinfo['stride_samples']
        time_s=((rf-1)/2+np.arange(len(q))*stride)/16000
        assert (len(q)-1)*stride+rf<=len(wave)
        row.update(model_inference_and_alignment_s=float(time.perf_counter()-before),header_rate=int(header),audio_channels=channels,full_pcm_frames=len(pcm),ctc_frames=len(q),repeats=repeat_count,transcript_tokens=len(tokens),
          log_ctc_alignment_likelihood=logZ,greedy_phone_error_rate=edit(greedy_collapse(logprob),tokens)/len(tokens),posterior_blank_mean=float(q[:,0].mean()),
          posterior_entropy_mean=float(-(q*np.log(np.maximum(q,1e-20))).sum(1).mean()),source_sha256=hashlib.sha256(source).hexdigest())
        # Dense raw logits retained only for three fixed replay cases, avoiding redundant disk use.
        replay={'logprob':logprob} if (user,word) in [(1,0),(2,16),(10,34)] else {}
        payload=dict(posterior=q,tokens=np.array(tokens),word_tokens=np.array(lex[word]['ids']),weight_sha=np.array(WEIGHT_SHA),clock_frame_end=end,clock_host=host,**replay)
      centers=((rf-1)/2+np.arange(len(q))*stride)/16000*nominal
      audio_host,regions=map_receptive_centers(centers,(rf-1)/2/16000*nominal,segments)
      row.update(**clockaudit,clock_invalid_ctc_frames=int(np.isnan(audio_host).sum()))
      payload.update(audio_host=audio_host,audio_regions=regions,audit_json=np.array(json.dumps(row)))
      np.savez_compressed(path,**payload)
      pos=np.flatnonzero((u==user)&(w==word));assert len(pos)>=2
      counts=0
      for i in pos:
        pp,mask=regional_targets(audio_host,q,framehost[i],regions,.1);mask&=data['valid'][i]
        targets[inverse[i]]=pp;valid[inverse[i]]=mask;counts+=int(mask.sum())
      row.update(mapped_windows=len(pos),valid_teacher_frames=counts);audit.append(row)
      print('AUDIO_TEACHER_RECORD',counter+1,'of',len(selected),'u',user,'w',word,'frames',counts,flush=True)
    if limit and limit<160:
      (OUT/'teacher_smoke_public.json').write_text(json.dumps(dict(status='SMOKE_ONLY',records=audit,model_info_available=(OUT/'teacher_model_public.json').exists()),indent=2));return
    assert len(audit)==160 and np.isfinite(targets).all() and not any(wo not in OPT+DEV for _,wo in Audio_read)
    assert len(set(row['source_sha256'] for row in audit))==160
    assert np.allclose(targets.sum(2),1,atol=1e-5) and (inverse[~np.isin(w,OPT+DEV)]==-1).all()
    dictionary=[];dictionary_valid=[]
    for word in OPT+DEV:
        qq,mm=template(lex[word]['ids'],392);dictionary.append(qq);dictionary_valid.append(mm)
    np.savez_compressed(PRIVATE/'teacher_private.npz',posterior=targets,valid=valid,indices=teacheridx,dictionary=np.array(dictionary),dictionary_valid=np.array(dictionary_valid),dictionary_words=np.array(OPT+DEV),blank_id=np.array(0))
    report=dict(status='PASS',actual_windows=len(teacheridx),records=audit,allowed_word_ids=OPT+DEV,teacher_recordings=160,Audio_recordings_read_this_run=len(Audio_read),Audio_clockfiles_read_this_run=len(clock_read),
      forbidden_role_Audio_reads=0,query_Audio_reads=0,query_Audio_clock_reads=0,teacher_shape=list(targets.shape),valid_frames=int(valid.sum()),windows_without_auxiliary=int((~valid.any(1)).sum()),
      nominal_resampling_rate_counts={str(rate):sum(row['resampling_nominal_rate']==rate for row in audit) for rate in [44100,48000]},
      model_revision=REVISION,weight_sha256=WEIGHT_SHA,frame_mapping='clock-only100callback-block median rate;75ms callbackexcess splits;validatedpiecewise affinePCMendframe;RFboundary masks;regionwiseGaussian100ms;no lagsearch',clock_policy=POLICY,
      blank_is_silence_claim=False,physical_sync_verified=False,elapsed_s=time.perf_counter()-startall,
      teacher_cache_sha256=hashlib.sha256((PRIVATE/'teacher_private.npz').read_bytes()).hexdigest())
    (OUT/'teacher_audit_public.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print('TEACHER_EXTRACTION_COMPLETE',report['teacher_shape'],flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int)
    with threadpool_limits(limits=4):main(parser.parse_args().limit)
