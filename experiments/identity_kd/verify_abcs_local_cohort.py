"""Verify archive CRC, PCM duplicates and channel bandwidth without assigning modalities."""
import argparse,json,hashlib,zlib,wave
from pathlib import Path
from collections import Counter
import numpy as np

def main(a):
    out=Path(a.audit);root=Path(a.root)
    coverage=json.loads((out/'available_text_coverage_private.json').read_text(encoding='utf-8'))['1']
    cohort=set(coverage['cohort']);texts=set(coverage['texts'])
    rows=json.loads((out/'transcript_mapping_private.json').read_text(encoding='utf-8'))
    archive={x['name']:x for x in json.loads((out/'archive_central_private.json').read_text())}
    selected=[r for r in rows if r['speaker_id'] in cohort and r['normalized_text'] in texts and r['present']]
    issues=[];seen={};duplicates=[];headers=Counter();ratios=[];total_seconds=0
    for r in selected:
        raw=(root/r['relative_wav']).read_bytes();entry=archive['ABCS_database/'+r['relative_wav']]
        if len(raw)!=entry['size'] or zlib.crc32(raw)!=entry['crc32']:
            issues.append({'pair_id':r['pair_id'],'error':'archive size/CRC mismatch'});continue
        try:
            with wave.open(str(root/r['relative_wav']),'rb') as f:
                sr=f.getframerate();ch=f.getnchannels();sw=f.getsampwidth();n=f.getnframes();pcm=f.readframes(n)
            headers[str((sr,ch,sw))]+=1;total_seconds+=n/sr
            h=hashlib.sha256(pcm).hexdigest()
            if h in seen:duplicates.append([seen[h],r['pair_id']])
            seen[h]=r['pair_id']
            if sw==2 and ch==2:
                x=np.frombuffer(pcm,dtype='<i2').reshape(-1,2).astype(np.float64)
                x-=x.mean(axis=0);power=np.abs(np.fft.rfft(x*np.hanning(len(x))[:,None],axis=0))**2
                freq=np.fft.rfftfreq(len(x),1/sr)
                ratio=power[(freq>=2000)&(freq<=7000)].sum(axis=0)/(power[(freq>=50)&(freq<2000)].sum(axis=0)+1e-12)
                ratios.append(ratio.tolist())
        except Exception as e:issues.append({'pair_id':r['pair_id'],'error':str(e)})
    result={'stage':'matched_text_cohort_integrity_not_authentication','cohort_size':len(cohort),'common_texts':len(texts),'selected_wavs':len(selected),'crc_or_decode_failures':len(issues),'exact_pcm_duplicate_pairs':len(duplicates),'duration_seconds':total_seconds,'headers':dict(headers),'median_high_to_low_power_by_channel':np.median(ratios,axis=0).tolist() if ratios else [],'channel_semantics':'unconfirmed; spectral contrast is diagnostic evidence only','warning':'common text across identities does not establish independent same-text repetitions within an identity'}
    (out/'cohort_integrity_public.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'cohort_integrity_private.json').write_text(json.dumps({'cohort':sorted(cohort),'issues':issues,'duplicates':duplicates}),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--audit',required=True);main(p.parse_args())
