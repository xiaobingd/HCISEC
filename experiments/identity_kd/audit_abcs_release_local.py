"""Inspect actual official transcript IDs against split-ZIP directory and local WAVs."""
import argparse,json,re,struct,os,wave,unicodedata
from pathlib import Path
from collections import Counter,defaultdict
from audit_abcs_content_manifest import canonical

def main(a):
    root=Path(a.root);out=Path(a.out);out.mkdir(exist_ok=True,parents=True);archive=json.loads(Path(a.central).read_text());expected={x['name'].replace('\\','/'):x for x in archive if x['name'].lower().endswith('.wav')}
    present={str(Path(folder,name).relative_to(root)).replace('\\','/') for folder,_,names in os.walk(root/'Audio') for name in names if name.lower().endswith('.wav')}
    expected_by=defaultdict(set);present_by=defaultdict(set)
    for path in expected:
        rel=path.split('ABCS_database/',1)[-1];speaker=Path(rel).parent.name;expected_by[speaker].add(rel)
    for path in present:present_by[Path(path).parent.name].add(path)
    complete=sorted([s for s,files in expected_by.items() if files<=present_by[s]],key=lambda s:int(s.replace('Speaker','')))
    rows=[];problems=[];seen={};by_text=defaultdict(lambda:defaultdict(set));available_text=defaultdict(lambda:defaultdict(set));ids_by_speaker=defaultdict(set)
    for file in sorted((root/'script').rglob('*.txt')):
        raw=file.read_bytes()
        try:content=raw.decode('utf-8-sig');encoding='utf-8-sig'
        except UnicodeError:content=raw.decode('gb18030');encoding='gb18030'
        for line_no,line in enumerate(content.splitlines(),1):
            if not line.strip():continue
            fields=line.strip().split(maxsplit=1)
            if len(fields)!=2:problems.append(dict(file=str(file),line=line_no,error='no explicit ID/text pair'));continue
            uid,text=fields;match=re.fullmatch(r'(Speaker\d+)_([CD])_(?:(\d+)_)?(\d+)',uid)
            if not match or not uid.startswith(file.stem+'_'):problems.append(dict(file=str(file),line=line_no,error='transcript ID conflicts with official speaker file'));continue
            speaker,kind,segment,index=match.groups();normalized=canonical(text)
            if uid in seen:problems.append(dict(pair_id=uid,error='duplicate utterance ID'));continue
            seen[uid]=True;rel='Audio/'+file.parent.name+'/'+speaker+'/'+uid+'.wav';key='ABCS_database/'+rel
            if key not in expected:problems.append(dict(pair_id=uid,error='transcript has no archive WAV'))
            row=dict(pair_id=uid,speaker_id=speaker,subset=file.parent.name,kind=kind,raw_text=text,normalized_text=normalized,relative_wav=rel,present=rel in present,text_source=str(file),line=line_no,encoding=encoding)
            rows.append(row);ids_by_speaker[speaker].add(uid);by_text[normalized][speaker].add(uid)
            if row['present']:available_text[normalized][speaker].add(uid)
    def coverage(table):
        result={}
        for repeats in [1,2,3,5]:
            cover={t:{s for s,units in speakers.items() if len(units)>=repeats} for t,speakers in table.items()}
            candidates={tuple(sorted(s,key=lambda x:int(x.replace('Speaker','')))[:a.users]) for s in cover.values() if len(s)>=a.users};best=();matched=[]
            for cohort in sorted(candidates):
                common=[t for t,speakers in cover.items() if set(cohort)<=speakers]
                if len(common)>len(matched):best=cohort;matched=common
            result[str(repeats)]=dict(cohort=list(best),common_text_count=len(matched),texts=matched)
        return result
    full=coverage(by_text);available=coverage(available_text)
    examples={}
    for row in rows:
        if row['present'] and row['speaker_id'] not in examples:
            try:
                with wave.open(str(root/row['relative_wav']),'rb') as f:examples[row['speaker_id']]=dict(sample_rate=f.getframerate(),channels=f.getnchannels(),sample_width=f.getsampwidth(),frames=f.getnframes())
            except Exception as e:problems.append(dict(pair_id=row['pair_id'],error='sample WAV header unreadable',detail=str(e)))
    public=dict(stage='release_transcript_and_archive_directory_audit_not_training',source_files_complete=(root.parent/'ABCS_database_ciaic.z01').exists(),expected_archive_wavs=len(expected),local_wavs=len(present),missing_wavs=len(set(expected_key.split('ABCS_database/',1)[-1] for expected_key in expected)-present),archive_speakers=len(expected_by),transcript_speakers=len(ids_by_speaker),local_complete_speakers=len(complete),transcript_rows=len(rows),mapping_issues=len(problems),kind_counts=dict(Counter(x['kind'] for x in rows)),header_examples=dict(Counter(str((x['sample_rate'],x['channels'],x['sample_width'])) for x in examples.values())),requested_cohort_size=a.users,all_transcript_coverage={k:dict(common_text_count=x['common_text_count'],cohort_size=len(x['cohort'])) for k,x in full.items()},available_audio_transcript_coverage={k:dict(common_text_count=x['common_text_count'],cohort_size=len(x['cohort'])) for k,x in available.items()},limitations=['transcript coverage is not raw-payload duplicate verification','channel semantics not yet independently confirmed','same text does not imply same independent repeated recording','actual utterance content requires sampled listening/ASR checks','candidate cohort search based only on transcript coverage,not exhaustive optimum'])
    (out/'release_audit_public.json').write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,obj in [('transcript_mapping_private.json',rows),('release_issues_private.json',problems),('full_text_coverage_private.json',full),('available_text_coverage_private.json',available),('complete_speakers_private.json',complete)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(public,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--central',required=True);p.add_argument('--out',required=True);p.add_argument('--users',type=int,default=8);main(p.parse_args())
