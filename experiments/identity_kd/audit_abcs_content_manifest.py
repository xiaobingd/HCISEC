"""Audit explicit corpus speaker/text mappings, never infer words from row order.

Input CSV/JSONL fields: pair_id,speaker_id,text,source_utterance_id,ac_path,
bc_path,speaker_evidence,text_evidence,pair_evidence. For stereo paths require
ac_channel/bc_channel. Evidence strings must refer to real source manifests;
their presence is not a substitute for semantic/acoustic spot checking.
"""
import argparse,csv,json,re,unicodedata,hashlib,wave
from collections import defaultdict,Counter
from pathlib import Path

REQUIRED=['pair_id','speaker_id','text','source_utterance_id','ac_path','bc_path','speaker_evidence','text_evidence','pair_evidence']
ROLES=['encoder_fit','encoder_dev','auth_fit','calibration','query']
def canonical(text):
    text=unicodedata.normalize('NFKC',str(text)).strip().casefold()
    # Preserve lexical tokens and numeric spelling; do not equate transcript IDs,
    # Arabic/Chinese numbers, homophones or ASR hypotheses with verified content.
    text=re.sub(r'\s+',' ',text)
    return text.strip(' 。！？!?，,；;')

def load(path):
    if path.suffix.lower()=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def main(a):
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True);records=load(Path(a.manifest));assert records,'empty source manifest'
    seen={};source_metadata={};counts=defaultdict(set);private=[];issues=[];headers={};payloads={};pair_payloads={};copied_pairs=0
    def header(path):
        key=str(path.resolve())
        if key not in headers:
            if path.suffix.lower()=='.wav':
                with wave.open(str(path),'rb') as f:headers[key]=(f.getframerate(),f.getnchannels(),f.getnframes())
            else:
                import soundfile
                f=soundfile.info(str(path));headers[key]=(f.samplerate,f.channels,f.frames)
        return headers[key]
    def payload(path):
        key=str(path.resolve())
        if key not in payloads:
            digest=hashlib.sha256()
            if path.suffix.lower()=='.wav':
                with wave.open(str(path),'rb') as f:
                    while True:
                        chunk=f.readframes(65536)
                        if not chunk:break
                        digest.update(chunk)
            else:
                import soundfile
                with soundfile.SoundFile(str(path)) as f:
                    while True:
                        chunk=f.read(65536,dtype='float32',always_2d=True)
                        if not len(chunk):break
                        digest.update(chunk.tobytes())
            payloads[key]=digest.hexdigest()
        return payloads[key]
    for row in records:
        missing=[k for k in REQUIRED if not str(row.get(k,'')).strip()]
        if missing:issues.append(dict(pair_id=row.get('pair_id'),error='missing explicit mapping/evidence',fields=missing));continue
        who=str(row['speaker_id']);pid=str(row['pair_id']);parent=str(row['source_utterance_id']);text=canonical(row['text'])
        if not text:issues.append(dict(pair_id=pid,error='empty normalized transcript'));continue
        if pid in seen:issues.append(dict(pair_id=pid,error='duplicate pair_id'));continue
        seen[pid]=True
        # Parent utterance IDs must be corpus-global and cannot be reused for another
        # identity/text. Multiple derived windows never count as new repetitions.
        if parent in source_metadata and source_metadata[parent]!=(who,text):issues.append(dict(pair_id=pid,error='source utterance has conflicting identity/text'));continue
        ac=Path(row['ac_path']);bc=Path(row['bc_path'])
        if not ac.is_absolute():ac=Path(a.data_root)/ac
        if not bc.is_absolute():bc=Path(a.data_root)/bc
        if not ac.is_file() or not bc.is_file():issues.append(dict(pair_id=pid,error='paired signal missing'));continue
        if ac.resolve()==bc.resolve():
            ca=str(row.get('ac_channel',''));cb=str(row.get('bc_channel',''))
            if not ca or not cb or ca==cb:issues.append(dict(pair_id=pid,error='stereo channel mapping absent or identical'));continue
        try:
            ah=header(ac);bh=header(bc)
            for path,h,field in [(ac,ah,'ac_channel'),(bc,bh,'bc_channel')]:
                channel=row.get(field,'')
                if not str(channel).strip():
                    if h[1]!=1:raise ValueError('multichannel signal needs explicit channel mapping')
                    channel=0
                if not 0<=int(channel)<h[1]:raise ValueError('channel outside file header')
            if ah[0]!=bh[0] or abs(ah[2]-bh[2])>max(1,int(.001*ah[0])):raise ValueError('paired header sample rates/lengths disagree')
        except Exception as error:issues.append(dict(pair_id=pid,error='audio header validation failed',detail=str(error)));continue
        ph=(payload(ac),payload(bc))
        if ph in pair_payloads:
            pu,pt,pp=pair_payloads[ph]
            if (pu,pt)!=(who,text):issues.append(dict(pair_id=pid,error='identical paired signal has conflicting identity/text'));continue
            if pp!=parent:copied_pairs+=1;parent=pp
        else:pair_payloads[ph]=(who,text,parent)
        source_metadata[parent]=(who,text)
        counts[text,who].add(parent);private.append(dict(row,normalized_text=text,canonical_source_utterance_id=parent))
    speakers=sorted({who for _,who in counts});texts=sorted({text for text,_ in counts});per_user={u:len({t for t,w in counts if w==u}) for u in speakers}
    text_users={t:{u for u in speakers if counts.get((t,u))} for t in texts}
    coverage={}
    for repeats in [1,2,3,5]:
        cover={t:{u for u in text_users[t] if len(counts[t,u])>=repeats} for t in texts}
        candidates={tuple(sorted(us)[:a.users]) for us in cover.values() if len(us)>=a.users}
        if len(speakers)>=a.users:candidates.add(tuple(speakers[:a.users]))
        best=();common=[]
        for cohort in sorted(candidates):
            matched=[t for t in texts if set(cohort)<=cover[t]]
            if len(matched)>len(common):best=cohort;common=matched
        coverage[str(repeats)]=dict(cohort=list(best),texts=common)
    chosen=coverage['5'];cohort=chosen['cohort'];shared=chosen['texts'];same_plan=[]
    for t in shared:
        for u in cohort:
            units=sorted(counts[t,u],key=lambda x:hashlib.sha256(('42:'+x).encode()).hexdigest())
            for role,unit in zip(ROLES,units[:5]):same_plan.append(dict(speaker_id=u,text=t,source_utterance_id=unit,role=role))
    # Content-disjoint plan groups by actual normalized transcript globally.
    ordered=sorted(texts,key=lambda t:hashlib.sha256(('42:'+t).encode()).hexdigest());fractions=[.5,.1,.15,.1,.15];assignment={};start=0
    for ix,(role,fraction) in enumerate(zip(ROLES,fractions)):
        end=len(ordered) if ix==len(ROLES)-1 else start+int(len(ordered)*fraction)
        for t in ordered[start:end]:assignment[t]=role
        start=end
    cross_plan=[dict(speaker_id=u,text=t,source_utterance_id=unit,role=assignment[t]) for (t,u),units in counts.items() for unit in units]
    for plan in [same_plan,cross_plan]:
        by_source=defaultdict(set)
        for row in plan:by_source[row['source_utterance_id']].add(row['role'])
        assert all(len(x)==1 for x in by_source.values()),'source utterance leaks across roles'
    roles_by_user={u:Counter(row['role'] for row in cross_plan if row['speaker_id']==u) for u in speakers}
    structurally_valid=not issues
    public=dict(state='structural_audit_complete_semantic_spotcheck_still_required',source_rows=len(records),valid_rows=len(private),speakers=len(speakers),unique_normalized_texts=len(texts),unique_source_utterances=len(source_metadata),copied_paired_payloads_not_counted_as_repeats=copied_pairs,mapping_issues=len(issues),requested_cohort_size=a.users,coverage={k:dict(best_found_common_text_count=len(x['texts']),cohort_size=len(x['cohort'])) for k,x in coverage.items()},same_content_attack_candidate=structurally_valid and len(coverage['1']['texts'])>=a.min_shared_texts,same_content_train_test_candidate=structurally_valid and len(coverage['2']['texts'])>=a.min_shared_texts,full_five_role_same_content_candidate=structurally_valid and len(shared)>=a.min_shared_texts,cross_content_candidate=structurally_valid and sum(all(roles_by_user[u][r]>=a.min_per_role for r in ROLES) for u in speakers)>=a.users,minimum_independent_utterances_per_role=a.min_per_role,semantic_verification='NOT implied by file names or evidence-field presence;check real transcripts and sampled Audio',notes=['content means exact normalized utterance/prompt,not disjoint vocabulary','different windows of one source utterance are not independent repetitions','text-based cohort coverage only,no authentication performance used','training gated on successful download/pairing/text checks;user permission already provided'])
    (out/'audit_public.json').write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,obj in [('coverage_private.json',coverage),('issues_private.json',issues),('speaker_counts_private.json',per_user),('same_content_split_private.json',same_plan),('cross_content_split_private.json',cross_plan)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'verified_mapping_private.jsonl').open('w',encoding='utf-8') as f:
        for row in private:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps(public,ensure_ascii=False,indent=2))
    return 0 if structurally_valid else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--data-root',required=True);p.add_argument('--out',required=True);p.add_argument('--users',type=int,default=8);p.add_argument('--min-shared-texts',type=int,default=20);p.add_argument('--min-per-role',type=int,default=5);raise SystemExit(main(p.parse_args()))