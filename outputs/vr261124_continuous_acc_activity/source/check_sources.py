import argparse,hashlib,json
from pathlib import Path
def main(root):
    base=root/'continuous_acc_activity';records=json.loads((base/'recordings_private.json').read_text(encoding='utf-8'));byword={};digests={};paths=set();audio_paths={};split=json.loads((base/'word_split_public.json').read_text())
    for rec in records:
      byword.setdefault(rec['word'],set()).add(' '.join(rec['prompt'].casefold().split()))
      path=Path(rec['stem']).with_name(Path(rec['stem']).name+'_full_imu.csv').resolve();assert path not in paths;paths.add(path)
      stem=Path(rec['stem']);meta=json.loads(stem.with_name(stem.name+'_meta.json').read_text(encoding='utf-8-sig'));ap=(stem.parent/meta['full_audio']).resolve();audio_paths.setdefault(ap,set()).add(rec['word'])
      h=hashlib.sha256()
      with path.open('rb') as file:
        while True:
          chunk=file.read(1024*1024)
          if not chunk:break
          h.update(chunk)
      digests.setdefault(h.hexdigest(),[]).append((rec['user'],rec['word']))
    aliases=sum(len(v)!=1 for v in byword.values());duplicate=sum(len(v)-1 for v in digests.values())
    assert aliases==0 and duplicate==0 and len({next(iter(v)) for v in byword.values()})==40
    overlap=sum(bool(words&set(split['test'])) and bool(words&set(split['train']+split['development'])) for words in audio_paths.values());assert overlap==0
    result=dict(status='PASS',source_recordings=len(records),unique_IMU_files=len(paths),unique_IMU_file_hashes=len(digests),duplicated_source_files=duplicate,word_identity_consistent=True,distinct_prompt_labels=40,shared_test_and_training_Audio_paths=overlap,no_Audio_read=True)
    (base/'source_independence_public.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
