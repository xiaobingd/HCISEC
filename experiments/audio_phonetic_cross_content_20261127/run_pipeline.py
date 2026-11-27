"""Resume at immutable checkpoints; never replace a completed historical study."""
import json,subprocess,sys,time
from pathlib import Path
from prepare_acc import PRIVATE,OUT,ROOT

def main():
    base=Path(__file__).resolve().parent;waitstart=time.time()
    while not (OUT/'teacher_audit_public.json').exists() or not (PRIVATE/'development_42_T2_private.pt').exists():
        if time.time()-waitstart>10800:raise TimeoutError('Teacher or first development arms not completed after3h')
        print('WAITING_FOR_TEACHER',len(list((PRIVATE/'teacher_records').glob('*.npz'))),'of160',flush=True);time.sleep(30)
    audit=json.loads((OUT/'teacher_audit_public.json').read_text(encoding='utf-8'));assert audit['status']=='PASS' and audit['query_Audio_reads']==0
    commands=[['train.py','--stage','development','--threads','4'],['development.py'],
      ['train.py','--stage','final','--threads','4'],['evaluate.py','--phase','calibrate'],
      ['evaluate.py','--phase','query'],['verify.py'],['report.py']]
    log=PRIVATE/'pipeline_log_private.txt'
    with log.open('a',encoding='utf-8') as stream:
      for args in commands:
        print('PIPELINE_COMMAND',*args,flush=True);stream.write('\nPIPELINE_COMMAND '+' '.join(args)+'\n');stream.flush()
        subprocess.run([sys.executable,str(base/args[0]),*args[1:]],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print('PIPELINE_COMPLETE_ALL15_MODELS',flush=True)

if __name__=='__main__':main()
