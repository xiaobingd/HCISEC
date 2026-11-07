"""Resource-serial replacement controls; never label missing assets as results."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path('/autodl-fs/data');BASE=ROOT/'mei8_strict_open_residual_20261108';PY=sys.executable
def status(**v):(ROOT/'strict_open_replacement_queue_status.json').write_text(json.dumps(v,indent=2))
def run(args):
    status(state='running',command=args)
    subprocess.run([PY]+args,check=True)
while True:
    p=BASE/'status.json';s=json.loads(p.read_text()) if p.exists() else {}
    if s.get('state')=='complete':break
    status(state='waiting_for_primary_24_folds',primary=s)
    time.sleep(30)
run(['/root/prepare_eres2netv2_control.py','--extract'])
run(['/root/run_strict_open_residual.py','--out',str(ROOT/'mei8_open_eres2netv2_20261108'),'--audio-cache',str(ROOT/'eres2netv2_control_20261108/audio_embedding_private.npy'),'--backbone-root',str(BASE)])
run(['/root/extract_vibravox_acc_transfer.py'])
for mode in ['voxceleb_init','vibravox_acc']:
    run(['/root/run_strict_open_residual.py','--out',str(ROOT/('mei8_open_'+mode+'_20261108')),'--acc-cache',str(ROOT/'vr_acc_pretrain_control_20261108'/(mode+'_embedding_private.npy'))])
status(state='all_replacement_controls_complete')
