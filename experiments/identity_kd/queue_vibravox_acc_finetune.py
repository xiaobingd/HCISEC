"""Append full-backbone adaptation without disrupting the running queue."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/autodl-fs/data');path=ROOT/'vibravox_acc_finetune_queue_status.json'
while True:
    statepath=ROOT/'strict_open_replacement_queue_status.json'
    try:state=json.loads(statepath.read_text()) if statepath.exists() else {}
    except json.JSONDecodeError:
        time.sleep(2)
        continue
    if state.get('state')=='all_replacement_controls_complete':break
    path.write_text(json.dumps(dict(state='waiting_for_parent_recovery' if state.get('state')=='failed' else 'waiting_for_frozen_controls',parent=state),indent=2))
    time.sleep(30)
try:
    path.write_text(json.dumps(dict(state='running_full_backbone_finetuning')))
    subprocess.run([sys.executable,'/root/finetune_vibravox_acc_open.py'],check=True)
    path.write_text(json.dumps(dict(state='complete')))
except Exception as exc:
    path.write_text(json.dumps(dict(state='failed',error=str(exc))))
    raise
