"""Run the user-prioritized full backbone adaptation, then resume controls."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/autodl-fs/data');state=ROOT/'vibravox_acc_finetune_queue_status.json'
try:
    state.write_text(json.dumps(dict(state='running_full_backbone_finetuning',priority='user_requested_immediate_start')))
    subprocess.run([sys.executable,'/root/finetune_vibravox_acc_open.py'],check=True)
    state.write_text(json.dumps(dict(state='finetuning_complete_resuming_frozen_controls')))
    subprocess.run([sys.executable,'/root/queue_strict_open_replacements.py'],check=True)
    state.write_text(json.dumps(dict(state='all_complete')))
except Exception as exc:
    state.write_text(json.dumps(dict(state='failed',error=str(exc))))
    raise
