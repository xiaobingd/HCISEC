"""Run two independent frozen Audio controls on local CPU, separate outputs."""
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

script=Path(__file__).with_name('run_p1_multimodal_enrollment.py')
def run(teacher):
    subprocess.run([sys.executable,str(script),'--bundle','outputs/p1_multimodal_bundle','--out','outputs/p1_enrollment_'+teacher+'_20261108','--teacher',teacher],check=True)
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run,['ecapa','eres']))
print('BOTH_AUDIO_MODELS_P1_COMPLETE',flush=True)
