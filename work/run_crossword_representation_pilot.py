"""Sequential training then fixed-query authentication; no daemon/automation."""
from pathlib import Path
import subprocess
import sys
root='outputs/mei_crossword_representation_20261116'
command=[sys.executable,'-u']
subprocess.run(command+['experiments/identity_kd/run_crossword_encoder_distill.py',
    '--frontend','outputs/mei_common_frontend_private.npz',
    '--teacher','outputs/mei_neural_verifier_20261116/redimnet2_train_embeddings_private.npz',
    '--out',root,'--seed','42','--warmup','400','--steps','1000'],check=True)
subprocess.run(command+['experiments/identity_kd/evaluate_crossword_encoder_distill.py','--out',root],check=True)
print('PILOT_COMPLETE',flush=True)
