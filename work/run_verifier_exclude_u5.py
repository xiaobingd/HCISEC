"""Run the training-exclusion experiment, verify it and generate its report."""
import subprocess
import sys

root='outputs/mei_neural_verifier_exclude_u5_20261116'
common=[sys.executable,'-u']
subprocess.run(common+['experiments/identity_kd/run_crossword_neural_verifier.py',
 '--frontend','outputs/mei_common_frontend_private.npz',
 '--manifest','outputs/mei_raw9_manifest_private.json','--out',root,
 '--encoder-steps','400','--steps','1000','--exclude-users','5',
 '--teacher-cache','outputs/mei_neural_verifier_20261116/redimnet2_train_embeddings_private.npz',
 '--methods','global','local','audio','shuffle'],check=True)
subprocess.run(common+['experiments/identity_kd/verify_crossword_neural_verifier.py',
 '--frontend','outputs/mei_common_frontend_private.npz','--out',root],check=True)
subprocess.run(common+['experiments/identity_kd/report_crossword_neural_verifier.py','--out',root],check=True)
print('EXCLUDE_U5_COMPLETE_AND_VERIFIED',flush=True)
