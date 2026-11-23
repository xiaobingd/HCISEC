import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

root=Path('outputs/vr261119_10x40_raw_acc').resolve()
target=root/'strict_episodic_6x4';target.mkdir(exist_ok=True)
with zipfile.ZipFile('work/audio_multiword_reference_private.zip') as z:
    for name in z.namelist():assert (target/name).resolve().is_relative_to(target)
    z.extractall(target)
expected={'paired_raw_acc_private.npz':'fd79b970d694d54a944c7bd61551e737c46dbb3ad1eb10d580acf4e19b1fa5c5',
          'stats27_private.npz':'44c7487febd99fe048a6190d4d38031d24cbe06e7b1e0a7bbe91f6d5dbe4dddf',
          'adapted_acc_private.pt':'267361c54a15867b0c1df1bdf5c55a3fa192ad72357657046c3ed119716a4ac4'}
for name,digest in expected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
ref=np.load(target/'official_probe_private.npz')
local=root/'audio_multiword_fitonly'
pool=np.load(local/'official_pool_private.npy',mmap_mode='r')[:1]
embedding=np.load(local/'official_embedding_private.npy',mmap_mode='r')[:1]
np.testing.assert_allclose(pool,ref['pool'],rtol=1e-4,atol=1e-3)
np.testing.assert_allclose(embedding,ref['embedding'],rtol=1e-4,atol=1e-4)
result=dict(status='PASS',source_dataset_stats_initialization_hashes_match_remote=True,
            reference_checkpoints=3,teacher_probe_rows=1,
            teacher_embedding_probe_max_abs=float(np.max(np.abs(embedding-ref['embedding']))),
            teacher_pool_probe_max_abs=float(np.max(np.abs(pool-ref['pool']))))
(local/'execution_verification_public.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))

