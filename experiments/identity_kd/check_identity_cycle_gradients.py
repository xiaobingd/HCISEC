"""Synthetic implementation check; these are not authentication results."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from identity_discriminative_cycle import train
torch.set_num_threads(2)
rng=np.random.RandomState(7);d=dict(user=np.repeat(np.arange(1,10),4),word=np.tile([0,0,1,1],9));data=dict(a=rng.uniform(-1,1,(36,3,40,32)).astype('float32'),b=rng.uniform(-1,1,(36,1,40,32)).astype('float32'));ix=dict(fit=np.arange(36),dev=np.arange(4));a=SimpleNamespace(device='cpu',steps=2,supcon='standard')
with tempfile.TemporaryDirectory() as folder:
 for mode in ['E2','E3','E4']:
  train(mode,42,d,data,ix,Path(folder),a)
  ck=torch.load(Path(folder)/'seed42'/mode/'checkpoint_private.pt',map_location='cpu',weights_only=False)
  from mei9_closed_personal_cycle import Generator
  torch.manual_seed(42);initial=Generator(3,1)
  delta=max(float((v-ck['G'][k]).abs().max()) for k,v in initial.state_dict().items() if k.startswith('encoder.'))
  assert delta>0
  print('PASS',mode,'encoder_parameter_delta',delta,flush=True)
