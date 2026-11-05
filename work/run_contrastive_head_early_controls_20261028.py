"""Early-stopping and random-projection controls prompted by TRAIN CV only."""
import inspect,sys
from pathlib import Path
sys.path.insert(0,'/root')
import run_contrastive_head_methods_20261028 as e
OriginalHead=e.Head
class ExtendedHead(OriginalHead):
 def __init__(self,variant):
  super().__init__('compact64' if variant=='mlp64' else variant)
  if variant=='mlp64':
   self.identity=e.nn.Sequential(e.nn.Linear(772,128),e.nn.LayerNorm(128),e.nn.SiLU(),e.nn.Dropout(.1),e.nn.Linear(128,64))
e.Head=ExtendedHead
text=inspect.getsource(e.fit_projection)
assert 'states={};losses=[]' in text
text=text.replace('states={};losses=[]',"states={};losses=[]\n if 0 in collect:states[0]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}")
text=text.replace("'first_loss':losses[0],'last_loss':losses[-1]", "'first_loss':losses[0] if losses else None,'last_loss':losses[-1] if losses else None")
exec(compile(text,'early_projection_fit','exec'),e.__dict__)
e.VARIANTS=('compact64','compact128','crossword64','semi_hard64','factorized64_16','mlp64')
e.EPOCHS=(0,5,10,20,50)
e.OUT=Path('/autodl-fs/data/contrastive_head_early_controls_20261028')
e.main()
