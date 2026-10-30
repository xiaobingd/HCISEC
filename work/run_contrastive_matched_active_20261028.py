"""Matched-capacity dual-head and early-active semi-hard ablation."""
import inspect,sys
from pathlib import Path
sys.path.insert(0,'/root')
import run_contrastive_head_methods_20261028 as e
OriginalHead=e.Head
class MatchedHead(OriginalHead):
 def __init__(self,variant):
  super().__init__('factorized64_16' if variant=='factorized_control64_16' else variant)
e.Head=MatchedHead
source=inspect.getsource(e.fit_projection)
source=source.replace('strength=max(0.,min(1.,(epoch-20)/80.))','strength=max(0.,min(1.,(epoch-2)/8.))')
source=source.replace('loss=loss+.2*F.cross_entropy(model.word_classifier(content),words)+covariance.square().sum()',"aux_weight=0. if variant=='factorized_control64_16' else 1.\n   loss=loss+aux_weight*(.2*F.cross_entropy(model.word_classifier(content),words)+covariance.square().sum())")
exec(compile(source,'matched_projection_fit','exec'),e.__dict__)
e.VARIANTS=('compact64','compact128','semi_hard64','factorized_control64_16','factorized64_16')
e.EPOCHS=(5,10,20,50)
e.OUT=Path('/autodl-fs/data/contrastive_matched_active_20261028')
if __name__=='__main__':e.main()
