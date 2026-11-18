"""Feature-origin audit: preserve enrollment feature mean for cosine prototypes.
All target users use identical scale-only feature preprocessing.
Keep centered run intact; never pick preprocessing per user on final labels.
"""
import sys,os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,'/root')
import run_pira_stage1_20261028 as s
old=s.ROOT
s.ROOT=Path('/autodl-fs/data/pira_stage1_scaleonly_20261028');s.ROOT.mkdir(parents=True,exist_ok=True)
class ScaleOnly(s.Transform):
 def fit(self,blocks):self.scalers=[StandardScaler(with_mean=False).fit(b) for b in blocks];return self
s.Transform=ScaleOnly
original_js=s.js
def audited_js(path,value):
 if path.name=="locked_protocol.json":value=dict(value,feature_origin="Scale-only; preserve mean for Cosine. Centered historical run retained separately.")
 return original_js(path,value)
s.js=audited_js
for u in s.TARGETS:
 source=old/('U%d'%u);target=s.ROOT/('U%d'%u);target.mkdir(exist_ok=True)
 for name in ('feature_cache.npz','manifest.json'):
  src=source/name;dst=target/name
  if not src.exists():raise RuntimeError('First run must finish features first: '+str(src))
  if not dst.exists():os.link(str(src),str(dst))
if __name__=='__main__':s.main()
