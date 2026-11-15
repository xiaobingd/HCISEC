"""Run the unchanged Mei training or verifier on available CUDA, otherwise CPU."""
import sys
from pathlib import Path
import torch

script=Path(sys.argv.pop(1))
sys.argv[0]=str(script)
device='cuda' if torch.cuda.is_available() else 'cpu'
source=script.read_text(encoding='utf-8')
if device=='cpu':
    source=source.replace('.cuda()', ".to('cpu')")
    source=source.replace("device='cuda'", "device='cpu'")
    source=source.replace("map_location='cuda'", "map_location='cpu'")
    # Preserve the original identity batch/loss while recomputing ECAPA activations.
    old="def forward(j):return F.normalize(emb(feat[j],lens[j]).squeeze(1),dim=1)"
    new="""def forward(j):
        from torch.utils.checkpoint import checkpoint
        chunks=[]
        for start in range(0,len(j),1):
            jj=j[start:start+1]
            fn=lambda xx,ll:F.normalize(emb(xx,ll).squeeze(1),dim=1)
            chunks.append(checkpoint(fn,feat[jj].detach().requires_grad_(True),lens[jj],use_reentrant=True) if torch.is_grad_enabled() else fn(feat[jj],lens[jj]))
        return torch.cat(chunks)
"""
    source=source.replace(old,new)
    source=source.replace('range(0,len(wave),16)', 'range(0,len(wave),1)')
    source=source.replace('wave[s:s+16]', 'wave[s:s+1]')
    if script.name=='verify_mei9_closed_personal_cycle.py':
        import types
        module_path=script.with_name('mei9_closed_personal_cycle.py')
        imported=module_path.read_text(encoding='utf8').replace('.cuda()', ".to('cpu')").replace("device='cuda'", "device='cpu'").replace("map_location='cuda'", "map_location='cpu'")
        module=types.ModuleType('mei9_closed_personal_cycle')
        module.__file__=str(module_path)
        sys.modules[module.__name__]=module
        exec(compile(imported,str(module_path),'exec'),module.__dict__)
print('EXECUTION_DEVICE',device,flush=True)
exec(compile(source,str(script),'exec'),{'__name__':'__main__','__file__':str(script)})
