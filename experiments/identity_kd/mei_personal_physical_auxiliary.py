"""Adapt the matched personal identity protocol to explicit physical Audio targets.
Only the auxiliary loss changes; backend, calibration, split and budget are reused.
"""
from pathlib import Path

source=Path(__file__).with_name('mei_personal_audio_guidance.py').read_text(encoding='utf8')
source=source.replace('mei_personal_audio_guidance_20261114','mei_personal_physical_auxiliary_20261114')
source=source.replace("['raw','identity','uniform','personal']", "['raw','identity','all_physical','selected_physical']")
source=source.replace("for u in range(1,10):\n owner=", """physical=dict(np.load('outputs/mei_physical_teacher_diagnostic_20261114/features_private.npz'))
pd=np.load('outputs/mei_physical_train_private.npz')
assert np.array_equal(pd['indices'],ti)
from sklearn.preprocessing import StandardScaler
diagnostic=json.loads(Path('outputs/mei_physical_teacher_diagnostic_20261114/per_user_public.json').read_text())
candidate={u:[v['family'] for v in diagnostic if v['user']==f'U{u}' and v['candidate']] for u in range(1,10)}
for u in range(1,10):
 owner=""")
start=source.index('   # Every owner')
end=source.index('   for step in range(200):',start)
source=source[:start]+"""   # A separate head per family; fit target scalers only to this owner's legal records.
   families=[] if mode=='identity' else list(physical) if mode=='all_physical' else candidate[u]
   heads=nn.ModuleDict({k:nn.Linear(64,physical[k].shape[1]) for k in physical})
   opt=torch.optim.AdamW(list(net.parameters())+list(head.parameters())+list(heads.parameters()),lr=1e-4,weight_decay=1e-3)
   targets={k:torch.tensor(StandardScaler().fit_transform(physical[k][owner]),dtype=torch.float32) for k in physical}
   pp=torch.tensor(pos);nnn=torch.tensor(neg);own=torch.tensor(owner)
"""+source[end:]
source=source.replace("    if mode!='identity':loss=loss+.1*((s-tm).square()*w).sum()/w.sum().clamp_min(1)", """    if families:
     auxiliary=torch.stack([F.mse_loss(heads[k](z[own]),targets[k]) for k in families]).mean()
     loss=loss+.1*auxiliary""")
source=source.replace("seed;archived teacher+shared identity42 initialization;owner-specific relation weights,not physical-target feature selection;CalA/B owner fusion;no query tuning", "seed;shared identity42 initialization;explicit physical auxiliary targets,owner-only16Audio;training-only eight-word diagnostic gate reused;CalA/B personal fusion;no query tuning")
source=source.replace("scope='known9", "scope='known9")
source=source.replace("print(summary,flush=True)", "(out/'teacher_targets_public.json').write_text(json.dumps({str(k):v for k,v in candidate.items()},indent=2));print(summary,flush=True)")
exec(compile(source,str(Path(__file__))+'::matched_protocol','exec'))
