"""Run the same experiment on local CPU using existing official weights."""
import sys
from pathlib import Path

workspace=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(workspace/'work/audio_multiword_runtime'),str(workspace/'work/verifier_deps'),
              str(workspace/'outputs/vr261119_10x40_raw_acc')]
sys.path.append(str(workspace/'work/esmb_audio_deps'))
from types import SimpleNamespace
import torch
from torch import nn

def local_model(model_path):
    from speechbrain.lobes.features import Fbank
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.processing.features import InputNormalization
    embedding=ECAPA_TDNN(input_size=80,channels=[1024,1024,1024,1024,3072],
                         kernel_sizes=[5,3,3,3,1],dilations=[1,2,3,4,1],
                         attention_channels=128,lin_neurons=192)
    embedding.load_state_dict(torch.load(model_path/'embedding_model.ckpt',map_location='cpu',weights_only=True))
    mods=nn.ModuleDict(dict(embedding_model=embedding,compute_features=Fbank(n_mels=80),
                           mean_var_norm=InputNormalization(norm_type='sentence',std_norm=False))).eval()
    for p in mods.parameters():p.requires_grad_(False)
    return SimpleNamespace(mods=mods)

source=(workspace/'outputs/vr261119_10x40_raw_acc/train_audio_multiword_fitonly.py').read_text(encoding='utf-8')
source=source.replace("ROOT=Path('/root')",f'ROOT=Path({str(workspace/"outputs/vr261119_10x40_raw_acc")!r})')
source=source.replace("OUT=ROOT/'full_run_10x40_raw_acc/audio_multiword_fitonly'","OUT=ROOT/'audio_multiword_fitonly'")
source=source.replace("ROOT/'full_run_10x40_raw_acc/model'",f'Path({str(workspace/"outputs/ecapa_voxceleb_20261103/model")!r})')
old="        import run_ecapa_pilot as pilot\n        pilot.p.DEVICE='cpu'\n        model=pilot.teacher_load(ROOT/'full_run_10x40_raw_acc')"
assert source.count(old)==1
source=source.replace(old,"        model=local_model(original_model)")
source=source.replace("        with torch.inference_mode():\n            for start in range(completed,len(fi),8):",
                      "        torch.set_num_threads(8)\n        with torch.inference_mode():\n            for start in range(completed,len(fi),8):")
source=source.replace("        hook.remove();del model,emb,waves", "        hook.remove();del model,emb,waves\n        torch.set_num_threads(2)")
source=source.replace("ROOT/'full_run_10x40_raw_acc/strict_episodic_6x4'","ROOT/'strict_episodic_6x4'")
source=source.replace("ROOT/'full_run_10x40_raw_acc/threeword_expanded/results_public.json'","ROOT/'threeword_expanded/results_public.json'")
# Cross-platform floating point can differ; score/decision replay is also checked.
source=source.replace("assert err<1e-6,('CE_REPLAY_MISMATCH',seed,err)","assert err<1e-4,('CE_REPLAY_MISMATCH',seed,err)")
scope={'__name__':'__main__','local_model':local_model}
exec(compile(source,'local_audio_multiword_fitonly.py','exec'),scope)

