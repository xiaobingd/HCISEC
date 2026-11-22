"""No-Audio fixed-margin control with matched episodic sampling."""
from pathlib import Path
source=Path("/root/train_coupled_authentication.py").read_text()
start=source.index("    result_folder=out")
end=source.index("    for method in",start)
source=source[:start]+"    tf=torch.zeros(len(u),192)\n"+source[end:]
source=source.replace('("CE","episode","coupled_margin","wrong_negative_relation")','("fixed_ACC_margin",)')
source=source.replace('"coupled_authentication"','"coupled_fixed_margin"')
source=source.replace('tm=target_score-ts.gather(1,hard)','tm=torch.ones_like(target_score-ts.gather(1,hard))')
exec(compile(source,"fixed_ACC_margin","exec"))

