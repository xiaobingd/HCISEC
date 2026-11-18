"""Read-only capture-timing template control; no neural fitting or source-test access."""
import os,argparse,json
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np

def corr(a,b):
    a=a.ravel();b=b.ravel();return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-9 and b.std()>1e-9 else 0.
def detail(a):return a-a.mean(-1,keepdims=True)-a.mean(-2,keepdims=True)+a.mean((-2,-1),keepdims=True)
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
with np.load(root/'data/mei_val.npz') as d:val={k:d[k] for k in ['x','y','ids']}
assert all(np.sum(val['ids']==n)==4 for n in np.unique(val['ids']))
rows={};allruns=json.loads((root/'results/results.json').read_text())+json.loads((root/'results_long/results.json').read_text())+json.loads((root/'results_long20/results.json').read_text())
for size in [20,70,180]:
    with np.load(root/'data'/('mei'+str(size)+'_train.npz')) as d:y=d['y'];ids=d['ids']
    assert all(np.sum(ids==n)==4 for n in np.unique(ids))
    # Every mei source has the same four starts0/.4/.8/1.2s; no VAL-derived fitting.
    template=y.reshape(size,4,80,81).mean(0);pred=np.tile(template,(60,1,1))
    records=[]
    for n in np.unique(val['ids']):
        ii=val['ids']==n;a,b=pred[ii],val['y'][ii]
        records.append(dict(recording=str(n),audio_mae=float(np.mean(abs(a-b))),audio_detail_corr=corr(detail(a),detail(b)),audio_envelope_corr=corr(a.mean(1),b.mean(1))))
    rows[str(size)]=dict(macro={k:float(np.mean([r[k] for r in records])) for k in records[0] if k!='recording'},per_record=records)
comparisons=[]
for run in allruns:
    if run['route']=='personal20':continue
    size=run['n_train_recordings'];baseline={r['recording']:r['audio_mae'] for r in rows[str(size)]['per_record']};rr=run['validation']['per_record']
    delta=np.array([baseline[r['recording']]-r['audio_mae'] for r in rr])
    comparisons.append(dict(route=run['route'],seed=run['seed'],model_audio_mae=run['validation']['macro']['audio_mae'],time_template_audio_mae=rows[str(size)]['macro']['audio_mae'],mean_template_minus_model=float(delta.mean()),n_model_beats_time_template=int(np.sum(delta>0)),n_records=len(delta)))
result=dict(protocol='TRAIN-only microphone means separately for0/.4/.8/1.2s crop positions; predicts VAL without ACC. Checks capture-flow prior; NOT a usable identity model. All same60source records, no VAL/test template fitting.',templates=rows,model_comparisons=comparisons)
(root/'time_template_controls.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(templates={k:v['macro'] for k,v in rows.items()},comparisons=comparisons),indent=2))
