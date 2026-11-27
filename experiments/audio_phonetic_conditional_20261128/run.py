"""Fixed nonblank conditional supervision; read-only reuse of previous studies."""
import sys,json,hashlib,time,gc,subprocess
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import xlogy
from conditional_loss import targets as conditional_targets,loss as conditional_kl,MASS_FLOOR
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'experiments/audio_phonetic_cross_content_20261127'
sys.path.insert(0,str(OLD))
from common import Student,crossword_supcon,phone_kl,shifted_posteriors
from train import batch_indices,gradient_stats
from evaluate import metrics,normalize
from prepare_acc import OPT,DEV,REG,CAL,SHORT,LONG,CHINESE

SOURCE=ROOT/'outputs/vr261119_10x40_raw_acc/hierarchical_phonetic'
PRIV=ROOT/'outputs/vr261119_10x40_raw_acc/hierarchical_phonetic_conditional'
OUT=ROOT/'outputs/vr261128_audio_phonetic_conditional'
PREVIOUS=ROOT/'outputs/vr261127_audio_phonetic_tuning'
FROZEN=ROOT/'outputs/vr261127_audio_phonetic_cross_content'
REFERENCE=ROOT/'outputs/vr261119_10x40_raw_acc/hierarchical_phonetic_tuning'
CONFIGS=dict(MATCHED=dict(phone=0.,content=.01,kind='none'),
  ORIGINAL=dict(phone=.01,content=.01,kind='full'),
  TUNED=dict(phone=.01,content=.01,kind='conditional'),
  SHUFFLED=dict(phone=.01,content=.01,kind='conditional'))
SEEDS=[42,43,44]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def freeze():
    OUT.mkdir(parents=True,exist_ok=True);PRIV.mkdir(parents=True,exist_ok=True)
    audit=json.loads((FROZEN/'data_audit_public.json').read_text(encoding='utf-8'))
    teacher=json.loads((FROZEN/'teacher_audit_public.json').read_text(encoding='utf-8'))
    assert sha(SOURCE/'acc_private.npz')==audit['data_sha256'] and sha(SOURCE/'teacher_private.npz')==teacher['teacher_cache_sha256']
    spec=dict(reference_commit='53be0e2539dcecfb9637656ae7b0f645946ffc03',
      data_sha256=audit['data_sha256'],teacher_sha256=teacher['teacher_cache_sha256'],
      loss_source_sha256=sha(Path(__file__).parent/'conditional_loss.py'),configs=CONFIGS,
      loss='teacher conditional q[k>0]/sum(q[k>0]);student logsoftmax(logits[k>0]);weight=nonblankmass*maxconditionalprob*valid*(mass>=.001);sumweightedKL/sumweights',
      confidence='CTC posterior proxy,not physicalVAD/groundtruthphoneconfidence',
      development=dict(words=OPT,evaluation_words=DEV,steps=400,seed=42,
        diagnostic='4cyclic DEV folds:2wordsK4register,1cal,1query;fixedhyperparameters,no selection scan'),
      final=dict(words=OPT+DEV,steps=1000,seeds=SEEDS,
        methods=['BASE','ORIGINAL','MATCHED','TUNED','SHUFFLED'],
        names=dict(ORIGINAL='matchedlambda full392KL',TUNED='conditional391KL',SHUFFLED='conditionaljointposteriorandweighttimeshift'),
        reuse='BASEoriginalpureACC,MATCHEDpreviouscontent.01noAudio',registration_K=12,
        thresholds='genuineCalQ10/impostorCalQ95midpoint;all150lockedbeforequeries'),
      unchanged='147channelCNN,192L2identity,392phonehead,CE+.1crosswordSupCon;lr.0003wd.01clip1;batch80;sigma100ms;all10users/all15querywords',
      scope='closedset;historicalquerypool exploratory;fixedonecoefficient,no queryhyperselection',query_Audio_reads=0)
    path=OUT/'protocol.json'
    if path.exists():assert digest(json.loads(path.read_text(encoding='utf-8')))==digest(spec)
    else:write('protocol.json',spec)
    return spec

def inputs(stage):
    data=np.load(SOURCE/'acc_private.npz');acc=data['acc'];valid=data['valid'];u,w=data['user'],data['word'];words=OPT if stage=='development' else OPT+DEV
    fit=np.flatnonzero(np.isin(w,words));values=acc[fit].transpose(0,2,1)[valid[fit]];mean=values.mean(0);std=np.maximum(values.std(0),1e-6)
    x=torch.tensor((acc-mean[None,:,None])/std[None,:,None]);del values,acc
    path=PRIV/f'normalization_{stage}_private.npz'
    if path.exists():
        old=np.load(path);assert np.array_equal(old['mean'],mean) and np.array_equal(old['std'],std)
    else:np.savez_compressed(path,mean=mean,std=std,fit_indices=fit)
    return data,x,torch.tensor(valid),u,w,words

def model_from(path,words):
    saved=torch.load(path,weights_only=True,map_location='cpu');model=Student(392,words);model.load_state_dict(saved['model']);model.eval()
    return model,saved

def embed(model,x,valid,ii):
    with torch.inference_mode():
        return np.concatenate([model(x[b],valid[b],auxiliary=False)[0].numpy() for b in np.array_split(ii,max(1,int(np.ceil(len(ii)/64))))])

def train_one(stage,code,seed,config,context,teacher,spec):
    data,x,valid,u,w,words=context;steps=400 if stage=='development' else 1000;path=PRIV/f'{stage}_{seed}_{code}_private.pt'
    if path.exists():
        saved=torch.load(path,weights_only=True);assert saved['spec_sha256']==digest(spec) and saved['steps']==steps and saved['config']==config
        print('REUSED',stage,seed,code,flush=True);return path
    batches=batch_indices(u,w,words,seed,steps);np.savez_compressed(PRIV/f'{stage}_{seed}_batches_private.npz',batches=batches)
    torch.manual_seed(seed);model=Student(392,len(words));model.train();initial=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);q,mask,inverse=teacher;shift=None;perm=None
    if code=='SHUFFLED':
        shift,perm,changed=shifted_posteriors(q,mask,seed);np.savez_compressed(PRIV/f'{stage}_{seed}_permutation_private.npz',permutation=perm)
        write(f'shuffle_{stage}_{seed}_public.json',dict(mean_frame_L1=float(np.mean(changed)),mask_preserved=True,window_marginal_preserved=True))
    target=shift if shift is not None else q
    if config['kind']=='conditional':target,phone_weight,logsum=conditional_targets(target,mask)
    else:phone_weight=mask;logsum=xlogy(target,target).sum(-1)
    yc=torch.tensor(u-1,dtype=torch.long);wc=torch.tensor(w,dtype=torch.long);lookup={wo:j for j,wo in enumerate(words)};wordclass=torch.tensor([lookup.get(int(wo),-1) for wo in w],dtype=torch.long);trace=[];grad=[];began=time.perf_counter()
    for step,ii in enumerate(batches):
        assert np.isin(w[ii],words).all();optimizer.zero_grad();z,phone,content=model(x[ii],valid[ii])
        ce=F.cross_entropy(16*z@F.normalize(model.classifier.weight,dim=1).T,yc[ii]);sup=crossword_supcon(z,yc[ii],wc[ii]);identity=ce+.1*sup
        lp=z.sum()*0;lc=z.sum()*0;frames=0
        if config['content']:lc=F.cross_entropy(content,wordclass[ii])
        if config['phone']:
            jj=inverse[ii];assert (jj>=0).all()
            if config['kind']=='conditional':
                mm=torch.tensor(phone_weight[jj])*valid[ii].float();frames=int((mm>0).sum())
                lp=conditional_kl(phone,torch.tensor(target[jj]),mm,torch.tensor(logsum[jj]))
            else:
                mm=torch.tensor(mask[jj])&valid[ii];frames=int(mm.sum());lp=phone_kl(phone,torch.tensor(target[jj]),mm,torch.tensor(logsum[jj]))
        loss=identity+config['phone']*lp+config['content']*lc;assert torch.isfinite(loss)
        if step in [0,99,steps-1]:grad.append(dict(step=step+1,**gradient_stats(identity,config['phone']*lp,config['content']*lc,list(model.backbone.parameters()))))
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));optimizer.step()
        if (step+1)%100==0:
            row=dict(step=step+1,identity_ce=float(ce.detach()),supcon=float(sup.detach()),phone_KL=float(lp.detach()),content_CE=float(lc.detach()),valid_phone_frames=frames,gradient_norm=norm,elapsed_s=time.perf_counter()-began)
            trace.append(row);print('TUNING_TRAIN',stage,seed,code,step+1,round(row['elapsed_s'],1),flush=True)
    saved=dict(model=model.state_dict(),spec_sha256=digest(spec),config=config,steps=steps,seed=seed,code=code,
      initial_sha256=initial,batch_sha256=hashlib.sha256(batches.tobytes()).hexdigest(),trace=trace,gradient=grad)
    torch.save(saved,path);write(f'train_{stage}_{seed}_{code}_public.json',{k:v for k,v in saved.items() if k!='model'})
    del model,optimizer,shift;gc.collect();return path

def development_score(path,context):
    data,x,valid,u,w,_=context;model,_=model_from(path,12);rows=[]
    for fold in range(4):
        qw,cw=DEV[fold],DEV[(fold+1)%4];rw=[DEV[(fold+2)%4],DEV[(fold+3)%4]]
        reg=np.array([[int(i) for wo in rw for i in np.flatnonzero((u==user)&(w==wo))[:2]] for user in range(1,11)])
        cal=np.flatnonzero(w==cw);query=np.flatnonzero(w==qw)
        assert reg.shape==(10,4) and len(set(reg.ravel())&set(cal))==len(set(reg.ravel())&set(query))==0 and not np.isin(query,cal).any()
        assert not np.isin(w[reg],OPT).any() and qw not in OPT and cw not in OPT
        prototype=normalize(embed(model,x,valid,reg.ravel()).reshape(10,4,192).mean(1));sc=embed(model,x,valid,cal)@prototype.T;sq=embed(model,x,valid,query)@prototype.T
        for user in range(1,11):
            tau=(np.quantile(sc[u[cal]==user,user-1],.1)+np.quantile(sc[u[cal]!=user,user-1],.95))/2
            rows.append(dict(fold=fold,query_word=qw,calibration_word=cw,user=user,**metrics(u[query]==user,sq[:,user-1],tau)))
    summary={key:float(np.mean([r[key] for r in rows])) for key in ['EER','HTER','FAR','FRR','AUC']};summary['selection_score']=(summary['EER']+summary['HTER'])/2
    return summary,rows

def main():
    torch.set_num_threads(4);spec=freeze();cache=np.load(SOURCE/'teacher_private.npz');data=np.load(SOURCE/'acc_private.npz')
    inverse=np.full(len(data['user']),-1,dtype=int);inverse[cache['indices']]=np.arange(len(cache['indices']))
    q,mask=cache['posterior'],cache['valid'];teacher=(q,mask,inverse)
    cond,weight,_=conditional_targets(q,mask);mass=q[...,1:].sum(-1)
    write('teacher_conditioning_audit_public.json',dict(windows=len(q),valid_teacher_frames=int(mask.sum()),
      positive_weight_frames=int((weight>0).sum()),windows_without_conditional_supervision=int((~(weight>0).any(1)).sum()),
      mean_valid_nonblank_mass=float(mass[mask].mean()),mass_floor=MASS_FLOOR,
      mean_positive_conditional_confidence=float(cond.max(-1)[weight>0].mean()),
      blank_is_silence_claim=False,student_blank_auxiliary_gradient='exactzero tested',query_Audio_reads=0))
    del cond,weight,cache,data
    context=inputs('development');summaries={};details={}
    for code,config in CONFIGS.items():
        path=train_one('development',code,42,config,context,teacher,spec);summaries[code],details[code]=development_score(path,context)
        print('CONDITIONAL_DEV',code,json.dumps(summaries[code]),flush=True)
    subprocess.run([sys.executable,str(Path(__file__).parent/'diagnose.py')],cwd=ROOT,check=True)
    lockpath=OUT/'selection_lock_public.json'
    fixed=CONFIGS['TUNED'];gate=all(summaries['TUNED'][k]<=summaries['MATCHED'][k] for k in ['EER','HTER'])
    lock=dict(selected='CONDITIONAL_FIXED',config=fixed,summaries=summaries,details=details,
      development_dual_metric_and_FAR_FRR_gate_passed=gate,gate_is_diagnostic_not_parameter_choice=True,
      locked_at_unix=time.time(),spec_sha256=digest(spec),teacher_sha256=spec['teacher_sha256'],
      final_query_scores_used_for_selection=0,fixed_single_lambda=.01,hyperparameter_search=False)
    if lockpath.exists():
        old=json.loads(lockpath.read_text(encoding='utf-8'));assert old['config']==fixed and old['spec_sha256']==digest(spec)
    else:write(lockpath.name,lock)
    print('FIXED_CONDITIONAL_CONFIG_LOCKED',fixed,flush=True)
    del context;gc.collect();context=inputs('final');catalog=[]
    for seed in SEEDS:
        for code in ['BASE','ORIGINAL','MATCHED','TUNED','SHUFFLED']:
            reuse=code in ['BASE','MATCHED']
            if code=='BASE':path=SOURCE/f'final_{seed}_T0_private.pt';config=dict(phone=0.,content=0.,kind='none')
            elif code=='MATCHED':path=REFERENCE/f'final_{seed}_MATCHED_private.pt';config=CONFIGS['MATCHED']
            else:
                config=CONFIGS[code];path=train_one('final',code,seed,config,context,teacher,spec)
            assert path.exists();catalog.append(dict(code=code,seed=seed,config=config,path=str(path),checkpoint_sha256=sha(path),reused_reference=reuse))
    (PRIV/'catalog_private.json').write_text(json.dumps(catalog,indent=2),encoding='utf-8')
    write('model_catalog_public.json',[{k:v for k,v in row.items() if k!='path'} for row in catalog])
    print('CONDITIONAL_ALL_MODELS_READY',flush=True)
    del context,teacher;gc.collect()
    for name in ['evaluate_conditional.py','report.py']:
        subprocess.run([sys.executable,str(Path(__file__).parent/name)],cwd=ROOT,check=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
