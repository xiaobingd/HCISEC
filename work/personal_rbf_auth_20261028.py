"""Personal binary RBF readout with content-grouped training-only selection."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
import joblib
sys.path.insert(0,'/root')
import optimize_personal_teacher_auth_20261028 as exp
import modelb

OUT=Path('/autodl-fs/data/personal_rbf_auth_20261028')

def profile(x):
    # Original ACC only: frequency profile, temporal profile and their variability.
    return np.concatenate([x.mean(2),x.std(2),x.mean(1),x.std(1)],1)

def fit(blocks,y,c,gamma):
    scalers=[StandardScaler().fit(x) for x in blocks]
    x=np.concatenate([sc.transform(z)/np.sqrt(z.shape[1]) for sc,z in zip(scalers,blocks)],1)
    model=SVC(C=c,gamma=gamma,kernel='rbf',class_weight='balanced').fit(x,y)
    return scalers,model

def score(fitted,blocks):
    scalers,model=fitted
    x=np.concatenate([sc.transform(z)/np.sqrt(z.shape[1]) for sc,z in zip(scalers,blocks)],1)
    return model.decision_function(x)

def main():
    OUT.mkdir(parents=True,exist_ok=True); results={}
    for u in (1,4,5,2,7):
        print('TARGET',u,flush=True)
        if u in (2,7):
            definition='/autodl-fs/data/cyclegan_forward_aligned_20260928/user2/cyclegan/model_drop.py'
            source=Path('/autodl-fs/data/additional_personal_users_20261028')
        else:
            definition='/root/autodl-tmp/cycle1/experiments/samebo200safe_user3/model_drop.py'
            source=exp.OLD
        spec=importlib.util.spec_from_file_location('personal_architecture',definition)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        modelb.GeneratorAccToAudio=module.GeneratorAccToAudio; modelb.GeneratorAudioToAcc=module.GeneratorAudioToAcc
        # Expand target support in memory; split logic remains unchanged.
        text=(exp.base.PROJECT/'target_auth_protocol_150.py').read_text()
        text=text.replace('if target_user not in (1, 3, 4, 5):','if target_user not in tuple(range(1,10)):')
        import types
        protocol_module=types.ModuleType('rbf_protocol'); sys.modules[protocol_module.__name__]=protocol_module
        exec(compile(text,'rbf_protocol','exec'),protocol_module.__dict__)
        p=protocol_module.build_target_protocol(u,42,42)
        ck=torch.load(str(source/('U%d_baseline'%u)/'trainb_binary_auth_head.pth'),map_location=exp.D,weights_only=False)
        model,_=modelb.create_multi_source_model(ck['result']['cyclegan_ckpt'],freeze_generators=True,proj_dim=128,device=exp.D)
        model.load_state_dict(ck['model_state_dict']); model.eval()
        binary=nn.Linear(512,1).to(exp.D); binary.load_state_dict(ck['binary_state_dict']); binary.eval()
        teacher=exp.base.Teacher().to(exp.D)
        teacher.load_state_dict(torch.load(str(source/('U%d_teacher.pt'%u)),map_location=exp.D,weights_only=False)['state']); teacher.eval()
        names=['auth_train','target_dev_pool','impostor_calibration']
        b=exp.baseline_features(model,binary,{k:p.arrays[k] for k in names})
        tf={k:exp.embed(teacher,p.arrays[k],exp.generate(model.g_acc2audio,p.arrays[k])) for k in names}
        def blocks(k,mode):
            if mode=='raw_profile':return [profile(p.arrays[k])]
            z=[b[k]['feature']]
            if mode in ('baseline_teacher','baseline_raw_teacher'):
                z.extend([tf[k]['acc'],tf[k]['generated']])
            if mode=='baseline_raw_teacher':z.append(profile(p.arrays[k]))
            return z
        y=p.labels['auth_train']==u-1
        words=np.asarray([int(str(s).split(':')[1])//10 for s in p.sample_ids['auth_train']])
        folds=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,words))
        assert all(not set(words[a])&set(words[b]) for a,b in folds)
        choices={}; history=[]
        for mode in ('baseline','baseline_teacher','baseline_raw_teacher','raw_profile'):
            best=None; bx=blocks('auth_train',mode)
            for c in (1.,10.,100.):
                for gamma in ('scale',.1,1.):
                    oof=np.full(len(y),np.nan)
                    for a,v in folds:
                        fitted=fit([x[a] for x in bx],y[a],c,gamma)
                        oof[v]=score(fitted,[x[v] for x in bx])
                    t=exp.base.original.choose_threshold(oof[y],oof[~y]); m=exp.metrics(oof[y],oof[~y],t)
                    row={'mode':mode,'C':c,'gamma':gamma,'oof':m}; history.append(row)
                    rank=(-m['auc'],m['hter'],c)
                    if best is None or rank<best[0]:best=(rank,row)
            choices[mode]=best[1]
            print('MODE_LOCKED',u,mode,best[1],flush=True)
        # Raw-profile-only is a diagnostic; deployed candidate keeps all baseline features.
        selected=min((k for k in choices if k!='raw_profile'),key=lambda k:(-choices[k]['oof']['auc'],choices[k]['oof']['hter'],k!='baseline'))
        configs={'selected':selected,'choices':choices,'history':history,'selection':'5-fold disjoint-word OOF AUC on auth_train; final samples unchanged'}
        (OUT/('U%d_locked.json'%u)).write_text(json.dumps(configs,indent=2))
        fitted={}
        for name,row in choices.items():
            fitted[name]=fit(blocks('auth_train',name),y,row['C'],row['gamma'])
            sg=score(fitted[name],blocks('target_dev_pool',name)); sn=score(fitted[name],blocks('impostor_calibration',name))
            row['threshold']=float(exp.base.original.choose_threshold(sg,sn))
        joblib.dump(fitted,str(OUT/('U%d_classifiers.joblib'%u)))
        (OUT/('U%d_locked.json'%u)).write_text(json.dumps(configs,indent=2))
        final=('final_genuine','final_impostor')
        b.update(exp.baseline_features(model,binary,{k:p.arrays[k] for k in final}))
        tf.update({k:exp.embed(teacher,p.arrays[k],exp.generate(model.g_acc2audio,p.arrays[k])) for k in final})
        rows={}; dumped={}
        for name,row in choices.items():
            ss={k:score(fitted[name],blocks(k,name)) for k in final}
            rows[name]=exp.metrics(ss['final_genuine'],ss['final_impostor'],row['threshold'])
            rows[name]['configuration']=row
            for k,s in ss.items():dumped[name+'_'+k]=s
        np.savez_compressed(str(OUT/('U%d_final_scores.npz'%u)),**dumped)
        results[str(u)]={'selected_on_training_oof':selected,'routes':rows,'old_reference':ck['result']['final_attack'],
                         'personal_teacher':str(source/('U%d_teacher.pt'%u)),
                         'generator':ck['result']['cyclegan_ckpt'],'all_final_queries_retained':True,'final_reused_exploratory':True}
        (OUT/'results.json').write_text(json.dumps(results,indent=2))
        print('TARGET_DONE',u,{k:(v['tar'],v['far'],v['auc']) for k,v in rows.items()},flush=True)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
