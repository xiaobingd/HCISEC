"""EH-only exploratory transfer: train Audio-guided ridge locator, ACC-only query."""
import json,itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from explore_phoneme_localization import ROOT,BASE,unit
from audit_local_identity_residual import holm
OUT=ROOT/'outputs/vr261126_acc_phoneme_locator'
WORDS=[6,11,13,14]
ARMS=['correct_Audio_trained','wrongrepeat_Audio_trained','ACC_self_energy_trained','ACC_energy_peak','fixed_support_selected','oracle_Audio_selected']

def grid_descriptors(wave,centers):
    frames=np.stack([wave[:,int(round((t-.06)*1000)):int(round((t+.06)*1000))] for t in centers])
    frames=frames-frames.mean(2,keepdims=True);power=abs(np.fft.rfft(frames*np.hanning(120),n=1024,axis=2))**2;freq=np.fft.rfftfreq(1024,1/1000)
    bands=[(20,50),(50,80),(80,120),(120,180),(180,250),(250,350),(350,450)]
    spec=np.stack([np.log(np.maximum(power[:,:,(freq>=lo)&(freq<hi)].mean(2),1e-20)) for lo,hi in bands],2)
    return np.c_[spec.reshape(len(centers),21),frames.std(2)]

def fit_locator(x,y):
    sc=StandardScaler().fit(x.reshape(-1,24));xx=sc.transform(x.reshape(-1,24));reg=Ridge(alpha=10).fit(xx,y.ravel())
    return sc,reg

def locate(model,x):
    sc,reg=model;pred=reg.predict(sc.transform(x.reshape(-1,24))).reshape(x.shape[:2]);return np.argmax(pred,axis=1)

def main():
    OUT.mkdir(exist_ok=True);private=BASE/'phoneme';c=np.load(private/'features_private.npz');p=np.load(private/'localization_private.npz')
    raw=np.load(BASE/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz');acc=raw['acc'];subset=np.flatnonzero(np.isin(c['word'],WORDS));idx=c['indices'][subset]
    u,w=c['user'][subset],c['word'][subset];assert len(idx)==320 and raw['fit'][idx].all()
    manifest=json.loads((BASE/'mapping_44100/consistent_timebase/aligned_manifest_private.json').read_text(encoding='utf-8'))['records']
    selections=json.loads((ROOT/'outputs/vr261126_phoneme_localization_controls/selection_public.json').read_text(encoding='utf-8'))
    centers=np.arange(1.1,1.901,.02);features=np.stack([grid_descriptors(acc[int(i)],centers) for i in idx]);assert features.shape==(320,41,24)
    for pos in [0,159,319]:
      k=int(np.argmin(abs(centers-p['energycenters'][subset[pos]])))
      assert np.allclose(features[pos,k],p['energy'][subset[pos]],atol=1e-9)
    own=np.zeros(320);wrongphase=np.zeros(320)
    for pos,i in enumerate(idx):
      rec=manifest[int(i)];lo,hi=rec['audio_start_s'],rec['audio_end_s'];own[pos]=(c['phonebounds'][subset[pos]].mean()-lo)/(hi-lo)
    donor=np.arange(320)
    for us in range(1,11):
      for wo in WORDS:
        ix=np.flatnonzero((u==us)&(w==wo));donor[ix]=np.roll(ix,-4)
    wrongphase=own[donor];duration=np.array([np.diff(manifest[int(i)]['imu_interval'])[0] for i in idx])
    protocol=dict(scope='EH selectedafterhistoricaldiagnostic positivecorrect-vswrong relationship;exploratorynotblind;4historicalfitwords320utterances10users',
      teacher='cachedautomaticvowelposition plus priornested-support-selected Audiooffset;wrongteacher sameuserwordfourthotherrepeat',
      model='framewise standardizedRidge(alpha10),24rawACCfeatures,NOtime/word/userinput;41candidate120mswindows step20ms,centers1.1..1.9;argmax',
      target='Gaussian softbump sigma100ms aroundteachercenter;fixedbeforefit,noquerylossselection',
      exclusion='outerqueryword andtargetidentity excludedfromALLlocatorfit/scalerfit;threeotherEHwordsx9baseusers;offset selectionalso excludedouterword/target',
      inference='locator takesACCwindowfeatures only;registration andquery bothselectedbyfrozenlocator;queryAudio usedONLY offlineMAEdiagnostic,notpositionscoresorhyperparameters',
      backend='24D selectedACCfeatures,scaler9baseusersregistrationwordsonly,unitidentityprototype cosine;all10usersretained',
      controls=ARMS,primary='correctAudio-trained locator vswrongteacher andACCenergy,paired10user signflipHolm2',
      added_control='after firstreadout add sameRidge/samebudget ACCenergy-teacher selflocalizer asposthoccontrol;nohyperparametersearch orfavorable selection',
      limits='legacy3seccachealreadyAudio/utteranceanchored;NOTcontinuousAudiofreepipeline;nomanuallabelsorphysicalsync;noEER/HTER;noexternalblindwords',
      final_Audio_opened=False,neural_network_trained=False)
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[];diagnostics=[];weights={};audit=[]
    for qword in WORDS:
      support=w!=qword;query=w==qword
      for target in range(1,11):
        selection=next(v for v in selections if v['query_word']==qword and v['target']==target)
        assert qword not in selection['selector_words'] and target not in selection['selector_users']
        offset=selection['audio_offset'];fixedoffset=selection['fixed_offset'];fit=support&(u!=target)
        assert fit.sum()==216 and not np.any(query&fit)
        teacher=1.5+(own-.5)*duration+offset;wrongteacher=1.5+(wrongphase-.5)*duration+offset
        y=np.exp(-.5*((centers[None,:]-teacher[:,None])/.1)**2);yb=np.exp(-.5*((centers[None,:]-wrongteacher[:,None])/.1)**2)
        views={};locations={}
        ye=np.exp(-.5*((centers[None,:]-p['energycenters'][subset,None])/.1)**2)
        for arm,labels in [('correct_Audio_trained',y),('wrongrepeat_Audio_trained',yb),('ACC_self_energy_trained',ye)]:
          model=fit_locator(features[fit],labels[fit]);k=locate(model,features);views[arm]=features[np.arange(320),k];locations[arm]=centers[k]
          sc,reg=model;prefix=f'w{qword}_u{target}_{arm}'
          weights[prefix+'_mean']=sc.mean_;weights[prefix+'_scale']=sc.scale_;weights[prefix+'_coef']=reg.coef_;weights[prefix+'_intercept']=np.array(reg.intercept_)
          # Query and target labels cannot affect fitting or localization.
          if qword==WORDS[0] and target==1 and arm=='correct_Audio_trained':
            perturbed=labels.copy();perturbed[~fit]=1-perturbed[~fit]
            assert np.array_equal(k,locate(fit_locator(features[fit],perturbed[fit]),features))
        from explore_phoneme_localization import GRID
        ka=GRID.index(offset);kf=GRID.index(fixedoffset)
        views['ACC_energy_peak']=p['energy'][subset];views['fixed_support_selected']=p['fixed'][kf,subset];views['oracle_Audio_selected']=p['audio'][ka,subset]
        for arm,z in views.items():
          zz=StandardScaler().fit(z[fit]).transform(z);proto=unit(np.stack([zz[support&(u==us)].mean(0) for us in range(1,11)]));q=unit(zz[query]);scores=q@proto[target-1];labels=u[query]==target
          mat=q[labels]@proto.T;margin=mat[:,target-1]-np.max(np.delete(mat,target-1,axis=1),1)
          rows.append(dict(user=target,query_word=qword,arm=arm,auc=float(roc_auc_score(labels,scores)),margin=float(margin.mean())))
          if arm in locations:
            mask=query&(u==target);diagnostics.append(dict(user=target,query_word=qword,arm=arm,offline_teacher_center_MAE_ms=float(np.mean(abs(locations[arm][mask]-teacher[mask]))*1000)))
        audit.append(dict(target=target,query_word=qword,fit_users=sorted(set(map(int,u[fit]))),fit_words=sorted(set(map(int,w[fit]))),fit_utterances=int(fit.sum()),teacher_offset=offset))
      print('LOCATOR_QUERY_WORD',qword,flush=True)
    np.savez_compressed(private/'locator_parameters_private.npz',**weights)
    np.savez_compressed(private/'locator_inputs_private.npz',features=features,centers=centers,indices=idx)
    users=[];summary=[]
    for us in range(1,11):
      for arm in ARMS:
        rr=[v for v in rows if v['user']==us and v['arm']==arm];users.append(dict(user=us,arm=arm,auc=float(np.mean([v['auc'] for v in rr])),margin=float(np.mean([v['margin'] for v in rr]))))
    for arm in ARMS:
      rr=[v for v in users if v['arm']==arm];summary.append(dict(arm=arm,auc=float(np.mean([v['auc'] for v in rr])),margin=float(np.mean([v['margin'] for v in rr]))))
    comparisons=[];signs=np.array(list(itertools.product([-1,1],repeat=10)))
    for control in ['wrongrepeat_Audio_trained','ACC_energy_peak','ACC_self_energy_trained']:
      delta=np.array([next(v['auc'] for v in users if v['user']==us and v['arm']=='correct_Audio_trained')-next(v['auc'] for v in users if v['user']==us and v['arm']==control) for us in range(1,11)])
      comparisons.append(dict(control=control,delta_auc=float(delta.mean()),improved_users=int((delta>0).sum()),p=float(np.mean((signs*delta).mean(1)>=delta.mean()-1e-12))))
    for v,pval in zip(comparisons[:2],holm([v['p'] for v in comparisons[:2]])):v['holm_p']=float(pval)
    comparisons[-1]['posthoc_control']=True
    for name,obj in [('summary_public.json',summary),('per_user_public.json',users),('per_word_user_public.json',rows),('comparison_public.json',comparisons),('exclusion_audit_public.json',audit),('offline_localization_public.json',diagnostics),('verification_public.json',dict(status='PASS',fit_folds=40,fitted_locators=120,utterances=320,all_outerword_target_excluded=True,query_label_perturbation_no_effect=True,reference_features_replayed=True,locator_query_inputs_ACC_only=True,legacy_input_already_segmented=True,final_Audio_opened=False))]:
      (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,comparisons=comparisons)))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
