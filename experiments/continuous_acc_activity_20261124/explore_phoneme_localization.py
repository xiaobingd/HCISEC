"""Nested support-only localization selection; cached training Audio boundaries."""
import json,itertools,hashlib
from pathlib import Path
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from audit_phoneme_identity import ROOT,BASE,GROUPS,window,unit
from audit_local_identity_residual import holm
OUT=ROOT/'outputs/vr261126_phoneme_localization_controls'
GRID=[-.2,-.1,0.,.1,.2]
ARMS=['word_center','fixed_support_selected','correct_Audio_selected','wrongrepeat_Audio_sameoffset','ACC_energy_peak','Audio_nominal','Audio_minus100_fixed']

def macro_auc(z,u,w,supportwords,queryword,cohort,excluded):
    support=np.isin(w,supportwords)&np.isin(u,cohort)
    query=(w==queryword)&np.isin(u,cohort)
    assert queryword not in supportwords and excluded not in cohort
    zz=StandardScaler().fit(z[support]).transform(z)
    p=unit(np.stack([zz[support&(u==us)].mean(0) for us in cohort]));q=unit(zz[query]);scores=q@p.T
    return float(np.mean([roc_auc_score(u[query]==us,scores[:,j]) for j,us in enumerate(cohort)]))

def choose(scores):
    return min(range(len(GRID)),key=lambda k:(-round(scores[k],12),abs(GRID[k]),GRID[k]))

def main():
    OUT.mkdir(exist_ok=True);private=BASE/'phoneme';c=np.load(private/'features_private.npz')
    raw=np.load(BASE/'conditional_correspondence/coarse_pilot/raw_inputs_private.npz');acc=raw['acc'];indices=c['indices'];u,w,ids=[c[k] for k in ['user','word','sample_id']]
    assert c['success'].all() and len(indices)==720
    assert np.all(raw['fit'][indices]) and np.array_equal(ids,raw['sample_id'][indices])
    manifest=json.loads((BASE/'mapping_44100/consistent_timebase/aligned_manifest_private.json').read_text(encoding='utf-8'))['records']
    original=json.loads((BASE/'phrase_auth/source_manifest_private.json').read_text(encoding='gb18030'))
    center_error=max(abs(np.mean(manifest[int(i)]['imu_interval'])-np.mean(original[int(i)]['imu_interval'])) for i in indices);assert center_error<1e-9
    protocol=dict(samples=720,users=10,groups=GROUPS,arms=ARMS,offset_grid_seconds=GRID,window_seconds=.12,
      source='same720historicalfit utterances andfrozenautomaticphoneboundaries;noAudiofileopenedthisrun;cachedqueryAudio remainsoracleinfo',
      selection='per outerqueryword,targetidentity: exclude BOTH from hyperparameterselection; remainingwords innerleavewordout within vowelgroup,9backgroundidentities;groupswith>=2remainingwords;mean innerAUC acrossgroups andwords,then fixed tie rule',
      equal_budget='fixed-wordcenteroffset andcorrectAudiooffset each5candidates;wrongrepeatusescorrectAudioselectedoffset,notseparatelytuned',
      wrongrepeat='sameuser/sameword fourthotherrepeat normalizedphonephase,thenmaptoownwordinterval;nototherusersmeanalignment',
      energy='ACC40-450Hz squared3axisenergy,120ms rollingwindow20msstep;maximumcenter in[1.1,1.9]sec offixed3seccache;nodevelopmenttuning',
      fixed='wordcenter plus support-selectedoffset;usesnophoneboundary',
      diagnostic='nominal andpreviouslyinspected -100ms retainedasreferences,notindependentconfirmation',
      primary='correctAudio_selected vsfixed_support_selected pervowelgroup,exactpaireduser signflip,Holm3;wrongrepeatsecondary',
      frontend='same24D fixed120msrawACCstatistics,registeredprototype cosine;scaler only9backgroundidentities/supportwords;all10users retained',
      limitation='all9words previouslyviewed exploratory;legacy3secACCcachecenter alreadyuseshistoricalutterance segmentation: energy/fixedarms arelocalnoadditionalAudio,notfullyAudio-freecontinuouspipeline;queryAudiooracleforcorrect/wrongarms;noEER/HTER/deployedmodelclaim',
      no_neural_training=True,no_final_test_audio=True,no_private_export=True)
    (OUT/'protocol_public.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    phase=np.zeros(720);duration=np.zeros(720)
    for pos,i in enumerate(indices):
      rec=manifest[int(i)];lo,hi=rec['audio_start_s'],rec['audio_end_s'];phase[pos]=(c['phonebounds'][pos].mean()-lo)/(hi-lo);duration[pos]=np.diff(rec['imu_interval'])[0]
    wrong=np.arange(720)
    for us in range(1,11):
      for wo in sorted(set(w)):
        pos=np.flatnonzero((u==us)&(w==wo));wrong[pos]=np.roll(pos,-4)
    assert np.all(wrong!=np.arange(720)) and np.array_equal(u[wrong],u) and np.array_equal(w[wrong],w)
    audio=np.zeros((5,720,24));fixed=audio.copy();bad=audio.copy();energy=np.zeros((720,24));energycenters=np.zeros(720)
    filt=signal.butter(4,[40,450],fs=1000,btype='bandpass',output='sos')
    for pos,i in enumerate(indices):
      wave=acc[i];basecenter=1.5+(phase[pos]-.5)*duration[pos];wrongcenter=1.5+(phase[wrong[pos]]-.5)*duration[pos]
      for k,offset in enumerate(GRID):
        audio[k,pos]=window(wave,basecenter+offset);fixed[k,pos]=window(wave,1.5+offset);bad[k,pos]=window(wave,wrongcenter+offset)
      filtered=signal.sosfiltfilt(filt,wave,axis=1);centers=np.arange(1.1,1.901,.02)
      powers=[np.mean(filtered[:,int(round((t-.06)*1000)):int(round((t+.06)*1000))]**2) for t in centers]
      center=centers[int(np.argmax(powers))];energycenters[pos]=center;energy[pos]=window(wave,center)
    assert np.isfinite(audio).all() and np.isfinite(fixed).all() and np.isfinite(bad).all() and np.isfinite(energy).all()
    assert np.allclose(audio[1],c['offset_minus100'],atol=1e-9) and np.allclose(audio[2],c['shared_phone'],atol=1e-9)
    np.savez_compressed(private/'localization_private.npz',audio=audio,fixed=fixed,wrongrepeat=bad,energy=energy,energycenters=energycenters,indices=indices)
    selections=[];rows=[]
    allwords=sorted(set(map(int,w)))
    for queryword in allwords:
      outergroup=next(g for g,words in GROUPS.items() if queryword in words);supportwords=[wo for wo in GROUPS[outergroup] if wo!=queryword]
      for target in range(1,11):
        cohort=[us for us in range(1,11) if us!=target];selected={};candidate_scores={}
        validgroups=[[wo for wo in words if wo!=queryword] for words in GROUPS.values()];validgroups=[ww for ww in validgroups if len(ww)>=2]
        for name,features in [('fixed',fixed),('audio',audio)]:
          values=[]
          for k in range(5):
            groupscores=[]
            for words in validgroups:
              groupscores.append(np.mean([macro_auc(features[k],u,w,[v for v in words if v!=held],held,cohort,target) for held in words]))
            values.append(float(np.mean(groupscores)))
          selected[name]=choose(values);candidate_scores[name]=values
        selectorwords=sorted(set(itertools.chain.from_iterable(validgroups)));assert queryword not in selectorwords and target not in cohort
        selections.append(dict(query_word=queryword,target=target,selector_words=selectorwords,selector_users=cohort,fixed_offset=GRID[selected['fixed']],audio_offset=GRID[selected['audio']],candidate_auc=candidate_scores))
        views=dict(word_center=fixed[2],fixed_support_selected=fixed[selected['fixed']],correct_Audio_selected=audio[selected['audio']],wrongrepeat_Audio_sameoffset=bad[selected['audio']],ACC_energy_peak=energy,Audio_nominal=audio[2],Audio_minus100_fixed=audio[1])
        support=np.isin(w,supportwords);query=w==queryword;fitmask=support&(u!=target)
        for arm,z in views.items():
          zz=StandardScaler().fit(z[fitmask]).transform(z);p=unit(np.stack([zz[support&(u==us)].mean(0) for us in range(1,11)]));q=unit(zz[query]);scores=q@p[target-1]
          labels=u[query]==target;matrix=q[labels]@p.T; margin=matrix[:,target-1]-np.max(np.delete(matrix,target-1,axis=1),1)
          rows.append(dict(group=outergroup,held_word=queryword,user=target,arm=arm,auc=float(roc_auc_score(labels,scores)),margin=float(margin.mean())))
      print('OUTER_WORD',queryword,flush=True)
    users=[];summary=[]
    for g in GROUPS:
      for target in range(1,11):
       for arm in ARMS:
        rr=[v for v in rows if v['group']==g and v['user']==target and v['arm']==arm];users.append(dict(group=g,user=target,arm=arm,auc=float(np.mean([v['auc'] for v in rr])),margin=float(np.mean([v['margin'] for v in rr]))))
      for arm in ARMS:
        rr=[v for v in users if v['group']==g and v['arm']==arm];summary.append(dict(group=g,arm=arm,auc=float(np.mean([v['auc'] for v in rr])),margin=float(np.mean([v['margin'] for v in rr]))))
    signs=np.array(list(itertools.product([-1,1],repeat=10)));comparisons=[]
    for g in GROUPS:
      for control in ['fixed_support_selected','wrongrepeat_Audio_sameoffset','ACC_energy_peak']:
        delta=np.array([next(v['auc'] for v in users if v['group']==g and v['user']==us and v['arm']=='correct_Audio_selected')-next(v['auc'] for v in users if v['group']==g and v['user']==us and v['arm']==control) for us in range(1,11)])
        comparisons.append(dict(group=g,control=control,delta_auc=float(delta.mean()),improved_users=int((delta>0).sum()),user_signflip_p=float(np.mean((signs*delta).mean(1)>=delta.mean()-1e-12))))
    primary=[v for v in comparisons if v['control']=='fixed_support_selected']
    for v,p in zip(primary,holm([v['user_signflip_p'] for v in primary])):v['holm_primary_p']=float(p)
    for name,obj in [('summary_public.json',summary),('per_user_public.json',users),('per_word_user_public.json',rows),('selection_public.json',selections),('comparison_public.json',comparisons),('verification_public.json',dict(status='PASS',samples=720,outer_word_target_folds=90,scaler_target_excluded=True,selector_queryword_and_target_excluded=True,old_reference_features_reproduced=True,wrongrepeat_user_word_matched=True,max_legacy_ACC_center_change_seconds=center_error,final_Audio_opened=False,query_Audio_oracle_diagnostic=True))]:
      (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,comparisons=comparisons)))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
