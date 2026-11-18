"""Frozen windows: separate XYZ log-magnitude spectra plus multiscale energy.

Signed axes are not rectified before STFT; complex phase is not retained.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy import signal,ndimage
OUT=Path('outputs/activity_alignment_20261018');OUT.mkdir(exist_ok=True)
base=Path('outputs/imu_original_frontend_20261016');manifest=json.loads((base/'manifest.json').read_text());cache={};export={}
bands=[(2,10),(10,40),(40,100),(100,200),(200,450),(450,490)]
threshold=json.loads(Path('outputs/crossmodal_activity_20261017/results.json').read_text())['protocol']['audio_gmm']['threshold']
for split in ('train','val'):
 rows=[r for r in manifest['samples'] if r['split']==split];spectra=[];energies=[];activity=[];audio_energy=[];axes=[];ids=[]
 for row in rows:
  g=row['recording']
  if g not in cache:
   with np.load(Path('outputs/raw_imu_processed_20261014')/(g+'.npz')) as d:x=d['acc_raw'].copy();a=d['audio'];fs=int(d['audio_fs'])
   x-=x.mean(0);channels=[]
   for xx in [x]+[signal.sosfiltfilt(signal.butter(4,[lo,hi],btype='bandpass',fs=1000,output='sos'),x,axis=0) for lo,hi in bands]:
    for ms in (50,250):
     power=np.maximum(ndimage.uniform_filter1d(xx*xx,size=ms,axis=0,mode='reflect'),0)
     channels.extend([np.log10(np.sqrt(power[:,k])+1e-12) for k in range(3)]);channels.append(np.log10(np.sqrt(power.sum(1))+1e-12))
   ar=np.log10(np.sqrt(np.maximum(ndimage.uniform_filter1d(a*a,size=int(fs*.02),mode='reflect'),0))+1e-9)
   cache[g]=x,np.array(channels),ar,fs
  x,e,ar,fs=cache[g];b=int(round(row['start_s']*1000));w=x[b:b+800].T;assert w.shape==(3,800)
  _,_,z=signal.stft(w,fs=1000,nperseg=128,noverlap=118,nfft=128,axis=-1)
  spectra.append(20*np.log10(abs(z)+1e-12));axes.append(w)
  ix=np.minimum(b+np.arange(81)*10,e.shape[1]-1);energies.append(e[:,ix])
  ai=np.minimum(np.rint((row['start_s']+np.arange(81)*.01)*fs).astype(int),len(ar)-1)
  audio_energy.append(ar[ai]);activity.append(ar[ai]>threshold);ids.append(g)
 y=np.load(base/split/'imu_audio.npy');expected=np.load(base/split/'recording_ids.npy');assert np.array_equal(ids,expected)
 export[split]=dict(spec=np.array(spectra),energy=np.array(energies),activity=np.array(activity,dtype='float32'),audio_logrms=np.array(audio_energy),raw_wave=np.array(axes),audio=y,ids=np.array(ids))
lo,hi=np.percentile(export['train']['spec'],[1,99]);mu=export['train']['energy'].mean((0,2),keepdims=True);sd=np.maximum(export['train']['energy'].std((0,2),keepdims=True),.05)
for s,d in export.items():
 # One shared scalar spectral calibration, no per-axis/frequency/user equalization or clipping.
 spec=(2*(d['spec']-lo)/(hi-lo)-1).reshape(len(d['spec']),195,81)
 eng=(d['energy']-mu)/sd
 x=np.concatenate([spec,eng],axis=1).astype('float32')
 # Ungated descriptor retained exactly alongside any learned embedding.
 descriptor=np.concatenate([spec.mean(2),spec.std(2),eng.mean(2),eng.std(2)],1).astype('float32')
 np.savez_compressed(OUT/('data_'+s+'.npz'),x=x,y=d['audio'],activity=d['activity'],audio_logrms=d['audio_logrms'],raw_descriptor=descriptor,ids=d['ids'],raw_wave=d['raw_wave'].astype('float32'))
 assert np.isfinite(x).all()
print('exported',[(s,len(v['ids'])) for s,v in export.items()])
protocol=dict(source='same personal raw OpenVR vAccel,20train4val existingrecordings,58/16 frozen original windows',test_and_28reserve='not read/exported',input='DCremoved separateXYZ, no time-domain norm or main-spectrum cutoff; 0-500Hz log-magnitude spectrum3x65x81 plus56 multiscale energy features; STFT complex phase is not retained',augmentation='training-only shared temporal roll with seam masks; no pitch/gain/axis-specific augmentation',scales=dict(spec_lo_hi=[float(lo),float(hi)],energy_mean=mu.flatten().tolist(),energy_std=sd.flatten().tolist()),audio_activity_threshold=threshold,audio_activity_labels='fixed TRAIN-only nonzero-frame energy GMM from20261017, not pYIN/human speech labels',normalization='sameparameters allrecords; futuremultiusercomparisons must freeze a shared pooledtrain calibration, never refit per targetuser',raw_descriptor_dim=502,identity_scope='singleperson cannot measure identity discrimination. Selected raw descriptors retained exactly, not lossless original waveform preservation; ACC feature reconstruction tests retention, not identityaccuracy',audio_reference='original80x81 normalized Mel, unchanged',notes='Activity supervision and crossmodal temporalalignment do not require matching spectralstructures. Alignmentloss is trainingonly; no microphone used at inference. No absolute-time positionalembeddings.')
(OUT/'data_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
