"""Clock-only recovery of PCM callback discontinuities; no signal correlation or labels."""
import numpy as np

POLICY=dict(nominal_candidates=[44100,48000],robust_rate_block_callbacks=100,maximum_rate_relative_error=.03,
  jump_excess_seconds=.075,minimum_segment_callbacks=20,minimum_segment_seconds=.5,
  maximum_segment_residual_p95_seconds=.03,receptive_field_boundary_exclusion=True,
  failed_segment_action='mask auxiliary only;retain every ACC identity window',query_data_used=False)

def affine(end,host):
    slope,offset=np.polyfit(end-end[0],host-host[0],1);intercept=host[0]+offset-slope*end[0]
    residual=host-(intercept+slope*end)
    return slope,intercept,float(np.quantile(abs(residual),.95))

def clock_regions(end,host):
    end=np.asarray(end,dtype=float);host=np.asarray(host,dtype=float)
    if not(len(end)==len(host)>20 and np.isfinite(end).all() and np.isfinite(host).all() and (np.diff(end)>0).all() and (np.diff(host)>0).all()):
        raise ValueError('PCM and host clocks must both be strictly increasing')
    block_rates=[]
    for start in range(0,len(end),100):
        e,h=end[start:start+100],host[start:start+100]
        if len(e)>=20:
            slope,_,_=affine(e,h)
            if slope>0:block_rates.append(1/slope)
    robust=float(np.median(block_rates));nominal=min([44100,48000],key=lambda r:abs(r-robust))
    if abs(robust-nominal)/nominal>=.03:raise ValueError('Even clock-block rate not compatible with44.1/48k')
    jumps=np.flatnonzero(np.diff(host)-np.diff(end)/nominal>.075)+1
    bounds=np.r_[0,jumps,len(end)];segments=[];rejected=0
    for start,stop in zip(bounds,bounds[1:]):
        e,h=end[start:stop],host[start:stop]
        if len(e)<20 or (e[-1]-e[0])/nominal<.5:rejected+=1;continue
        slope,intercept,p95=affine(e,h)
        if slope<=0 or abs(1/slope-nominal)/nominal>=.03 or p95>.03:rejected+=1;continue
        segments.append(dict(pcm_start=0. if start==0 else float(end[start]),pcm_end=float(end[stop-1]),
          slope=float(slope),intercept=float(intercept),residual_p95_ms=p95*1000))
    global_slope,_,global_p95=affine(end,host)
    audit=dict(clock_estimated_rate=1/global_slope,clock_block_median_rate=robust,resampling_nominal_rate=nominal,
      callback_global_residual_p95_ms=global_p95*1000,callback_residual_p95_ms=max([s['residual_p95_ms'] for s in segments],default=0.),
      callback_discontinuities=len(jumps),valid_clock_segments=len(segments),masked_clock_segments=rejected)
    return nominal,segments,audit

def map_receptive_centers(centers_pcm,halfwidth_pcm,segments):
    centers_pcm=np.asarray(centers_pcm);host=np.full(len(centers_pcm),np.nan);regions=[]
    for segment in segments:
        good=(centers_pcm-halfwidth_pcm>=segment['pcm_start']-1e-8)&(centers_pcm+halfwidth_pcm<=segment['pcm_end']+1e-8)
        ids=np.flatnonzero(good)
        if len(ids)<2:continue
        assert np.array_equal(ids,np.arange(ids[0],ids[-1]+1))
        host[ids]=segment['intercept']+segment['slope']*centers_pcm[ids];regions.append([int(ids[0]),int(ids[-1]+1)])
    return host,np.array(regions,dtype=int).reshape(-1,2)

def regional_targets(audio_time,q,acc_time,regions,sigma=.1):
    from common import smooth_to_acc
    result=np.zeros((len(acc_time),q.shape[1]),dtype='float32');result[:,0]=1;valid=np.zeros(len(acc_time),bool)
    for first,last in regions:
        pp,mask=smooth_to_acc(audio_time[first:last],q[first:last],acc_time,sigma)
        assert not (valid&mask).any(),'Overlapping host clock regions'
        result[mask]=pp[mask];valid|=mask
    return result,valid
