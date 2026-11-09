"""Publish explicitly authorized anonymous per-user aggregate metrics only."""
import json,argparse
from pathlib import Path
import numpy as np
from run_rkd01_vibravox import thresholds,metrics,comparisons,prepared
def table(rows):
    s='\n## 逐用户结果（匿名编号）\n\n按用户要求补充。只公开逐用户汇总，不公开原始样本、分数明细、阈值或身份映射；以下均为Cal-EER工作点。\n\n| 用户 | 条件 | 表示 | TAR | FAR | HTER | AUC | 本人/攻击次数 |\n|---|---|---|---:|---:|---:|---:|---|\n'
    for u,c,m,v in rows:s+=f"| {u} | {c} | {m} | {100*v['tar']:.2f}% | {100*v['far']:.2f}% | {100*v['hter']:.2f}% | {v['auc']:.4f} | {v['genuine_n']}/{v['impostor_n']} |\n"
    return s
def main(a):
    root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for name,n,sub in [('abcs',8,'abcs_shape_v4_20261110'),('mei9',9,'mei9_shape_v4_20261110')]:
        rows=[]
        for u in range(1,n+1):
            r=json.loads((root/sub/f'target{u}'/'results_private.json').read_text())['results']
            for m,v in r.items():
                if name=='abcs':rows.append((f'ABCS-{u:02}','跨句',m,v['unseen_attacker']['Cal_EER']))
                else:
                    for c in ['sameword','crossword']:rows.append((f'raw9-{u:02}',c,m,v['two_unseen'][c]['Cal_EER']))
        (out/f'{name}_users_public.md').write_text(table(rows),encoding='utf8')
    prior=root/'kd_rkd01_vibravox_20261109';d=dict(np.load(prior/'inputs_private.npz'));_,_,shape,_=prepared(d,list(range(35)));rows=[];cal=list(range(35,47));test=list(range(47,59));cs,cy,_=comparisons(shape,d,cal,12);tau=thresholds(cs,cy)['Cal_EER']
    for j,u in enumerate(test,1):
        s,y,_=comparisons(shape,d,[u]+[v for v in test if v!=u],12);count=288;ss=s[:count];yy=y[:count];rows.append((f'Test-{j:02}','跨句','shape40',metrics(ss,yy,tau)))
    # Shared comparisons returns all clients; select explicit owner blocks.
    for tag,folder in [('student192',root/'registration_rkd02_20261110'),('supplement232',root/'shape_supplement_rkd03_20261110')]:
        if tag=='supplement232':
            q=json.loads((folder/'scores_private.json').read_text());cc=q['calibration'];tt=q['test'];t=q['thresholds']['Cal_EER']
        else:
            cc=[json.loads((folder/'outer'/f'owner{u}'/'V4_rho0.3_private.json').read_text()) for u in cal];tt=[json.loads((folder/'outer'/f'owner{u}'/'V4_rho0.3_private.json').read_text()) for u in test];t=thresholds(np.concatenate([v['scores'] for v in cc]),np.concatenate([v['labels'] for v in cc]))['Cal_EER']
        for j,v in enumerate(tt,1):rows.append((f'Test-{j:02}','跨句',tag,metrics(np.array(v['scores']),np.array(v['labels']),t)))
    (out/'vibravox_users_public.md').write_text(table(rows),encoding='utf8');print('EXPORTED_ANONYMOUS_USER_TABLES')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/autodl-fs/data');p.add_argument('--out',required=True);main(p.parse_args())
