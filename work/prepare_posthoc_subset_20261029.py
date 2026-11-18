from pathlib import Path
import ast
p=Path('work/run_pruned_auth_users_20261029.py');s=p.read_text(encoding='utf-8')
def change(a,b):
 global s
 assert a in s,a
 s=s.replace(a,b)
change('Remove train-OOF-hard negative identities from auth fitting and calibration, never final queries.','POST-HOC TEST-INFORMED restricted-cohort experiment; not independent validation.')
change('mei_pruned_auth_users_20261029','mei_posthoc_subset_20261029')
change("ranking='Lowest pairwise AUC per attacker from original fit80 baseline RBF TRAIN nested OOF; ties UID; no final scores used'","ranking='Highest prior baseline RBF FINAL FAR per attacker, ties UID; deliberately test-informed'")
change("never_removed_from='final test;all eight attack identities and all queries preserved'","final_evaluation='Primary excludes selected identities; full-eight scores retained as diagnostic; TEST-INFORMED selection'")
change("scope='Identity selection sees original train OOF and upstream teachers saw all nine users; NOT truly unseen identity/open-set experiment'","scope='Restricted attacker cohort selected using prior final results; NOT independent final performance or open-set validation; upstream remains all nine users'")
start=s.index("  original=np.load(str(FIT/")
end=s.index('  models={};locked={};private_oof={}',start)
s=s[:start]+'''  previous=json.loads((PRIOR/('U%d'%u)/'results_public.json').read_text())
  attacks=previous['reference']['baseline_rbf']['attack_by_user'];hardness=[]
  for v,b in attacks.items():
   hardness.append(dict(user=int(v),prior_test_far=float(b['accepted']/b['n']),prior_test_accepted=b['accepted'],prior_test_n=b['n']))
  hardness.sort(key=lambda a:(-a['prior_test_far'],a['user']))
  save(folder/'ranking_test_informed.json',dict(target=u,ranking=hardness,scope='Explicitly selected using prior final attack outcomes; descriptive post-hoc subset only',test_informed=True))
'''+s[end:]
change("dict(target=u,routes=locked,final_seen=False)","dict(target=u,routes=locked,selection_test_informed=True,new_scores_read=False)")
change("   rows[key]=result",'''   all_diagnostic=result
   keep=~np.isin(p.labels['final_impostor'],np.asarray(spec['removed'])-1)
   import types
   limited_protocol=types.SimpleNamespace(labels={'final_impostor':p.labels['final_impostor'][keep]})
   limited=old.sel.evaluate_route({'final_genuine':ss['final_genuine'],'final_impostor':ss['final_impostor'][keep]},limited_protocol,u,spec['threshold'])
   limited.update({k:v for k,v in spec.items() if k!='candidates'})
   limited['test_informed_subset']=True
   limited['all_attackers_diagnostic']=all_diagnostic
   rows[key]=limited''')
ast.parse(s)
Path('work/run_posthoc_subset_20261029.py').write_text(s,encoding='utf-8')
print('Generated explicit test-informed subset script')

