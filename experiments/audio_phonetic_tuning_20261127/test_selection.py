from train_select import select

def row(e,h,f=.04,r=.08):return dict(EER=e,HTER=h,FAR=f,FRR=r,selection_score=(e+h)/2)
chosen,gate=select(dict(BASE=row(.07,.09),A=row(.05,.12),B=row(.06,.08)))
assert chosen=='B' and gate,'EER-only improvement cannot override HTER degradation'
chosen,gate=select(dict(BASE=row(.07,.09),A=row(.06,.08,.06,.08),B=row(.069,.089)))
assert chosen=='B' and gate,'Large FAR degradation cannot pass the development guard'
chosen,gate=select(dict(BASE=row(.05,.06),A=row(.07,.08),B=row(.08,.09)))
assert chosen=='A' and not gate,'No passing candidate must be marked exploratory,not accepted'
print('PASS:jointEER/HTER,FAR/FRRguard,deterministic exploratory fallback')
