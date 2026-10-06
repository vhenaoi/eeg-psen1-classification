"""
partial_confounder_test_v2.py
=============================
Partial confounder test (Spisak 2022, mlconfound) sobre predicciones out-of-fold PROMEDIADAS entre las 5 particiones
(ENS_soft). Corrige la v1: el sexo ausente (Seoul y 29 de Medellin_duque) ya NO se codifica como 0.5; cada confusor se
evalua solo en los participantes donde esta disponible (n por confusor en la salida).

Uso: python partial_confounder_test_v2.py <exp_oof_dir> <cvrep_dir> <feather> <salida.csv> [--perms 1000] E03 E05 ...
"""
import glob, os, sys
import numpy as np, pandas as pd
import scipy.sparse as _sp
for _k in (_sp.csr_matrix, _sp.csc_matrix, _sp.coo_matrix):
    if not hasattr(_k, 'A'): _k.A = property(lambda self: self.toarray())
from mlconfound.stats import partial_confound_test

args = [a for a in sys.argv[1:]]
perms = 1000
if '--perms' in args:
    i = args.index('--perms'); perms = int(args[i + 1]); del args[i:i + 2]
oof_dir, cv_dir, feather, out_csv, ids = args[0], args[1], args[2], args[3], args[4:]
fe = pd.read_feather(feather, columns=['subject', 'SITE', 'sex']).drop_duplicates('subject').set_index('subject')
rows = []
for e in ids:
    files = [glob.glob(f'{oof_dir}/{e}_*/oof_predictions.csv')[0]] + [glob.glob(f'{cv_dir}/seed{s}/{e}_*/oof_predictions.csv')[0] for s in (1, 2, 3, 4)]
    O = [pd.read_csv(f).set_index('subject') for f in files]
    d = O[0][['y', 'age']].copy()
    d['p'] = pd.concat([o['proba_ENS_soft'] for o in O], axis=1).mean(axis=1)
    d['SITE'] = fe['SITE'].reindex(d.index)
    d['MEDELLIN'] = d['SITE'].fillna('').str.lower().str.startswith('medel').astype(int)
    d['sex'] = fe['sex'].reindex(d.index).map({'M': 1, 'F': 0})          # faltante -> NaN (no 0.5)
    for cname, cat in [('age', False), ('sex', True), ('SITE', True), ('MEDELLIN', True)]:
        m = d[['y', 'p', cname]].dropna()
        y, p = m['y'].values.astype(int), m['p'].values
        c = pd.factorize(m[cname])[0] if cat else m[cname].values.astype(float)
        if len(np.unique(c)) < 2 or y.sum() < 5: continue
        r = partial_confound_test(y, p, c, num_perms=perms, cat_y=True, cat_yhat=False, cat_c=cat, random_state=0, progress=False, n_jobs=1)
        rows.append(dict(exp=e, confounder=cname, n=len(m), n_pos=int(y.sum()), p_partial=r.p))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(out_csv, index=False)
print('done')
