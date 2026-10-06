"""
oof_site_stratified_auc.py
==========================
Evaluacion dentro de Medellin de modelos entrenados con TODO el dataset.

Para cada experimento se promedian por participante las probabilidades out-of-fold del ensemble blando (ENS_soft) de las cinco
particiones (semillas 42, 1-4; oof_predictions.csv de cada corrida) y se calcula el AUC solo sobre participantes de Medellin:
  (a) todos los participantes de Medellin;
  (b) los que estan en el rango de edad de los portadores (percentiles 5-95 de la edad de los portadores).
Se reporta el IC95% bootstrap por participante y, como referencia, el AUC de la edad sola.

Uso:
  python oof_site_stratified_auc.py <exp_dir_semilla42> <cvrep_dir> <feather> <salida.csv> E03 E05 E40 E41
"""
import glob, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); out = []
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        if 0 < y[b].sum() < len(b): out.append(roc_auc_score(y[b], p[b]))
    return np.percentile(out, [2.5, 97.5])

if __name__ == '__main__':
    exp42, cvrep, feather, out_csv, ids = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
    fe = pd.read_feather(feather, columns=['subject', 'SITE']).drop_duplicates('subject')
    site = dict(zip(fe['subject'], fe['SITE']))
    rows = []
    for e in ids:
        files = [glob.glob(f'{exp42}/{e}_*/oof_predictions.csv')[0]] + [glob.glob(f'{cvrep}/seed{s}/{e}_*/oof_predictions.csv')[0] for s in (1, 2, 3, 4)]
        O = [pd.read_csv(f).set_index('subject') for f in files]
        d = O[0][['y', 'age']].copy()
        d['p'] = pd.concat([o['proba_ENS_soft'] for o in O], axis=1).mean(axis=1)
        d['SITE'] = d.index.map(site)
        med = d[d['SITE'].fillna('').str.lower().str.startswith('medel')]
        car = med[med.y == 1]; lo, hi = car.age.quantile(0.05), car.age.quantile(0.95)
        for label, s in [('Medellin (all)', med), ('Medellin, age range %.0f-%.0f' % (lo, hi), med[(med.age >= lo) & (med.age <= hi)])]:
            y = s.y.values.astype(int); p = s.p.values
            l, h = boot_ci(y, p); a = roc_auc_score(y, s.age.values)
            rows.append(dict(exp=e, subset=label, n_pos=int(y.sum()), n_neg=int((1 - y).sum()), auc=round(roc_auc_score(y, p), 3),
                             ci_low=round(l, 3), ci_high=round(h, 3), auc_age_alone=round(max(a, 1 - a), 3)))
    r = pd.DataFrame(rows); print(r.to_string(index=False)); r.to_csv(out_csv, index=False)
