"""
effect_sizes_median_g.py
========================
Tamano de efecto (Hedges g) por feature, portadores vs controles, sobre los 544 features armonizados (media por sujeto).
Resumen por comparacion: mediana de |g| sobre los 544 features, IC95% bootstrap (remuestreo de participantes dentro de cada grupo)
y p de permutacion (barajado de etiquetas, mismos tamanos de grupo) de la mediana de |g|.
El AUC depende del tamano de muestra de entrenamiento; g no se penaliza por reducir el numero de controles.

Uso: python effect_sizes_median_g.py <HARMONIZED.feather> <salida.csv> [--n 1000]
"""
import re, sys
import numpy as np, pandas as pd

feather, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 1000
H = pd.read_feather(feather)
meta = re.compile(r'^(group|age|SITE|orig_group|sex|education|subject)(_(sl|coh|ent|cross))?$')
feats = [c for c in H.columns if not meta.match(c)]
assert len(feats) == 544
S = H.groupby('subject')[feats].mean().join(H.groupby('subject')[['SITE', 'orig_group', 'age']].first())
MED = ['Medellín_hd', 'Medellin_duque', 'Medellin_ld']
hc = S[S.orig_group.fillna('').str.startswith('HC_')]
groups = {
    'SCr': S[S.orig_group == 'SCr'], 'ACr': S[S.orig_group == 'ACr'],
    'HC (all)': hc, 'HC (Medellin)': hc[hc.SITE.isin(MED)],
}
def hedges(A, B):
    n1, n2 = len(A), len(B)
    m1, m2 = A.mean(0), B.mean(0)
    v1, v2 = A.var(0, ddof=1), B.var(0, ddof=1)
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    g = (m1 - m2) / sp
    return g * (1 - 3 / (4 * (n1 + n2) - 9))
rng = np.random.default_rng(0)
rows = []
for case, ctrl in [('SCr', 'HC (all)'), ('SCr', 'HC (Medellin)'), ('ACr', 'HC (all)'), ('ACr', 'HC (Medellin)')]:
    A = groups[case][feats].values.astype(float); B = groups[ctrl][feats].values.astype(float)
    A = np.nan_to_num(A, nan=np.nanmean(A)); B = np.nan_to_num(B, nan=np.nanmean(B))
    obs = np.median(np.abs(hedges(A, B)))
    boot = [np.median(np.abs(hedges(A[rng.integers(0, len(A), len(A))], B[rng.integers(0, len(B), len(B))]))) for _ in range(N)]
    X = np.vstack([A, B]); n1 = len(A)
    null = []
    for _ in range(N):
        idx = rng.permutation(len(X)); null.append(np.median(np.abs(hedges(X[idx[:n1]], X[idx[n1:]]))))
    null = np.array(null)
    rows.append(dict(case=case, control=ctrl, n_case=len(A), n_ctrl=len(B), median_abs_g=obs,
                     ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                     null_median=np.median(null), null_p95=np.percentile(null, 95),
                     perm_p=(1 + (null >= obs).sum()) / (1 + N),
                     age_case=groups[case]['age'].mean(), age_ctrl=groups[ctrl]['age'].mean()))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
