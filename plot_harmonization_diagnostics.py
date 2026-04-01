"""
Before/After Harmonization Diagnostic Plots
Generates 3 figures:
  1. PCA 2x2: site coloring and group coloring, before and after
  2. R^2 distribution: variance explained by site per feature
  3. Per-site boxplot of top SAGE feature (C3_Beta3)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

OUT = 'E:/Academico/Universidad/Posgrado/Tesis/Datos/PORTABLES/Resultados/harmonization_models'

META = {'age','sex','gender','SITE','education','education_cross','diagnosis',
        'subject_id','subject','session','database','group',
        'group_sl','age_sl','SITE_sl','group_coh','age_coh','SITE_coh',
        'group_ent','age_ent','SITE_ent','group_cross','age_cross','SITE_cross'}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading data...")
pre  = pd.read_feather('E:/Academico/Universidad/Posgrado/Tesis/Datos/PORTABLES/Resultados/Data_complete_ce_roi.feather')
post = pd.read_feather('E:/Academico/Universidad/Posgrado/Tesis/Datos/PORTABLES/Resultados/Data_complete_ce_roi_HARMONIZED.feather')

feat_cols = [c for c in pre.columns if c not in META]
X_pre  = pre[feat_cols].values.astype(np.float32)
X_post = post[feat_cols].values.astype(np.float32)
sites  = pre['SITE'].values
groups = pre['group'].values

unique_sites  = sorted(set(sites))
unique_groups = ['Control', 'PSEN1', 'ADMCI']

site_cmap   = plt.cm.get_cmap('tab10', len(unique_sites))
site_colors = {s: site_cmap(i) for i, s in enumerate(unique_sites)}
group_colors  = {'Control': '#4878CF', 'PSEN1': '#D65F5F', 'ADMCI': '#6ACC65'}
group_markers = {'Control': 'o', 'PSEN1': '*', 'ADMCI': '^'}

METRIC_COLORS = {
    'power':     ('#E07B54', 'Power'),
    'sl':        ('#5B8DB8', 'Synch. Likelihood'),
    'cohfreq':   ('#6AB187', 'Coherence'),
    'entropy':   ('#C85250', 'Entropy'),
    'crossfreq': ('#9B59B6', 'Cross-freq'),
}

def get_metric(col):
    if '/' in col:           return 'crossfreq'
    if col.endswith('_sl'):  return 'sl'
    if col.endswith('_coh'): return 'cohfreq'
    if col.endswith('_ent'): return 'entropy'
    return 'power'

# ---------------------------------------------------------------------------
# PCA (fit on pre, project both)
# ---------------------------------------------------------------------------
print("Running PCA...")
scaler = StandardScaler()
X_pre_s  = scaler.fit_transform(X_pre)
X_post_s = scaler.transform(X_post)

pca = PCA(n_components=2, random_state=42)
pca.fit(X_pre_s)
pc_pre  = pca.transform(X_pre_s)
pc_post = pca.transform(X_post_s)
var_exp = pca.explained_variance_ratio_ * 100

# ---------------------------------------------------------------------------
# R^2 of site regression per feature
# ---------------------------------------------------------------------------
print("Computing R^2 before harmonization...")
le  = LabelEncoder()
ohe = OneHotEncoder(sparse_output=False)
S   = ohe.fit_transform(le.fit_transform(sites).reshape(-1, 1))

def batch_r2(X):
    preds = S @ np.linalg.lstsq(S, X, rcond=None)[0]
    ss_res = np.sum((X - preds)**2, axis=0)
    ss_tot = np.sum((X - X.mean(axis=0))**2, axis=0)
    r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)
    return np.clip(r2, 0, 1)

r2_pre  = batch_r2(X_pre)
print("Computing R^2 after harmonization...")
r2_post = batch_r2(X_post)
print(f"  Mean R^2 before: {r2_pre.mean():.4f}  |  after: {r2_post.mean():.4f}")
print(f"  Median R^2 before: {np.median(r2_pre):.4f}  |  after: {np.median(r2_post):.4f}")

# ===========================================================================
# FIGURE 1 — PCA 2x2
# ===========================================================================
print("Generating PCA 2x2 figure...")
fig = plt.figure(figsize=(16, 13))
fig.suptitle(
    'EEG Feature Space Before and After Reference-Based ComBat Harmonization\n'
    '(PCA fit on pre-harmonization data; same projection applied to both)',
    fontsize=13, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
panels = [
    (pc_pre,  'Before — colored by Site',  'site'),
    (pc_post, 'After — colored by Site',   'site'),
    (pc_pre,  'Before — colored by Group', 'group'),
    (pc_post, 'After — colored by Group',  'group'),
]

for ax, (pc, title, mode) in zip(axes, panels):
    if mode == 'site':
        for site in unique_sites:
            mask = sites == site
            ax.scatter(pc[mask, 0], pc[mask, 1],
                       color=site_colors[site], s=5, alpha=0.3,
                       label=site, rasterized=True)
        ax.legend(fontsize=6, loc='upper right', framealpha=0.7,
                  markerscale=2, ncol=2)
    else:
        for grp in unique_groups:
            mask = groups == grp
            if not mask.any():
                continue
            ax.scatter(pc[mask, 0], pc[mask, 1],
                       color=group_colors[grp],
                       marker=group_markers[grp],
                       s=18 if grp == 'PSEN1' else 5,
                       alpha=0.7 if grp == 'PSEN1' else 0.3,
                       label=grp, rasterized=True)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.7, markerscale=2)

    ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2)

fig.savefig(f'{OUT}/harmonization_pca_before_after.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: harmonization_pca_before_after.png")

# ===========================================================================
# FIGURE 2 — R^2 distribution
# ===========================================================================
print("Generating R^2 figure...")
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle(
    'Site Effect per Feature (R\u00b2) \u2014 Before vs After Harmonization',
    fontsize=13, fontweight='bold')

# Left: histogram
ax = axes2[0]
bins = np.linspace(0, max(r2_pre.max(), r2_post.max()) + 0.02, 50)
ax.hist(r2_pre,  bins=bins, alpha=0.65, color='#D65F5F',
        label=f'Before  mean={r2_pre.mean():.3f}, median={np.median(r2_pre):.3f}')
ax.hist(r2_post, bins=bins, alpha=0.65, color='#4878CF',
        label=f'After   mean={r2_post.mean():.3f}, median={np.median(r2_post):.3f}')
ax.axvline(r2_pre.mean(),  color='#D65F5F', linestyle='--', linewidth=1.8)
ax.axvline(r2_post.mean(), color='#4878CF', linestyle='--', linewidth=1.8)
ax.set_xlabel('R\u00b2 (variance explained by site)', fontsize=11)
ax.set_ylabel('Number of features', fontsize=11)
ax.set_title('Distribution across 544 features', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: scatter per feature colored by metric type
ax2 = axes2[1]
for metric, (color, label) in METRIC_COLORS.items():
    idx = [i for i, c in enumerate(feat_cols) if get_metric(c) == metric]
    ax2.scatter(r2_pre[idx], r2_post[idx],
                color=color, s=10, alpha=0.5, label=label, rasterized=True)
lim = max(r2_pre.max(), r2_post.max()) + 0.02
ax2.plot([0, lim], [0, lim], 'k--', linewidth=1.2, alpha=0.6, label='No change')
ax2.set_xlabel('R\u00b2 Before harmonization', fontsize=11)
ax2.set_ylabel('R\u00b2 After harmonization', fontsize=11)
ax2.set_title('Per-feature R\u00b2 (below diagonal = reduced site effect)', fontsize=11)
ax2.legend(fontsize=8, framealpha=0.7)
ax2.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(f'{OUT}/harmonization_site_r2_before_after.png', dpi=200, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved: harmonization_site_r2_before_after.png")

# ===========================================================================
# FIGURE 3 — Per-site boxplot of C3_Beta3
# ===========================================================================
print("Generating per-site distribution figure...")
top_feat = 'C3_Beta3'
fi = feat_cols.index(top_feat)

fig3, axes3 = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
fig3.suptitle(
    f'Per-Site Distribution of {top_feat} (top SAGE feature) \u2014 Before vs After',
    fontsize=13, fontweight='bold')

for ax3, (X_data, label) in zip(axes3, [(X_pre, 'Before harmonization'),
                                          (X_post, 'After harmonization')]):
    vals_by_site = [X_data[sites == s, fi] for s in unique_sites]
    bp = ax3.boxplot(vals_by_site,
                     patch_artist=True,
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(linewidth=1),
                     flierprops=dict(marker='.', markersize=2, alpha=0.3))
    for patch, site in zip(bp['boxes'], unique_sites):
        patch.set_facecolor(site_colors[site])
        patch.set_alpha(0.75)
    ax3.set_xticks(range(1, len(unique_sites)+1))
    ax3.set_xticklabels(unique_sites, rotation=38, ha='right', fontsize=9)
    ax3.set_ylabel('Feature value (log scale)', fontsize=10)
    ax3.set_title(label, fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.2, axis='y')

fig3.tight_layout()
fig3.savefig(f'{OUT}/harmonization_feature_distribution_before_after.png',
             dpi=200, bbox_inches='tight')
plt.close(fig3)
print(f"  Saved: harmonization_feature_distribution_before_after.png")

print("\nAll plots generated successfully.")
print(f"R^2 reduction: {r2_pre.mean():.3f} -> {r2_post.mean():.3f} "
      f"({(1 - r2_post.mean()/r2_pre.mean())*100:.1f}% reduction in mean site variance)")
