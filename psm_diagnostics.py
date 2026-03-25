"""
PSM Diagnostic Tool
Evaluates quality of propensity score matching
Generates comprehensive balance reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def normalize_sex(series):
    """Normalize sex variable to binary: Male=1, Female=0"""
    return series.map({
        'M': 1, 'Male': 1, 'male': 1, 1: 1, '1': 1, 1.0: 1,
        'F': 0, 'Female': 0, 'female': 0, 0: 0, '0': 0, 0.0: 0
    })


def compute_smd_continuous(x1, x2):
    pooled_sd = np.sqrt((x1.var(ddof=1) + x2.var(ddof=1)) / 2)
    return (x1.mean() - x2.mean()) / pooled_sd if pooled_sd > 0 else 0


def compute_smd_binary(p1, p2):
    pooled_p = (p1 + p2) / 2
    return (p1 - p2) / np.sqrt(pooled_p * (1 - pooled_p)) if 0 < pooled_p < 1 else 0


def get_balance_status(smd):
    if smd < 0.1:
        return "[OK] Excellent"
    elif smd < 0.25:
        return "[!] Acceptable"
    else:
        return "[X] Poor"


# --------------------------------------------------
# Balance table
# --------------------------------------------------

def create_balance_table(data, group_col='group', group1='PSEN1', group2='Control'):

    g1 = data[data[group_col] == group1]
    g2 = data[data[group_col] == group2]

    g1 = g1.drop_duplicates(subset='subject')
    g2 = g2.drop_duplicates(subset='subject')

    rows = []
    continuous_vars = ['age', 'education', 'ses', 'mmse', 'moca']

    # ---------- Continuous ----------
    for var in continuous_vars:
        if var not in data.columns:
            continue

        x1 = g1[var].dropna()
        x2 = g2[var].dropna()

        if len(x1) == 0 or len(x2) == 0:
            continue

        smd = abs(compute_smd_continuous(x1, x2))
        _, p = stats.ttest_ind(x1, x2, equal_var=False)

        rows.append({
            'Variable': var.capitalize(),
            f'{group1}_Mean': f"{x1.mean():.1f}",
            f'{group1}_SD': f"{x1.std():.1f}",
            f'{group2}_Mean': f"{x2.mean():.1f}",
            f'{group2}_SD': f"{x2.std():.1f}",
            'SMD': f"{smd:.3f}",
            'P_value': f"{p:.3f}",
            'Balance': get_balance_status(smd)
        })

    # ---------- Sex ----------
    if 'sex' in data.columns:
        g1_sex = normalize_sex(g1['sex']).dropna()
        g2_sex = normalize_sex(g2['sex']).dropna()

        g1_m = (g1_sex == 1).sum()
        g1_f = (g1_sex == 0).sum()
        g2_m = (g2_sex == 1).sum()
        g2_f = (g2_sex == 0).sum()

        p1 = g1_m / (g1_m + g1_f) if (g1_m + g1_f) > 0 else 0
        p2 = g2_m / (g2_m + g2_f) if (g2_m + g2_f) > 0 else 0

        smd = abs(compute_smd_binary(p1, p2))
        chi2, p, _, _ = stats.chi2_contingency([[g1_m, g1_f], [g2_m, g2_f]])

        rows.append({
            'Variable': 'Sex (M:F)',
            f'{group1}_Mean': f"{g1_m}:{g1_f}",
            f'{group1}_SD': f"{p1*100:.1f}% M",
            f'{group2}_Mean': f"{g2_m}:{g2_f}",
            f'{group2}_SD': f"{p2*100:.1f}% M",
            'SMD': f"{smd:.3f}",
            'P_value': f"{p:.3f}",
            'Balance': get_balance_status(smd)
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Plots
# --------------------------------------------------
def plot_balance_diagnostics(
    data_before,
    data_after,
    group_col='group',
    group1='PSEN1',
    group2='Control',
    output_path='balance_diagnostics.png'
):
    """
    Create comprehensive balance diagnostic plots

    Parameters:
    -----------
    data_before : DataFrame
        Data before matching
    data_after : DataFrame
        Data after matching
    group_col : str
        Group column name
    group1 : str
        Treatment group
    group2 : str
        Control group
    output_path : str
        Path to save plot
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        'Propensity Score Matching Balance Diagnostics',
        fontsize=16,
        fontweight='bold'
    )

    # Define color palette
    colors = {group1: '#E74C3C', group2: '#3498DB'}

    # 1. Age distribution - Before
    ax = axes[0, 0]
    if 'age' in data_before.columns:
        for group in [group1, group2]:
            group_data = data_before[data_before[group_col] == group]['age'].dropna()
            ax.hist(
                group_data,
                alpha=0.6,
                bins=20,
                label=group,
                color=colors[group]
            )
        ax.set_title('Age Distribution - Before Matching', fontweight='bold')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)

    # 2. Age distribution - After
    ax = axes[0, 1]
    if 'age' in data_after.columns:
        for group in [group1, group2]:
            group_data = data_after[data_after[group_col] == group]['age'].dropna()
            ax.hist(
                group_data,
                alpha=0.6,
                bins=20,
                label=group,
                color=colors[group]
            )
        ax.set_title('Age Distribution - After Matching', fontweight='bold')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)

    # 3. Age boxplot comparison
    ax = axes[0, 2]
    if 'age' in data_before.columns and 'age' in data_after.columns:
        data_combined = pd.DataFrame({
            'Before_' + group1: data_before[data_before[group_col] == group1]['age'].dropna(),
            'Before_' + group2: data_before[data_before[group_col] == group2]['age'].dropna(),
            'After_' + group1: data_after[data_after[group_col] == group1]['age'].dropna(),
            'After_' + group2: data_after[data_after[group_col] == group2]['age'].dropna()
        })

        bp = ax.boxplot(
            [data_combined[col].dropna() for col in data_combined.columns],
            labels=[
                'Before\n' + group1,
                'Before\n' + group2,
                'After\n' + group1,
                'After\n' + group2
            ],
            patch_artist=True
        )

        # Color boxes
        for patch, color in zip(
            bp['boxes'],
            [colors[group1], colors[group2], colors[group1], colors[group2]]
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title('Age Balance Comparison', fontweight='bold')
        ax.set_ylabel('Age (years)')
        ax.grid(alpha=0.3, axis='y')

    # 4. Sex distribution - Before
    ax = axes[1, 0]
    if 'sex' in data_before.columns:
        sex_counts_before = []

        for group in [group1, group2]:
            group_data = data_before[data_before[group_col] == group]
            male_count = group_data['sex'].isin(['M', 'Male', 1, '1', 1.0]).sum()
            female_count = group_data['sex'].isin(['F', 'Female', 0, '0', 0.0]).sum()
            sex_counts_before.append([male_count, female_count])

        x = np.arange(2)
        width = 0.35

        ax.bar(
            x - width / 2,
            [sex_counts_before[0][0], sex_counts_before[0][1]],
            width,
            label=group1,
            color=colors[group1],
            alpha=0.8
        )
        ax.bar(
            x + width / 2,
            [sex_counts_before[1][0], sex_counts_before[1][1]],
            width,
            label=group2,
            color=colors[group2],
            alpha=0.8
        )

        ax.set_title('Sex Distribution - Before Matching', fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(['Male', 'Female'])
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

    # 5. Sex distribution - After
    ax = axes[1, 1]
    if 'sex' in data_after.columns:
        sex_counts_after = []

        for group in [group1, group2]:
            group_data = data_after[data_after[group_col] == group]
            male_count = group_data['sex'].isin(['M', 'Male', 1, '1', 1.0]).sum()
            female_count = group_data['sex'].isin(['F', 'Female', 0, '0', 0.0]).sum()
            sex_counts_after.append([male_count, female_count])

        x = np.arange(2)
        width = 0.35

        ax.bar(
            x - width / 2,
            [sex_counts_after[0][0], sex_counts_after[0][1]],
            width,
            label=group1,
            color=colors[group1],
            alpha=0.8
        )
        ax.bar(
            x + width / 2,
            [sex_counts_after[1][0], sex_counts_after[1][1]],
            width,
            label=group2,
            color=colors[group2],
            alpha=0.8
        )

        ax.set_title('Sex Distribution - After Matching', fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(['Male', 'Female'])
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

    # 6. SMD comparison plot
    ax = axes[1, 2]

    variables = ['age', 'sex', 'education', 'ses', 'mmse', 'moca']
    smds_before = []
    smds_after = []
    var_names = []

    for var in variables:
        if var in data_before.columns:
            g1_before = data_before[data_before[group_col] == group1][var].dropna()
            g2_before = data_before[data_before[group_col] == group2][var].dropna()

            if len(g1_before) > 0 and len(g2_before) > 0:
                if var == 'sex':
                    g1_prop = g1_before.isin(['M', 'Male', 1, '1', 1.0]).mean()
                    g2_prop = g2_before.isin(['M', 'Male', 1, '1', 1.0]).mean()
                    pooled_p = (g1_prop + g2_prop) / 2
                    smd_before = (
                        (g1_prop - g2_prop) / np.sqrt(pooled_p * (1 - pooled_p))
                        if 0 < pooled_p < 1 else 0
                    )
                else:
                    pooled_sd = np.sqrt((g1_before.var() + g2_before.var()) / 2)
                    smd_before = (
                        (g1_before.mean() - g2_before.mean()) / pooled_sd
                        if pooled_sd > 0 else 0
                    )

                g1_after = data_after[data_after[group_col] == group1][var].dropna()
                g2_after = data_after[data_after[group_col] == group2][var].dropna()

                if var == 'sex':
                    g1_prop = g1_after.isin(['M', 'Male', 1, '1', 1.0]).mean()
                    g2_prop = g2_after.isin(['M', 'Male', 1, '1', 1.0]).mean()
                    pooled_p = (g1_prop + g2_prop) / 2
                    smd_after = (
                        (g1_prop - g2_prop) / np.sqrt(pooled_p * (1 - pooled_p))
                        if 0 < pooled_p < 1 else 0
                    )
                else:
                    pooled_sd = np.sqrt((g1_after.var() + g2_after.var()) / 2)
                    smd_after = (
                        (g1_after.mean() - g2_after.mean()) / pooled_sd
                        if pooled_sd > 0 else 0
                    )

                smds_before.append(abs(smd_before))
                smds_after.append(abs(smd_after))
                var_names.append(var.capitalize())

    if var_names:
        x = np.arange(len(var_names))
        width = 0.35

        ax.bar(
            x - width / 2,
            smds_before,
            width,
            label='Before Matching',
            color='#E74C3C',
            alpha=0.8
        )
        ax.bar(
            x + width / 2,
            smds_after,
            width,
            label='After Matching',
            color='#27AE60',
            alpha=0.8
        )

        ax.axhline(
            y=0.1,
            color='green',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label='Excellent (<0.1)'
        )
        ax.axhline(
            y=0.25,
            color='orange',
            linestyle='--',
            linewidth=2,
            alpha=0.7,
            label='Acceptable (<0.25)'
        )

        ax.set_title('Standardized Mean Differences', fontweight='bold')
        ax.set_ylabel('|SMD|')
        ax.set_xticks(x)
        ax.set_xticklabels(var_names, rotation=45)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        ax.set_ylim(0, max(max(smds_before), 0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[OK] Balance diagnostic plots saved: {output_path}")


# --------------------------------------------------
# Main diagnostic
# --------------------------------------------------

def diagnose_psm_quality(data_before, data_after, group1='PSEN1', group2='Control',
                         output_dir='psm_diagnostics'):

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("PSM QUALITY DIAGNOSTIC REPORT")
    print("=" * 60)

    balance = create_balance_table(data_after, 'group', group1, group2)
    balance_path = os.path.join(output_dir, 'balance_table.xlsx')
    balance.to_excel(balance_path, index=False)

    print(f"\n[OK] Balance table saved: {balance_path}\n")
    print(balance.to_string(index=False))

    plot_balance_diagnostics(
        data_before, data_after, 'group',
        group1, group2,
        os.path.join(output_dir, 'balance_diagnostics.png')
    )

    max_smd = balance['SMD'].astype(float).max() if not balance.empty else 0

    print("\n" + "=" * 60)
    print("OVERALL QUALITY ASSESSMENT")
    print("=" * 60)
    print(f"Maximum SMD: {max_smd:.3f}")

    if max_smd < 0.1:
        quality = "EXCELLENT"
    elif max_smd < 0.25:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR"

    print(f"Quality: {quality}")
    print("=" * 60)

    return {
        'max_smd': max_smd,
        'quality': quality,
        'balance_table': balance
    }

