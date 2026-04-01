"""
Propensity Score Matching (PSM) Module - SUBJECT-LEVEL MATCHING
Implements proper Nearest Neighbor Matching at SUBJECT level (not row level)
Support for 1:1, 2:1, 5:1, and 10:1 ratios
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import linear_sum_assignment


def calculate_smd(treatment, control, variable_name=''):
    """
    Calculate Standardized Mean Difference (SMD)
    
    SMD = (mean_treatment - mean_control) / pooled_SD
    
    Interpretation:
    - SMD < 0.1: Excellent balance
    - SMD < 0.25: Acceptable balance
    - SMD >= 0.25: Poor balance (unacceptable)
    """
    mean_t = np.mean(treatment)
    mean_c = np.mean(control)
    
    var_t = np.var(treatment, ddof=1)
    var_c = np.var(control, ddof=1)
    
    # Pooled standard deviation
    pooled_sd = np.sqrt((var_t + var_c) / 2)
    
    if pooled_sd == 0:
        return 0.0
    
    smd = (mean_t - mean_c) / pooled_sd
    
    return smd


def get_subject_demographics(data):
    """
    Extract one row per subject with demographics
    
    Parameters:
    -----------
    data : DataFrame
        Full dataset with multiple rows per subject
    
    Returns:
    --------
    DataFrame : One row per subject with demographics
    """
    # Get demographic columns
    demo_cols = ['subject', 'group', 'age', 'sex', 'education']
    
    # Handle different column name variations
    if 'gender' in data.columns and 'sex' not in data.columns:
        demo_cols[demo_cols.index('sex')] = 'gender'
    
    # Filter to only existing columns
    demo_cols = [col for col in demo_cols if col in data.columns]
    
    # Get unique subjects (take first occurrence of each subject)
    subject_data = data[demo_cols].drop_duplicates(subset=['subject'], keep='first')
    
    return subject_data.reset_index(drop=True)


def validate_balance(matched_subjects, group1, group2, matching_features):
    """
    Validate demographic balance after matching using SMD
    AT SUBJECT LEVEL
    
    Parameters:
    -----------
    matched_subjects : DataFrame
        Matched subjects (one row per subject)
    group1 : str
        Treatment group name
    group2 : str
        Control group name
    matching_features : list
        Features used in matching
    
    Returns:
    --------
    dict : Balance metrics including SMD for each variable
    """
    g1_data = matched_subjects[matched_subjects['group'] == group1]
    g2_data = matched_subjects[matched_subjects['group'] == group2]
    
    balance_results = {
        'n_treatment': len(g1_data),
        'n_control': len(g2_data),
        'balance_quality': 'Unknown'
    }
    
    smds = {}
    
    # Calculate SMD for each matching feature
    for feature in matching_features:
        if feature in matched_subjects.columns:
            # Handle categorical (sex/gender) separately
            if feature in ['sex', 'gender']:
                # Calculate proportion male in each group
                g1_vals = g1_data[feature]
                g2_vals = g2_data[feature]
                
                g1_male_prop = (g1_vals.isin(['M', 'Male', 1, '1', 1.0])).mean()
                g2_male_prop = (g2_vals.isin(['M', 'Male', 1, '1', 1.0])).mean()
                
                # For binary variables, SMD formula is different
                pooled_p = (g1_male_prop + g2_male_prop) / 2
                if pooled_p > 0 and pooled_p < 1:
                    smd_sex = (g1_male_prop - g2_male_prop) / np.sqrt(pooled_p * (1 - pooled_p))
                else:
                    smd_sex = 0.0
                
                smds[feature] = smd_sex
                balance_results[f'{feature}_prop_male_g1'] = g1_male_prop
                balance_results[f'{feature}_prop_male_g2'] = g2_male_prop
            else:
                # Continuous variable (age, education, etc.)
                g1_vals = g1_data[feature].dropna()
                g2_vals = g2_data[feature].dropna()
                
                if len(g1_vals) > 0 and len(g2_vals) > 0:
                    smd = calculate_smd(g1_vals, g2_vals, feature)
                    smds[feature] = smd
                    
                    balance_results[f'{feature}_mean_g1'] = g1_vals.mean()
                    balance_results[f'{feature}_mean_g2'] = g2_vals.mean()
                    balance_results[f'{feature}_sd_g1'] = g1_vals.std()
                    balance_results[f'{feature}_sd_g2'] = g2_vals.std()
    
    balance_results['smds'] = smds
    
    # Determine overall balance quality
    max_smd = max([abs(v) for v in smds.values()]) if smds else 0
    
    if max_smd < 0.1:
        balance_results['balance_quality'] = 'EXCELLENT'
    elif max_smd < 0.25:
        balance_results['balance_quality'] = 'ACCEPTABLE'
    else:
        balance_results['balance_quality'] = 'POOR'
    
    balance_results['max_smd'] = max_smd
    
    return balance_results


def match_and_optimize(data, group1, group2, ratio_str='2:1', caliper=0.2,
                       matching_covariates=None):
    """
    Perform propensity score matching AT SUBJECT LEVEL.

    Parameters
    ----------
    data : DataFrame
        Input dataframe with EEG data (multiple rows per subject).
    group1 : str
        Name of treatment group.
    group2 : str or list
        Name(s) of control group.
    ratio_str : str
        Matching ratio: '1:1', '2:1', '5:1', or '10:1'.
    caliper : float
        Maximum distance in propensity score (in SD units).
    matching_covariates : list or None
        Columns to use for propensity score (e.g. ['age'] or ['age', 'education']).
        None -> auto-detect all available demographic columns.

    Returns
    -------
    DataFrame : Matched dataset (all rows for matched subjects).
    """
    print(f"\n{'='*60}")
    print(f"PROPENSITY SCORE MATCHING: {ratio_str}")
    print(f"Treatment: {group1} | Control: {group2}")
    print(f"Method: Nearest Neighbor WITHOUT replacement (SUBJECT-LEVEL)")
    print(f"Caliper: {caliper} SD")
    print(f"{'='*60}\n")
    
    # CRITICAL: Extract subject-level data
    print("Extracting subject-level demographics...")
    subject_data = get_subject_demographics(data.copy())
    
    print(f"  Total unique subjects: {len(subject_data)}")
    print(f"  Total rows in original data: {len(data)}")
    
    # Handle multiple control groups
    if isinstance(group2, list):
        print(f"Combining control groups: {group2}")
        subject_data['group'] = subject_data['group'].apply(
            lambda x: 'Control' if x in group2 else x
        )
        group2 = 'Control'
    
    # Parse ratio
    ratio_multiplier = parse_ratio(ratio_str)
    
    # Create treatment indicator
    subject_data['treat_indicator'] = (subject_data['group'] == group1).astype(int)
    
    # Separate groups AT SUBJECT LEVEL
    g1_subjects = subject_data[subject_data['group'] == group1].copy()
    g2_subjects = subject_data[subject_data['group'] == group2].copy()
    
    print(f"\nBefore matching (SUBJECTS):")
    print(f"  {group1}: {len(g1_subjects)} subjects")
    print(f"  {group2}: {len(g2_subjects)} subjects")
    
    # Check if matching is possible
    max_possible_pairs = min(len(g1_subjects), len(g2_subjects) // ratio_multiplier)
    print(f"\nMaximum possible matches:")
    print(f"  {max_possible_pairs} {group1} subjects x (1 + {ratio_multiplier} controls)")
    print(f"  = {max_possible_pairs * (1 + ratio_multiplier)} total subjects")
    
    # Calculate propensity scores AT SUBJECT LEVEL
    prop_scores, matching_features = calculate_propensity_scores(
        subject_data, matching_covariates=matching_covariates
    )
    
    if prop_scores is None:
        print("\n[!] Propensity score calculation failed. Using random sampling.")
        return simple_sampling_subjects(data, subject_data, group1, group2, ratio_multiplier)
    
    # Assign propensity scores
    subject_data['propensity_score'] = prop_scores
    g1_subjects = subject_data[subject_data['group'] == group1].copy()
    g2_subjects = subject_data[subject_data['group'] == group2].copy()
    
    g1_prop_scores = g1_subjects['propensity_score'].values
    g2_prop_scores = g2_subjects['propensity_score'].values
    
    # Visualize propensity scores
    plot_propensity_scores(g1_prop_scores, g2_prop_scores, ratio_str, group1)
    
    # PERFORM MATCHING AT SUBJECT LEVEL
    matched_subject_list = perform_nn_matching_subjects(
        g1_subjects, 
        g2_subjects, 
        g1_prop_scores, 
        g2_prop_scores,
        group1, 
        group2, 
        ratio_multiplier,
        caliper
    )
    
    if len(matched_subject_list) == 0:
        print("\n[X] ERROR: Matching failed completely!")
        return pd.DataFrame(columns=data.columns)
    
    # CRITICAL: Get all rows for matched subjects
    print(f"\nRetrieving all rows for matched subjects...")
    matched_data = data[data['subject'].isin(matched_subject_list)].copy()
    
    print(f"  Matched subjects: {len(matched_subject_list)}")
    print(f"  Matched rows: {len(matched_data)}")
    
    # Validate balance AT SUBJECT LEVEL
    matched_subjects = subject_data[subject_data['subject'].isin(matched_subject_list)]
    
    print("\n" + "="*60)
    print("BALANCE VALIDATION (SMD) - SUBJECT LEVEL")
    print("="*60)
    
    balance_results = validate_balance(matched_subjects, group1, group2, matching_features)
    
    print(f"\nOverall balance quality: {balance_results['balance_quality']}")
    print(f"Maximum SMD: {balance_results['max_smd']:.3f}")
    print(f"\nSMD by variable:")
    for var, smd in balance_results['smds'].items():
        status = "[OK]" if abs(smd) < 0.1 else ("[!]" if abs(smd) < 0.25 else "[X]")
        print(f"  {status} {var}: {smd:.3f}")
    
    # Detailed age comparison
    if 'age' in matching_features:
        print(f"\nAge comparison:")
        print(f"  {group1}: {balance_results['age_mean_g1']:.1f} ± {balance_results['age_sd_g1']:.1f}")
        print(f"  {group2}: {balance_results['age_mean_g2']:.1f} ± {balance_results['age_sd_g2']:.1f}")
        print(f"  Difference: {abs(balance_results['age_mean_g1'] - balance_results['age_mean_g2']):.1f} years")
    
    # Warning if poor balance
    if balance_results['balance_quality'] == 'POOR':
        print(f"\n[!] WARNING: Poor balance detected (SMD >= 0.25)")
        print(f"  Consider:")
        print(f"  1. Using a smaller ratio (e.g., 1:1 instead of {ratio_str})")
        print(f"  2. Increasing caliper (currently {caliper})")
        print(f"  3. Checking if sufficient overlap in covariate distributions")
    
    # Clean up
    matched_data = matched_data.drop(columns=['treat_indicator', 'propensity_score'], errors='ignore')
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"MATCHING SUMMARY - {ratio_str}")
    print(f"{'='*60}")
    
    # Count unique subjects in matched data
    matched_g1_subjects = matched_data[matched_data['group'] == group1]['subject'].nunique()
    matched_g2_subjects = matched_data[matched_data['group'] == group2]['subject'].nunique()
    
    print(f"\nSubjects:")
    print(f"  Before matching:")
    print(f"    {group1}: {len(g1_subjects)} subjects")
    print(f"    {group2}: {len(g2_subjects)} subjects")
    
    print(f"\n  After matching:")
    print(f"    {group1}: {matched_g1_subjects} subjects")
    print(f"    {group2}: {matched_g2_subjects} subjects")
    print(f"    Total: {matched_g1_subjects + matched_g2_subjects} subjects")
    
    if matched_g1_subjects > 0:
        actual_ratio = matched_g2_subjects / matched_g1_subjects
        print(f"    Ratio achieved: 1:{actual_ratio:.2f}")
    
    print(f"\nRows (features):")
    print(f"  Total matched rows: {len(matched_data)}")
    
    print(f"\n  Subjects lost:")
    print(f"    {group1}: {len(g1_subjects) - matched_g1_subjects} ({(len(g1_subjects) - matched_g1_subjects)/len(g1_subjects)*100:.1f}%)")
    print(f"    {group2}: {len(g2_subjects) - matched_g2_subjects} ({(len(g2_subjects) - matched_g2_subjects)/len(g2_subjects)*100:.1f}%)")
    
    print(f"{'='*60}\n")
    
    return matched_data.reset_index(drop=True)


def parse_ratio(ratio_str):
    """Parse ratio string to multiplier"""
    ratio_map = {'1:1': 1, '2:1': 2, '5:1': 5, '10:1': 10}
    multiplier = ratio_map.get(ratio_str)
    
    if multiplier is None:
        print(f"[!] Warning: Unknown ratio '{ratio_str}'. Defaulting to 2:1")
        return 2
    
    return multiplier


def calculate_propensity_scores(subject_data, matching_covariates=None):
    """
    Calculate propensity scores using logistic regression AT SUBJECT LEVEL.

    Parameters
    ----------
    subject_data : DataFrame
        One row per subject with demographic columns.
    matching_covariates : list or None
        Explicit list of column names to use (e.g. ['age'] or ['age', 'education']).
        If None, all available demographic columns are used (auto-detect: age, sex/gender,
        education).  Columns not present in subject_data are silently skipped.

    Returns
    -------
    tuple : (propensity_scores, matching_features_used)
    """
    if matching_covariates is not None:
        # Use only explicitly requested columns that are actually present
        matching_features = [
            c for c in matching_covariates
            if c in subject_data.columns and subject_data[c].notna().sum() > 0
        ]
        skipped = [c for c in matching_covariates if c not in matching_features]
        if skipped:
            print(f"  [!] Covariables solicitadas no disponibles/vacías: {skipped}")
    else:
        # Auto-detect (legacy behaviour)
        matching_features = []

        if 'age' in subject_data.columns and subject_data['age'].notna().sum() > 0:
            matching_features.append('age')

        sex_col = None
        if 'sex' in subject_data.columns and subject_data['sex'].notna().sum() > 0:
            sex_col = 'sex'
            matching_features.append('sex')
        elif 'gender' in subject_data.columns and subject_data['gender'].notna().sum() > 0:
            sex_col = 'gender'
            matching_features.append('gender')

        if 'education' in subject_data.columns and subject_data['education'].notna().sum() > 0:
            matching_features.append('education')

    if len(matching_features) == 0:
        print("[!] No matching features available")
        return None, []

    # Prepare data
    X = subject_data[matching_features].copy()

    # Encode categorical sex/gender if still as strings
    for sex_col in ['sex', 'gender']:
        if sex_col in X.columns and X[sex_col].dtype == object:
            X[sex_col] = X[sex_col].map({'F': 0, 'Female': 0, 'M': 1, 'Male': 1})

    # Fill remaining missing values with column median
    X = X.fillna(X.median())

    y = subject_data['treat_indicator']

    try:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        prop_scores = model.predict_proba(X)[:, 1]
        print(f"[OK] Propensity scores calculados usando: {matching_features}")
        return prop_scores, matching_features
    except Exception as e:
        print(f"[X] Error calculando propensity scores: {e}")
        return None, []


def perform_nn_matching_subjects(g1_subjects, g2_subjects, g1_prop_scores, g2_prop_scores,
                                  group1, group2, ratio_multiplier, caliper=0.2):
    """
    SUBJECT-LEVEL MATCHING: Optimal 1:1 or constrained-greedy k:1, WITHOUT replacement.

    For 1:1  — uses linear_sum_assignment (globally optimal: maximises matched
               pairs and minimises total PS distance simultaneously).

    For k:1  — uses greedy nearest-neighbour with "most constrained first"
               ordering: treatment subjects with the fewest controls in their
               caliper window are processed first, preventing the most restricted
               subjects from being left without controls by less-constrained ones.
               The linear_sum_assignment objective (minimise total cost) does not
               maximise the number of treatments that receive exactly k controls,
               so greedy with optimal ordering outperforms it for k > 1.

    Parameters
    ----------
    g1_subjects : DataFrame  (one row per treatment subject)
    g2_subjects : DataFrame  (one row per control subject)
    g1_prop_scores : array
    g2_prop_scores : array
    group1 : str
    group2 : str
    ratio_multiplier : int   controls per treatment subject (k)
    caliper : float          max PS distance in SD units

    Returns
    -------
    list : subject IDs of all matched subjects (treatment + control)
    """
    g1_subjects = g1_subjects.reset_index(drop=True)
    g2_subjects = g2_subjects.reset_index(drop=True)

    all_scores = np.concatenate([g1_prop_scores, g2_prop_scores])
    caliper_absolute = caliper * np.std(all_scores)

    n_treat = len(g1_subjects)
    n_ctrl  = len(g2_subjects)

    print(f"  Caliper: {caliper} SD = {caliper_absolute:.4f} in PS units")
    print(f"  Target ratio: 1:{ratio_multiplier}")

    # ------------------------------------------------------------------
    # 1:1  — globally optimal via linear_sum_assignment
    # ------------------------------------------------------------------
    if ratio_multiplier == 1:
        print(f"\nPerforming Optimal 1:1 Matching (linear_sum_assignment)...")

        PENALTY = 1e9
        cost_matrix = np.abs(g1_prop_scores[:, None] - g2_prop_scores[None, :])
        cost_matrix[cost_matrix > caliper_absolute] = PENALTY

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs = [
            (r, [c]) for r, c in zip(row_ind, col_ind)
            if cost_matrix[r, c] < PENALTY
        ]

    # ------------------------------------------------------------------
    # k:1  — greedy nearest-neighbour, most-constrained-first ordering
    # ------------------------------------------------------------------
    else:
        print(f"\nPerforming Greedy k:1 Matching (most-constrained-first)...")

        # Count controls within caliper for each treatment — process most
        # restricted subjects first so they are not crowded out
        n_in_caliper = np.array([
            np.sum(np.abs(g2_prop_scores - g1_prop_scores[i]) <= caliper_absolute)
            for i in range(n_treat)
        ])
        processing_order = np.argsort(n_in_caliper)
        print(f"  Controls in caliper per treatment: "
              f"min={n_in_caliper.min():.0f}, max={n_in_caliper.max():.0f}")

        available = set(range(n_ctrl))
        matched_pairs = []

        for i in processing_order:
            treat_ps = g1_prop_scores[i]
            ctrl_idx  = np.array(sorted(available))
            dists     = np.abs(g2_prop_scores[ctrl_idx] - treat_ps)
            order     = np.argsort(dists)
            selected  = [ctrl_idx[j] for j in order
                         if dists[j] <= caliper_absolute][:ratio_multiplier]

            if len(selected) == ratio_multiplier:
                matched_pairs.append((i, selected))
                available -= set(selected)

    # ------------------------------------------------------------------
    # Common reporting and output
    # ------------------------------------------------------------------
    n_matched   = len(matched_pairs)
    n_failed    = n_treat - n_matched
    n_ctrl_used = n_matched * ratio_multiplier

    print(f"\n  Matching results:")
    print(f"    Treatments successfully matched: {n_matched}")
    print(f"    Treatments failed (insufficient controls in caliper): {n_failed}")
    print(f"    Controls used: {n_ctrl_used}")
    print(f"    Controls remaining: {n_ctrl - n_ctrl_used}")

    if n_matched == 0:
        print(f"\n  [X] ERROR: No matches found within caliper!")
        print(f"  Suggestions:")
        print(f"    1. Increase caliper (current: {caliper})")
        print(f"    2. Use smaller ratio")
        print(f"    3. Check propensity score overlap")
        return []

    matched_treatment_subjects = [g1_subjects.iloc[i]['subject'] for i, _ in matched_pairs]
    matched_control_subjects   = [g2_subjects.iloc[c]['subject']
                                   for _, ctrls in matched_pairs for c in ctrls]
    all_matched_subjects = matched_treatment_subjects + matched_control_subjects

    assert len(matched_control_subjects) == n_matched * ratio_multiplier, \
        "Control count mismatch!"
    assert len(set(all_matched_subjects)) == len(all_matched_subjects), \
        "Duplicate subjects detected!"

    print(f"\n  [OK] Matched subjects:")
    print(f"    {group1}: {n_matched} subjects")
    print(f"    {group2}: {len(matched_control_subjects)} subjects")
    print(f"    Total: {len(all_matched_subjects)} subjects")
    print(f"    Actual ratio: 1:{len(matched_control_subjects)/n_matched:.2f}")

    return all_matched_subjects


def simple_sampling_subjects(data, subject_data, group1, group2, ratio_multiplier):
    """Simple random sampling fallback AT SUBJECT LEVEL"""
    g1_subjects = subject_data[subject_data['group'] == group1].copy()
    g2_subjects = subject_data[subject_data['group'] == group2].copy()
    
    max_g1 = len(g1_subjects)
    max_g2_for_g1 = len(g2_subjects) // ratio_multiplier
    
    target_g1 = min(max_g1, max_g2_for_g1)
    target_g2 = target_g1 * ratio_multiplier
    
    g1_sampled = g1_subjects.sample(n=target_g1, random_state=42)
    g2_sampled = g2_subjects.sample(n=target_g2, random_state=42)
    
    matched_subjects = pd.concat([g1_sampled, g2_sampled])['subject'].tolist()
    
    # Get all rows for these subjects
    sampled_data = data[data['subject'].isin(matched_subjects)]
    
    print(f"\nRandom sampling results:")
    print(f"  {group1}: {len(g1_sampled)} subjects")
    print(f"  {group2}: {len(g2_sampled)} subjects")
    print(f"  Total rows: {len(sampled_data)}")
    
    return sampled_data


def plot_propensity_scores(g1_prop_scores, g2_prop_scores, ratio_str, group1):
    """Plot propensity score distribution"""
    try:
        plt.figure(figsize=(12, 6))
        plt.rc('font', size=14)
        
        all_scores = np.concatenate([g1_prop_scores, g2_prop_scores])
        bins = np.linspace(all_scores.min(), all_scores.max(), 30)
        
        g1_hist, _ = np.histogram(g1_prop_scores, bins=bins)
        g2_hist, _ = np.histogram(g2_prop_scores, bins=bins)
        
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        bar_width = (bins[1] - bins[0]) * 0.9
        
        plt.bar(bin_centers, g1_hist, width=bar_width, color='salmon', 
               label=f'{group1} (Treatment)', alpha=0.8)
        plt.bar(bin_centers, -g2_hist, width=bar_width, color='slateblue', 
               label='Control', alpha=0.8)
        
        plt.axhline(0, color='black', linewidth=0.8)
        
        plt.xlabel('Propensity Score')
        plt.ylabel('Frequency')
        plt.title(f'Common Support - {ratio_str} Matching (Subject Level)')
        plt.legend(title='Group')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('propensity_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Propensity score plot saved: propensity_scores.png")
    except Exception as e:
        print(f"[!] Could not create propensity score plot: {e}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Propensity Score Matching Module - SUBJECT-LEVEL MATCHING")
    print("=" * 60)
    print("\nKey features:")
    print("  [OK] Matching at SUBJECT level (not row level)")
    print("  [OK] Nearest Neighbor matching WITHOUT replacement")
    print("  [OK] SMD validation for balance assessment")
    print("  [OK] Caliper filtering for quality control")
    print("  [OK] Returns all rows for matched subjects")
    print("\nSupported ratios:")
    print("  - 1:1  (1 control per treatment)")
    print("  - 2:1  (2 controls per treatment)")
    print("  - 5:1  (5 controls per treatment)")
    print("  - 10:1 (10 controls per treatment)")
    print("\nUsage:")
    print("  matched_data = match_and_optimize(data, 'PSEN1', 'Control', '2:1')")
