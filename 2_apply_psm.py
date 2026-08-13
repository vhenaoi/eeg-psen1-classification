"""
PSM Application Script - SUBJECT-LEVEL MATCHING
Apply Propensity Score Matching at subject level with balance validation
"""

import os
import pandas as pd
from pairs import match_and_optimize
from psm_diagnostics import diagnose_psm_quality


class PSMConfig:
    """Configuration for PSM pipeline"""
    
    def __init__(self, base_path, data_type='ce', space='roi'):
        self.base_path = base_path
        self.data_type = data_type
        self.space = space
        
        # Setup paths
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup directory structure and handle HARMONIZED files"""
        # Construir la ruta con HARMONIZED
        harmonized_file = os.path.join(
            self.base_path,
            f'Data_complete_{self.data_type}_{self.space}_HARMONIZED.feather'
        )
        
        # Construir la ruta normal
        normal_file = os.path.join(
            self.base_path,
            f'Data_complete_{self.data_type}_{self.space}.feather'
        )
        
        # Elegir cuál archivo existe
        if os.path.exists(harmonized_file):
            self.path_input = harmonized_file
        else:
            self.path_input = normal_file
        
        # Rutas de salida y diagnósticos
        self.path_output = os.path.join(self.base_path, 'PSM_datasets')
        self.path_diagnostics = os.path.join(self.base_path, 'PSM_diagnostics')
        
        # Crear directorios si no existen
        os.makedirs(self.path_output, exist_ok=True)
        os.makedirs(self.path_diagnostics, exist_ok=True)


def apply_psm_multiple_ratios(config, group1, group2, ratios, caliper=0.2,
                              run_diagnostics=True, matching_covariates=None):
    """
    Apply PSM with multiple ratios at SUBJECT LEVEL.

    Parameters
    ----------
    config : PSMConfig
    group1 : str   Treatment group
    group2 : str or list   Control group(s)
    ratios : list  e.g. ['1:1', '2:1', '5:1']
    caliper : float   Max PS distance in SD units
    run_diagnostics : bool
    matching_covariates : list or None
        Columns used for propensity score (e.g. ['age'] or ['age', 'education']).
        None -> auto-detect all available demographic columns.

    Returns
    -------
    tuple : (matched_datasets dict, diagnostic_results dict)
    """
    print(f"\n{'#'*80}")
    print(f"# PSM PIPELINE - SUBJECT-LEVEL MATCHING")
    print(f"# Treatment: {group1} | Control: {group2}")
    print(f"# Ratios: {ratios}")
    print(f"# Caliper: {caliper} SD")
    print(f"{'#'*80}\n")
    
    # Load data
    print("Loading data...")
    data_original = pd.read_feather(config.path_input)
    print(f"[OK] Loaded: {data_original.shape}")
    
    # Verify group column exists
    if 'group' not in data_original.columns:
        raise ValueError("ERROR: 'group' column not found in data!")
    
    # CRITICAL: Count unique SUBJECTS, not rows
    print(f"\nGroup distribution in original data (SUBJECTS):")
    group_counts = data_original[['subject', 'group']].drop_duplicates().groupby('group').size()
    for group, count in group_counts.items():
        print(f"  {group}: {count} subjects")
    
    print(f"\nTotal rows in dataset: {len(data_original)}")
    print(f"Total unique subjects: {data_original['subject'].nunique()}")
    
    # Apply PSM for each ratio
    matched_datasets = {}
    diagnostic_results = {}

    for ratio in ratios:
        print(f"\n{'='*80}")
        print(f"Processing ratio: {ratio}")
        print(f"{'='*80}")
        
        # Perform matching AT SUBJECT LEVEL
        matched_data = match_and_optimize(
            data_original.copy(),
            group1,
            group2,
            ratio,
            caliper=caliper,
            matching_covariates=matching_covariates,
        )
        
        # Check if matching succeeded
        if len(matched_data) == 0:
            print(f"\n[X] Matching failed for {ratio} - skipping")
            continue
        
        # STRICT VERIFICATION AT SUBJECT LEVEL
        print(f"\n{'='*60}")
        print(f"STRICT VERIFICATION - {ratio} (SUBJECT LEVEL)")
        print(f"{'='*60}")
        
        verify_matched_dataset(matched_data, group1, group2, ratio)
        
        # Save matched dataset
        output_file = os.path.join(
            config.path_output,
            f'Data_matched_{config.data_type}_{config.space}_{group1}_{ratio.replace(":", "to")}.feather'
        )
        matched_data.to_feather(output_file)
        print(f"\n[OK] Saved matched dataset: {output_file}")
        
        # Run diagnostics if requested
        if run_diagnostics:
            print(f"\nRunning balance diagnostics for {ratio}...")
            
            diagnostic_dir = os.path.join(
                config.path_diagnostics,
                f'{group1}_{ratio.replace(":", "to")}'
            )
            
            try:
                diagnostic_result = diagnose_psm_quality(
                    data_original,
                    matched_data,
                    group1,
                    group2[0] if isinstance(group2, list) else group2,
                    diagnostic_dir
                )
                diagnostic_results[ratio] = diagnostic_result
            except Exception as e:
                print(f"[!] Warning: Diagnostics failed for {ratio}: {e}")
                diagnostic_results[ratio] = {
                    'quality': 'UNKNOWN',
                    'max_smd': 999.0
                }
        
        # Store
        matched_datasets[ratio] = matched_data
    
    # Create combined diagnostic summary
    if run_diagnostics and diagnostic_results:
        create_combined_diagnostic_report(
            diagnostic_results,
            config.path_diagnostics,
            group1
        )
    
    return matched_datasets, diagnostic_results


def verify_matched_dataset(matched_data, group1, group2, ratio):
    """
    Strict verification of matched dataset AT SUBJECT LEVEL
    
    Parameters:
    -----------
    matched_data : DataFrame
        Matched dataset to verify
    group1 : str
        Treatment group name
    group2 : str or list
        Control group name(s)
    ratio : str
        Expected ratio (e.g., '1:1', '2:1')
    """
    # CRITICAL: Count unique SUBJECTS, not rows
    subject_counts = matched_data[['subject', 'group']].drop_duplicates().groupby('group').size()
    
    # Handle multiple control groups
    if isinstance(group2, list):
        group2_name = 'Control'
    else:
        group2_name = group2
    
    n_treatment = subject_counts.get(group1, 0)
    n_control = subject_counts.get(group2_name, 0)
    
    # Parse expected ratio
    expected_ratio_num = int(ratio.split(':')[1])
    expected_controls = n_treatment * expected_ratio_num
    expected_total = n_treatment + expected_controls
    
    print(f"Sample size verification (SUBJECTS):")
    print(f"  Treatment ({group1}): {n_treatment} subjects")
    print(f"  Control ({group2_name}): {n_control} subjects")
    print(f"  Total subjects: {n_treatment + n_control}")
    print(f"  Total rows: {len(matched_data)}")
    print(f"  Expected controls: {expected_controls}")
    print(f"  Expected total subjects: {expected_total}")
    
    # Verification checks
    all_passed = True
    
    # Check 1: Ratio is correct
    actual_ratio = n_control / n_treatment if n_treatment > 0 else 0
    if abs(actual_ratio - expected_ratio_num) < 0.1:
        print(f"  [OK] Check 1: Ratio is correct (1:{actual_ratio:.2f})")
    else:
        print(f"  [X] Check 1 FAILED: Ratio mismatch!")
        print(f"    Expected: 1:{expected_ratio_num}")
        print(f"    Actual: 1:{actual_ratio:.2f}")
        all_passed = False
    
    # Check 2: No duplicate subjects
    n_unique_subjects = matched_data['subject'].nunique()
    n_total_subjects = n_treatment + n_control
    if n_unique_subjects == n_total_subjects:
        print(f"  [OK] Check 2: No duplicate subjects")
    else:
        print(f"  [X] Check 2 FAILED: Duplicate subjects detected!")
        print(f"    Unique subjects: {n_unique_subjects}")
        print(f"    Expected: {n_total_subjects}")
        all_passed = False
    
    # Check 3: Only expected groups present
    expected_groups = {group1, group2_name}
    actual_groups = set(matched_data['group'].unique())
    if actual_groups == expected_groups:
        print(f"  [OK] Check 3: Only expected groups present")
    else:
        unexpected = actual_groups - expected_groups
        missing = expected_groups - actual_groups
        if unexpected:
            print(f"  [X] Check 3 FAILED: Unexpected groups: {unexpected}")
            all_passed = False
        if missing:
            print(f"  [X] Check 3 FAILED: Missing groups: {missing}")
            all_passed = False
    
    # Check 4: All rows have valid subjects
    rows_per_subject = matched_data.groupby('subject').size()
    print(f"\n  Rows per subject statistics:")
    print(f"    Mean: {rows_per_subject.mean():.1f}")
    print(f"    Min: {rows_per_subject.min()}")
    print(f"    Max: {rows_per_subject.max()}")
    
    # Final verdict
    if all_passed:
        print(f"\n  [OK][OK][OK] ALL CHECKS PASSED [OK][OK][OK]")
    else:
        print(f"\n  [X][X][X] VERIFICATION FAILED [X][X][X]")
        print(f"  DO NOT USE THIS DATASET FOR ANALYSIS!")


def create_combined_diagnostic_report(diagnostic_results, output_dir, group1):
    """Create combined summary of all diagnostic results"""
    print(f"\n{'='*80}")
    print("COMBINED DIAGNOSTIC SUMMARY")
    print(f"{'='*80}\n")
    
    summary_rows = []
    
    for ratio, results in diagnostic_results.items():
        summary_rows.append({
            'Ratio': ratio,
            'Max_SMD': f"{results['max_smd']:.3f}",
            'Quality': results['quality'],
            'Recommendation': 'Use for analysis' if results['quality'] in ['EXCELLENT', 'ACCEPTABLE'] else 'DO NOT USE'
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(output_dir, f'{group1}_diagnostic_summary.xlsx')
    summary_df.to_excel(summary_file, index=False)
    
    print(summary_df.to_string(index=False))
    print(f"\n[OK] Combined summary saved: {summary_file}")
    
    # Print recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR ANALYSIS")
    print(f"{'='*80}")
    
    for ratio, results in diagnostic_results.items():
        status_icon = "[OK]" if results['quality'] in ['EXCELLENT', 'ACCEPTABLE'] else "[X]"
        print(f"\n{status_icon} {ratio}:")
        print(f"  Quality: {results['quality']} (SMD={results['max_smd']:.3f})")
        
        if results['quality'] == 'EXCELLENT':
            print(f"  -> RECOMMENDED for primary analysis")
        elif results['quality'] == 'ACCEPTABLE':
            print(f"  -> Can be used, consider sensitivity analyses")
        else:
            print(f"  -> DO NOT USE - balance is poor")
    
    print(f"{'='*80}\n")


def create_summary_report(matched_datasets, output_path):
    """Create summary report of matching results AT SUBJECT LEVEL"""
    summary = []
    
    for ratio, data in matched_datasets.items():
        # Count SUBJECTS, not rows
        subject_counts = data[['subject', 'group']].drop_duplicates().groupby('group').size()
        
        summary.append({
            'Ratio': ratio,
            'Total_Subjects': data['subject'].nunique(),
            'Total_Rows': len(data),
            'Groups': ', '.join([f"{g}: {c}" for g, c in subject_counts.items()]),
            'Features': len(data.columns)
        })
    
    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(output_path, 'PSM_summary.xlsx')
    summary_df.to_excel(summary_file, index=False)
    
    print(f"\n{'='*80}")
    print("PSM SAMPLE SIZE SUMMARY (SUBJECT LEVEL)")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))
    print(f"\n[OK] Summary saved: {summary_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    from config import BASE_PATH
    BASE_PATH = os.path.join(BASE_PATH, 'Resultados')
    DATA_TYPE = 'ce'
    SPACE     = 'roi'

    GROUP1  = 'PSEN1'
    GROUP2  = ['Control', 'Relative']

    RATIOS  = ['1:1', '2:1', '4:1','5:1']
    CALIPER = 0.2
    RUN_DIAGNOSTICS = True

    # -------------------------------------------------------------------------
    # PSM scenarios
    #
    # PRIMARY    — propensity score built on age only.
    #              Uses all 91 PSEN1 and all available Controls.
    #              SITE confound is handled by NeuroHarmonize (run first).
    #
    # SENSITIVITY— propensity score built on age + education.
    #              Restricted to subjects where education is available
    #              (91 PSEN1 = 100 %, ~31 Controls = 7 %).
    #              Generates small-n datasets useful for robustness reporting.
    # -------------------------------------------------------------------------
    PSM_SCENARIOS = {
        'primary':     ['age'],
        'sensitivity': ['age', 'education'],
    }

    config = PSMConfig(BASE_PATH, DATA_TYPE, SPACE)

    print(f"\n{'#'*80}")
    print(f"# PSM CONFIGURATION")
    print(f"{'#'*80}")
    print(f"Input file  : {config.path_input}")
    print(f"Output dir  : {config.path_output}")
    print(f"Diagnostics : {config.path_diagnostics}")
    print(f"Scenarios   : {list(PSM_SCENARIOS.keys())}")
    print(f"{'#'*80}\n")

    all_scenario_results = {}

    for scenario_name, cov_list in PSM_SCENARIOS.items():
        print(f"\n{'#'*80}")
        print(f"# SCENARIO: {scenario_name.upper()}  —  covariables: {cov_list}")
        print(f"{'#'*80}\n")

        # For 'sensitivity': restrict data to subjects with complete education
        if scenario_name == 'sensitivity':
            data_full = pd.read_feather(config.path_input)
            if 'education' not in data_full.columns:
                print("[!]  'education' no encontrada en el feather. "
                      "Ejecuta 1_make_dataframe.py primero. Saltando escenario sensitivity.")
                continue
            n_before = data_full['subject'].nunique()
            data_full = data_full[data_full['education'].notna()].copy()
            n_after = data_full['subject'].nunique()
            print(f"  Restricción por education: {n_before} -> {n_after} sujetos")

            # Temporarily save filtered file so apply_psm_multiple_ratios can load it
            import tempfile, os as _os
            tmp_path = _os.path.join(BASE_PATH, f'_tmp_sensitivity.feather')
            data_full.reset_index(drop=True).to_feather(tmp_path)

            config_sens = PSMConfig.__new__(PSMConfig)
            config_sens.__dict__.update(config.__dict__)
            config_sens.path_input      = tmp_path
            config_sens.path_output     = config.path_output
            config_sens.path_diagnostics = config.path_diagnostics
            # Use a different data_type tag so auto-save inside apply_psm_multiple_ratios
            # writes to e.g. Data_matched_ce_sens_roi_PSEN1_1to1.feather
            # and does NOT overwrite the primary files (Data_matched_ce_roi_*).
            config_sens.data_type = f'{DATA_TYPE}_sens'
            active_config = config_sens
        else:
            active_config = config

        matched_datasets, diagnostics = apply_psm_multiple_ratios(
            active_config,
            GROUP1,
            GROUP2,
            RATIOS,
            caliper=CALIPER,
            run_diagnostics=RUN_DIAGNOSTICS,
            matching_covariates=cov_list,
        )

        # Rename output files to include scenario tag
        import os as _os
        for ratio, mdata in matched_datasets.items():
            ratio_tag = ratio.replace(':', 'to')
            new_name = _os.path.join(
                config.path_output,
                f'Data_matched_{DATA_TYPE}_{SPACE}_{GROUP1}_{ratio_tag}_{scenario_name}.feather'
            )
            mdata.reset_index(drop=True).to_feather(new_name)
            print(f"  [OK] Guardado escenario {scenario_name} {ratio}: {new_name}")

        all_scenario_results[scenario_name] = {
            'datasets':    matched_datasets,
            'diagnostics': diagnostics,
        }

        # Clean up temp file for sensitivity
        if scenario_name == 'sensitivity' and _os.path.exists(tmp_path):
            _os.remove(tmp_path)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("PSM PIPELINE COMPLETADO")
    print("="*80)
    for scenario_name, res in all_scenario_results.items():
        n_datasets = len(res['datasets'])
        print(f"\n  Escenario '{scenario_name}': {n_datasets} datasets generados")
        for ratio, diag in res['diagnostics'].items():
            q = diag.get('quality', '?')
            smd = diag.get('max_smd', 999)
            icon = '[OK]' if q in ('EXCELLENT', 'ACCEPTABLE') else '[X]'
            print(f"    {icon} {ratio}: {q} (max SMD={smd:.3f})")
    print(f"\n  Datasets en: {config.path_output}\n")
