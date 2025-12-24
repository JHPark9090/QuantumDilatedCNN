#!/usr/bin/env python3
"""
Aggregate comprehensive metrics results across all configurations.

This script loads all individual result files from the SLURM array job
and creates summary tables and visualizations for publication.

Usage:
    python aggregate_comprehensive_results.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import argparse


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))

    d = (mean1 - mean2) / pooled_std
    return d


def load_all_results(base_dir='./comprehensive_results'):
    """Load all result files from the comprehensive metrics run."""
    base_path = Path(base_dir)

    all_data = []

    # Search for all .npy files
    for npy_file in base_path.rglob('*_comprehensive_*.npy'):
        # Parse filename: {model}_comprehensive_{n_qubits}q_{n_layers}l_seed{seed}.npy
        filename = npy_file.stem  # Remove .npy extension

        # Extract model name
        if filename.startswith('QCNN_comprehensive'):
            model_name = 'QCNN'
            remainder = filename.replace('QCNN_comprehensive_', '')
        elif filename.startswith('QuantumDilatedCNN_comprehensive'):
            model_name = 'QuantumDilatedCNN'
            remainder = filename.replace('QuantumDilatedCNN_comprehensive_', '')
        else:
            print(f"Warning: Unknown model in {filename}")
            continue

        # Parse: {n_qubits}q_{n_layers}l_seed{seed}
        parts = remainder.split('_')
        n_qubits = int(parts[0].replace('q', ''))
        n_layers = int(parts[1].replace('l', ''))
        seed = int(parts[2].replace('seed', ''))

        # Load data
        try:
            data = np.load(npy_file, allow_pickle=True).item()

            # Extract metrics
            row = {
                'model': model_name,
                'n_qubits': n_qubits,
                'n_layers': n_layers,
                'seed': seed,
                'meyer_wallach_mean': data['meyer_wallach']['mean'],
                'meyer_wallach_std': data['meyer_wallach']['std'],
            }

            # Concentratable (may not exist for large qubits)
            if 'concentratable' in data:
                row['concentratable_mean'] = data['concentratable']['mean']
                row['concentratable_std'] = data['concentratable']['std']
            else:
                row['concentratable_mean'] = np.nan
                row['concentratable_std'] = np.nan

            # Distance-dependent entanglement
            if 'distance_entanglement' in data:
                # Get all distances
                for d in sorted(data['distance_entanglement'].keys()):
                    row[f'mi_d{d}_mean'] = data['distance_entanglement'][d]['mean']
                    row[f'mi_d{d}_std'] = data['distance_entanglement'][d]['std']

            # Expressibility
            if 'expressibility' in data:
                row['expressibility_kl'] = data['expressibility']['kl_divergence']
                row['expressibility_js'] = data['expressibility']['js_divergence']
                row['expressibility_wasserstein'] = data['expressibility']['wasserstein_distance']
            else:
                row['expressibility_kl'] = np.nan
                row['expressibility_js'] = np.nan
                row['expressibility_wasserstein'] = np.nan

            # Effective dimension
            if 'effective_dimension' in data:
                row['eff_dim'] = data['effective_dimension']['effective_dimension']
                row['eff_dim_normalized'] = data['effective_dimension']['normalized_dimension']
            else:
                row['eff_dim'] = np.nan
                row['eff_dim_normalized'] = np.nan

            all_data.append(row)

        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
            continue

    if not all_data:
        raise ValueError("No data files found! Check the base directory.")

    df = pd.DataFrame(all_data)
    return df


def create_summary_table(df, n_qubits, n_layers):
    """Create publication-ready summary table for a specific configuration."""

    # Filter for this configuration
    df_config = df[(df['n_qubits'] == n_qubits) & (df['n_layers'] == n_layers)]

    if len(df_config) == 0:
        print(f"Warning: No data for {n_qubits}q, {n_layers}l")
        return None

    # Separate by model
    qcnn = df_config[df_config['model'] == 'QCNN']
    dilated = df_config[df_config['model'] == 'QuantumDilatedCNN']

    if len(qcnn) == 0 or len(dilated) == 0:
        print(f"Warning: Missing model data for {n_qubits}q, {n_layers}l")
        return None

    # Statistical tests
    results = []

    metrics = [
        ('meyer_wallach_mean', 'Meyer-Wallach'),
        ('concentratable_mean', 'Concentratable'),
        ('mi_d1_mean', 'Local MI (d=1)'),
        ('expressibility_kl', 'Expressibility (KL)'),
        ('expressibility_js', 'Expressibility (JS)'),
        ('eff_dim', 'Effective Dim'),
    ]

    for metric_col, metric_name in metrics:
        if metric_col not in qcnn.columns or qcnn[metric_col].isna().all():
            continue

        qcnn_vals = qcnn[metric_col].dropna().values
        dilated_vals = dilated[metric_col].dropna().values

        if len(qcnn_vals) == 0 or len(dilated_vals) == 0:
            continue

        # Compute statistics
        qcnn_mean = np.mean(qcnn_vals)
        qcnn_std = np.std(qcnn_vals, ddof=1)
        qcnn_sem = qcnn_std / np.sqrt(len(qcnn_vals))

        dilated_mean = np.mean(dilated_vals)
        dilated_std = np.std(dilated_vals, ddof=1)
        dilated_sem = dilated_std / np.sqrt(len(dilated_vals))

        # T-test
        t_stat, p_value = stats.ttest_ind(qcnn_vals, dilated_vals, equal_var=False)

        # Cohen's d
        d = cohens_d(qcnn_vals, dilated_vals)

        # Effect size label
        if abs(d) < 0.2:
            effect = 'Neg'
        elif abs(d) < 0.5:
            effect = 'S'
        elif abs(d) < 0.8:
            effect = 'M'
        else:
            effect = 'L'

        # Significance stars
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        results.append({
            'Metric': metric_name,
            'QCNN': f'{qcnn_mean:.4f} ± {qcnn_sem:.4f}',
            'Dilated': f'{dilated_mean:.4f} ± {dilated_sem:.4f}',
            'Δ': f'{qcnn_mean - dilated_mean:+.4f}',
            'p-value': f'{p_value:.4f}{sig}',
            "Cohen's d": f'{d:.2f} ({effect})'
        })

    # Also add global MI (d>1) if available
    global_cols = [col for col in qcnn.columns if col.startswith('mi_d') and col != 'mi_d1_mean']
    if global_cols:
        # Average over all d>1
        qcnn_global = qcnn[global_cols].mean(axis=1).values
        dilated_global = dilated[global_cols].mean(axis=1).values

        qcnn_mean = np.mean(qcnn_global)
        qcnn_sem = np.std(qcnn_global, ddof=1) / np.sqrt(len(qcnn_global))
        dilated_mean = np.mean(dilated_global)
        dilated_sem = np.std(dilated_global, ddof=1) / np.sqrt(len(dilated_global))

        t_stat, p_value = stats.ttest_ind(qcnn_global, dilated_global, equal_var=False)
        d = cohens_d(qcnn_global, dilated_global)

        if abs(d) < 0.2:
            effect = 'Neg'
        elif abs(d) < 0.5:
            effect = 'S'
        elif abs(d) < 0.8:
            effect = 'M'
        else:
            effect = 'L'

        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        results.append({
            'Metric': 'Global MI (d>1)',
            'QCNN': f'{qcnn_mean:.4f} ± {qcnn_sem:.4f}',
            'Dilated': f'{dilated_mean:.4f} ± {dilated_sem:.4f}',
            'Δ': f'{qcnn_mean - dilated_mean:+.4f}',
            'p-value': f'{p_value:.4f}{sig}',
            "Cohen's d": f'{d:.2f} ({effect})'
        })

    summary_df = pd.DataFrame(results)
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Aggregate comprehensive metrics results')
    parser.add_argument('--base-dir', type=str, default='/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/results/comprehensive_metrics',
                       help='Base directory containing results')
    parser.add_argument('--output-dir', type=str, default='/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/results/comprehensive_metrics',
                       help='Output directory for aggregated results')
    args = parser.parse_args()

    print("="*80)
    print("AGGREGATING COMPREHENSIVE METRICS RESULTS")
    print("="*80)
    print()

    # Load all data
    print("Loading all result files...")
    df = load_all_results(args.base_dir)
    print(f"✓ Loaded {len(df)} result files")
    print(f"  Models: {df['model'].unique()}")
    print(f"  Qubits: {sorted(df['n_qubits'].unique())}")
    print(f"  Layers: {sorted(df['n_layers'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    print()

    # Save complete data
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    all_results_file = output_path / 'comprehensive_all_results.csv'
    df.to_csv(all_results_file, index=False)
    print(f"✓ Saved complete results: {all_results_file}")
    print()

    # Create summary tables for each configuration
    print("="*80)
    print("CREATING SUMMARY TABLES")
    print("="*80)
    print()

    all_summaries = []

    for n_qubits in sorted(df['n_qubits'].unique()):
        for n_layers in sorted(df['n_layers'].unique()):
            print(f"\n{'='*80}")
            print(f"Configuration: {n_qubits} qubits, {n_layers} layers")
            print(f"{'='*80}\n")

            summary = create_summary_table(df, n_qubits, n_layers)

            if summary is not None:
                print(summary.to_string(index=False))
                print()

                # Save individual summary
                summary_file = output_path / f'summary_{n_qubits}q_{n_layers}l.csv'
                summary.to_csv(summary_file, index=False)
                print(f"✓ Saved: {summary_file}\n")

                # Add to combined summary
                summary['n_qubits'] = n_qubits
                summary['n_layers'] = n_layers
                all_summaries.append(summary)

    # Save combined summary
    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_file = output_path / 'comprehensive_summary_all_configs.csv'
        combined_summary.to_csv(combined_file, index=False)
        print(f"\n✓ Saved combined summary: {combined_file}")

    print()
    print("="*80)
    print("AGGREGATION COMPLETE")
    print("="*80)
    print(f"\nOutput files in: {output_path}")
    print("  - comprehensive_all_results.csv (raw data)")
    print("  - summary_*q_*l.csv (individual config summaries)")
    print("  - comprehensive_summary_all_configs.csv (combined summaries)")
    print()


if __name__ == '__main__':
    main()
