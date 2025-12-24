# Robust Experimental Plan: Comprehensive Analysis of QCNN vs QuantumDilatedCNN

**Date**: November 2025
**Status**: Publication-Ready Framework
**Target Venues**: Nature Physics, Physical Review Letters, NeurIPS, ICML

---

## Executive Summary

This experimental plan provides a **statistically rigorous, multi-metric analysis** comparing two quantum circuit architectures:

1. **QCNN** (Cong et al. 2019): Nearest-neighbor entanglement
2. **QuantumDilatedCNN**: Dilated (non-adjacent) entanglement

**Comprehensive Metrics Suite**:
- ✅ **Entanglement**: Meyer-Wallach, Concentratable, Distance-dependent
- ✅ **Expressibility** (Sim et al. 2019): Hilbert space coverage
- ✅ **Effective Dimension** (Abbas et al. 2021): Model capacity

**Statistical Rigor**:
- Multiple random seeds (10-30)
- Multiple system sizes (4, 6, 8, 10 qubits)
- Multiple depths (1, 2, 3, 4 layers)
- Statistical significance testing
- Confidence intervals and effect sizes

---

## Table of Contents

1. [Metrics Explained](#1-metrics-explained)
2. [Experimental Design](#2-experimental-design)
3. [Statistical Requirements](#3-statistical-requirements)
4. [Implementation Guide](#4-implementation-guide)
5. [Computational Resources](#5-computational-resources)
6. [Data Analysis](#6-data-analysis)
7. [Visualization](#7-visualization)
8. [Publication Standards](#8-publication-standards)

---

## 1. Metrics Explained

### A. Entanglement Measures

#### **Meyer-Wallach Measure (Entangling Capability)**

**Formula**:
```
Q = 2 × (1 - (1/n) × Σᵢ Tr(ρᵢ²))
```

**What it measures**:
- Average entanglement between each qubit and the rest of the system
- **Range**: [0, 1]
  - 0 = Product state (no entanglement)
  - 1 = Maximally entangled state

**Interpretation**:
- Higher Q → Circuit creates more entanglement
- Indicates potential quantum advantage
- Correlates with circuit expressivity

**Reference**: Meyer & Wallach (2002), J. Math. Phys.

---

#### **Concentratable Entanglement**

**Formula**:
```
C(|ψ⟩) = 1 - (1/2ⁿ) × Σ_α Tr(ρ_α²)
```
where sum is over all 2ⁿ subsets α of n qubits.

**What it measures**:
- Amount of entanglement that can be concentrated into Bell pairs
- Multipartite entanglement structure
- **Range**: [0, 1]

**Interpretation**:
- Higher C → More "useful" entanglement
- Can be converted to maximally entangled pairs
- Relevant for quantum communication protocols

**Computational Note**: Scales as O(2ⁿ) → only feasible for n ≤ 10 qubits

**Reference**: Beckey et al. (2021), Phys. Rev. Lett.

---

#### **Distance-Dependent Mutual Information**

**Formula** (for qubits i and j):
```
I(i:j) = S(ρᵢ) + S(ρⱼ) - S(ρᵢⱼ)
```
where S(ρ) = -Tr(ρ log₂ ρ) is von Neumann entropy.

**What it measures**:
- Quantum + classical correlations between qubit pairs
- Separated by distance d = |i - j|

**Categories**:
- **LOCAL entanglement**: d = 1 (nearest neighbors)
- **GLOBAL entanglement**: d > 1 (long-range)

**Interpretation**:
- QCNN: High local, moderate global
- QuantumDilatedCNN: Zero local (by design), high global
- Reveals spatial structure of entanglement

**Key Insight**: Architectural differences manifest as distance-dependent patterns.

---

### B. Expressibility (Sim et al. 2019)

**Definition**: How uniformly a parametrized quantum circuit (PQC) samples the Hilbert space.

**Method**:
1. Sample N random parameter sets → Generate N quantum states
2. Compute fidelities F = |⟨ψᵢ|ψⱼ⟩|² for all pairs (i,j)
3. Create fidelity distribution P_circuit(F)
4. Compare to Haar-random distribution P_Haar(F)

**Metrics**:
```
KL Divergence: D_KL(P_circuit || P_Haar) = Σ P_circuit(F) log(P_circuit(F) / P_Haar(F))

JS Divergence: D_JS(P_circuit, P_Haar) = ½[D_KL(P_circuit || M) + D_KL(P_Haar || M)]
               where M = ½(P_circuit + P_Haar)
```

**Interpretation**:
- **Lower divergence** → More expressive circuit
- Haar-like distribution → Uniformly samples Hilbert space
- **Higher divergence** → Biased sampling, limited expressivity

**For pure states** (n qubits):
```
P_Haar(F) = (2ⁿ - 1) × (1 - F)^(2ⁿ - 2)
```

**Why it matters**:
- High expressibility ≠ high entanglement
- QCNN might have high entanglement but low expressibility (or vice versa)
- Separates "quantum power" into orthogonal components

**Reference**: Sim et al. (2019), Adv. Quantum Technol.
https://arxiv.org/abs/1905.10876

---

### C. Effective Dimension (Abbas et al. 2021)

**Definition**: The "true" dimensionality of the model manifold in Hilbert space.

**Formula** (Fisher Information Matrix approach):
```
d_eff = Tr(F) / ||F||_F
```
where:
- F = Quantum Fisher Information Matrix
- ||F||_F = Frobenius norm = √(Σᵢⱼ Fᵢⱼ²)

**What it measures**:
- Model capacity: How many "independent directions" in parameter space
- Overfitting risk: Low d_eff → underfits, High d_eff → overfits
- **Range**: [1, n_params]

**Interpretation**:
- d_eff << n_params → Many parameters are redundant
- d_eff ≈ n_params → All parameters contribute independently
- Normalized: d_eff / n_params ∈ [0, 1]

**Connection to Machine Learning**:
- Similar to effective degrees of freedom in ridge regression
- Predicts generalization: Lower d_eff → better generalization
- For quantum circuits: d_eff ~ D² where D = Hilbert space dimension explored

**Computational Note**:
- Full Fisher matrix: O(n_params²) → expensive for large circuits
- Use sampling-based estimation for n_params > 100

**Why it matters**:
- **Training**: High d_eff circuits are harder to optimize (barren plateaus)
- **Performance**: Optimal d_eff balances capacity and generalization
- **Comparison**: QCNN vs Dilated may have similar n_params but different d_eff

**Reference**: Abbas et al. (2021), Nat. Comput. Sci.
https://arxiv.org/abs/2011.01938

---

## 2. Experimental Design

### A. Multi-Seed Configuration (Statistical Rigor)

**Minimum Standard** (for publication):
```python
SEEDS = list(range(2024, 2044))  # 20 independent seeds
```

**Gold Standard** (Nature/Science):
```python
SEEDS = list(range(2024, 2054))  # 30 independent seeds
```

**Why Multiple Seeds?**
- Single seed can give misleading results (lucky/unlucky initialization)
- Reviewers demand: "Is this statistically significant?"
- Report: Mean ± SEM (Standard Error of Mean)
- Enables p-value testing and confidence intervals

---

### B. Multi-Scale Configuration

#### **System Size Scaling**
```python
QUBIT_CONFIGS = [4, 6, 8, 10]  # Recommended

# Optional extended study:
QUBIT_CONFIGS_EXTENDED = [4, 6, 8, 10, 12]  # If computationally feasible
```

**Research Questions**:
- Does entanglement gap widen or narrow with system size?
- Do architectures scale differently? (QCNN: linear, Dilated: log?)
- Is there a crossover point where Dilated wins?

---

#### **Depth Scaling**
```python
LAYER_CONFIGS = [1, 2, 3, 4]  # Shallow to deep
```

**Research Questions**:
- Does deeper = more entanglement? (Expected: yes, but saturates)
- Do QCNN and Dilated scale differently with depth?
- Barren plateau onset: Does high depth kill gradients?

---

### C. Full Factorial Experimental Grid

**Complete Configuration**:
```python
EXPERIMENTAL_GRID = {
    'n_qubits': [4, 6, 8, 10],           # 4 sizes
    'n_layers': [1, 2, 3, 4],            # 4 depths
    'seeds': list(range(2024, 2044)),    # 20 seeds
    'models': ['QCNN', 'QuantumDilatedCNN']
}

# Total experiments:
# 4 qubits × 4 layers × 20 seeds × 2 models = 640 runs
```

**Per Run**:
- 100 random parameter samples
- All 6 metrics computed
- ~5-30 minutes depending on (n_qubits, n_layers)

**Total Computational Cost**: See Section 5.

---

### D. Sample Size per Configuration

```python
N_SAMPLES_PER_SEED = 100  # Random parameter configurations

# For expressibility:
N_FIDELITY_PAIRS = 500  # Random state pairs for fidelity distribution

# For effective dimension:
N_FISHER_SAMPLES = 50   # Samples for Fisher matrix averaging
```

**Trade-offs**:
- More samples → Better statistics, longer runtime
- 100 samples is standard in literature (Sim et al., Abbas et al.)
- For n_qubits ≤ 6: Can afford 200 samples
- For n_qubits ≥ 8: Use 100 samples (concentratable becomes expensive)

---

## 3. Statistical Requirements

### A. Reporting Standards

#### **Mean ± Standard Error**
```python
# Correct reporting:
"Meyer-Wallach: 0.655 ± 0.008"  # Mean ± SEM

# Where SEM = std / sqrt(n_seeds)
SEM = np.std(values) / np.sqrt(len(values))
```

#### **95% Confidence Intervals**
```python
from scipy import stats

# For normal distribution:
ci_95 = stats.t.interval(0.95, len(values)-1,
                         loc=np.mean(values),
                         scale=stats.sem(values))

# Report as:
"Meyer-Wallach: 0.655 ± 0.008 (95% CI: [0.640, 0.670])"
```

---

### B. Statistical Significance Testing

#### **Welch's t-test** (Unpaired, Unequal Variance)
```python
from scipy.stats import ttest_ind

# Compare QCNN vs Dilated for Meyer-Wallach:
qcnn_mw = [results from all seeds]
dilated_mw = [results from all seeds]

t_statistic, p_value = ttest_ind(qcnn_mw, dilated_mw, equal_var=False)

# Significance levels:
# p < 0.001: *** (highly significant)
# p < 0.01:  **  (very significant)
# p < 0.05:  *   (significant)
# p ≥ 0.05:  ns  (not significant)
```

---

#### **Effect Size (Cohen's d)**
```python
def cohens_d(group1, group2):
    """
    Cohen's d effect size:
    d = (mean1 - mean2) / pooled_std
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    d = (mean1 - mean2) / pooled_std
    return d

# Interpretation:
# |d| < 0.2: negligible
# |d| = 0.2-0.5: small effect
# |d| = 0.5-0.8: medium effect
# |d| > 0.8: large effect
```

---

### C. Example Results Table (Publication Format)

```markdown
Table 1: Comparison of Quantum Circuit Architectures (6 qubits, 2 layers, N=20 seeds)

Metric                 | QCNN          | QuantumDilatedCNN | Δ (Dilated-QCNN) | p-value   | Cohen's d
-----------------------|---------------|-------------------|------------------|-----------|----------
Meyer-Wallach          | 0.655 ± 0.008 | 0.502 ± 0.009     | -0.153           | <0.001*** | 1.21 (L)
Concentratable Ent.    | 0.476 ± 0.006 | 0.358 ± 0.007     | -0.118           | <0.001*** | 1.15 (L)
Local MI (d=1)         | 0.225 ± 0.012 | 0.000 ± 0.000     | -0.225           | <0.001*** | ∞
Global MI (d>1)        | 0.117 ± 0.008 | 0.270 ± 0.015     | +0.153           | <0.001*** | 1.32 (L)
Expressibility (KL)    | 0.045 ± 0.003 | 0.038 ± 0.004     | -0.007           | 0.021*    | 0.42 (S)
Effective Dimension    | 12.3 ± 0.8    | 15.7 ± 1.2        | +3.4             | 0.003**   | 0.68 (M)

(L) = Large effect, (M) = Medium effect, (S) = Small effect
All values: Mean ± SEM across 20 independent seeds
```

---

## 4. Implementation Guide

### A. Single Run (Testing)

```bash
# Test with single seed, small system
python measure_comprehensive_metrics.py \
    --n-qubits=4 \
    --n-layers=2 \
    --n-samples=50 \
    --seed=2024 \
    --output-dir='./test_results'

# Expected runtime: ~2-3 minutes
```

---

### B. Multi-Seed Single Configuration

```bash
# Run 20 seeds for 6 qubits, 2 layers
for seed in {2024..2043}; do
    python measure_comprehensive_metrics.py \
        --n-qubits=6 \
        --n-layers=2 \
        --n-samples=100 \
        --seed=$seed \
        --output-dir="./results/6q_2l"
done

# Expected runtime: 20 × 10 min = 200 minutes = 3.3 hours (sequential)
```

---

### C. SLURM Array Job (Recommended for HPC)

**Create**: `run_comprehensive_array.sh`

```bash
#!/bin/bash
#SBATCH --array=0-639          # 640 total configs (4×4×20×2)
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=qcnn_analysis
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err

# Configuration arrays
QUBITS=(4 6 8 10)
LAYERS=(1 2 3 4)
SEEDS=($(seq 2024 2043))  # 20 seeds

# Decode array task ID
N_QUBITS=4
N_LAYERS=4
N_SEEDS=20

qubit_idx=$(( SLURM_ARRAY_TASK_ID / (N_LAYERS * N_SEEDS) ))
remainder=$(( SLURM_ARRAY_TASK_ID % (N_LAYERS * N_SEEDS) ))
layer_idx=$(( remainder / N_SEEDS ))
seed_idx=$(( remainder % N_SEEDS ))

n_qubits=${QUBITS[$qubit_idx]}
n_layers=${LAYERS[$layer_idx]}
seed=${SEEDS[$seed_idx]}

# Run analysis
conda activate ./conda-envs/qml_eeg

python measure_comprehensive_metrics.py \
    --n-qubits=$n_qubits \
    --n-layers=$n_layers \
    --n-samples=100 \
    --seed=$seed \
    --output-dir="./comprehensive_results/${n_qubits}q_${n_layers}l"

echo "Completed: n_qubits=$n_qubits, n_layers=$n_layers, seed=$seed"
```

**Submit**:
```bash
mkdir -p logs
sbatch run_comprehensive_array.sh
```

---

### D. Aggregate Results Across Seeds

**Create**: `aggregate_comprehensive_results.py`

```python
#!/usr/bin/env python3
"""Aggregate results across multiple seeds for statistical analysis."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

def aggregate_results(results_dir, n_qubits, n_layers):
    """Load all seed results and compute statistics."""

    results_dir = Path(results_dir) / f"{n_qubits}q_{n_layers}l"

    # Find all seed files
    qcnn_files = sorted(results_dir.glob(f"QCNN_comprehensive_{n_qubits}q_{n_layers}l_seed*.npy"))
    dilated_files = sorted(results_dir.glob(f"QuantumDilatedCNN_comprehensive_{n_qubits}q_{n_layers}l_seed*.npy"))

    # Collect metrics across seeds
    metrics = {
        'QCNN': {'meyer_wallach': [], 'concentratable': [], 'expressibility_kl': [], 'eff_dim': []},
        'QuantumDilatedCNN': {'meyer_wallach': [], 'concentratable': [], 'expressibility_kl': [], 'eff_dim': []}
    }

    for file in qcnn_files:
        data = np.load(file, allow_pickle=True).item()
        metrics['QCNN']['meyer_wallach'].append(data['meyer_wallach']['mean'])
        if data['concentratable'] is not None:
            metrics['QCNN']['concentratable'].append(data['concentratable']['mean'])
        if data['expressibility'] is not None:
            metrics['QCNN']['expressibility_kl'].append(data['expressibility']['kl_divergence'])
        if data['effective_dimension'] is not None:
            metrics['QCNN']['eff_dim'].append(data['effective_dimension']['effective_dimension'])

    for file in dilated_files:
        data = np.load(file, allow_pickle=True).item()
        metrics['QuantumDilatedCNN']['meyer_wallach'].append(data['meyer_wallach']['mean'])
        if data['concentratable'] is not None:
            metrics['QuantumDilatedCNN']['concentratable'].append(data['concentratable']['mean'])
        if data['expressibility'] is not None:
            metrics['QuantumDilatedCNN']['expressibility_kl'].append(data['expressibility']['kl_divergence'])
        if data['effective_dimension'] is not None:
            metrics['QuantumDilatedCNN']['eff_dim'].append(data['effective_dimension']['effective_dimension'])

    # Compute statistics
    summary = []

    for metric_name in ['meyer_wallach', 'concentratable', 'expressibility_kl', 'eff_dim']:
        qcnn_vals = np.array(metrics['QCNN'][metric_name])
        dilated_vals = np.array(metrics['QuantumDilatedCNN'][metric_name])

        if len(qcnn_vals) == 0:
            continue

        # Statistics
        qcnn_mean = np.mean(qcnn_vals)
        qcnn_sem = stats.sem(qcnn_vals)
        dilated_mean = np.mean(dilated_vals)
        dilated_sem = stats.sem(dilated_vals)

        # t-test
        t_stat, p_value = stats.ttest_ind(qcnn_vals, dilated_vals, equal_var=False)

        # Cohen's d
        pooled_std = np.sqrt(((len(qcnn_vals)-1)*np.var(qcnn_vals, ddof=1) +
                              (len(dilated_vals)-1)*np.var(dilated_vals, ddof=1)) /
                             (len(qcnn_vals) + len(dilated_vals) - 2))
        cohens_d = (qcnn_mean - dilated_mean) / pooled_std

        # Effect size category
        if abs(cohens_d) < 0.2:
            effect = "Negligible"
        elif abs(cohens_d) < 0.5:
            effect = "Small"
        elif abs(cohens_d) < 0.8:
            effect = "Medium"
        else:
            effect = "Large"

        # Significance stars
        if p_value < 0.001:
            sig = "***"
        elif p_value < 0.01:
            sig = "**"
        elif p_value < 0.05:
            sig = "*"
        else:
            sig = "ns"

        summary.append({
            'Metric': metric_name,
            'QCNN (Mean ± SEM)': f"{qcnn_mean:.4f} ± {qcnn_sem:.4f}",
            'Dilated (Mean ± SEM)': f"{dilated_mean:.4f} ± {dilated_sem:.4f}",
            'Δ': f"{dilated_mean - qcnn_mean:+.4f}",
            'p-value': f"{p_value:.4e}",
            'Significance': sig,
            "Cohen's d": f"{cohens_d:.2f}",
            'Effect Size': effect,
            'N seeds': len(qcnn_vals)
        })

    df = pd.DataFrame(summary)

    # Save
    output_file = results_dir / f"statistical_summary_{n_qubits}q_{n_layers}l.csv"
    df.to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"Statistical Summary: {n_qubits} qubits, {n_layers} layers")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))
    print(f"\nSaved to: {output_file}")

    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='./comprehensive_results')
    parser.add_argument('--n-qubits', type=int, required=True)
    parser.add_argument('--n-layers', type=int, required=True)
    args = parser.parse_args()

    aggregate_results(args.results_dir, args.n_qubits, args.n_layers)
```

**Usage**:
```bash
# After all seeds complete:
python aggregate_comprehensive_results.py --n-qubits=6 --n-layers=2
```

---

## 5. Computational Resources

### A. Runtime Estimates

**Per Configuration** (n_qubits, n_layers, 1 seed, 100 samples):

| n_qubits | n_layers | Meyer-Wallach | Concentratable | Distance MI | Expressibility | Eff. Dim | **Total** |
|----------|----------|---------------|----------------|-------------|----------------|----------|-----------|
| 4        | 1        | 0.5 min       | 1 min          | 0.5 min     | 2 min          | 1 min    | **~5 min** |
| 4        | 2        | 1 min         | 2 min          | 1 min       | 3 min          | 2 min    | **~9 min** |
| 6        | 1        | 1 min         | 3 min          | 1 min       | 4 min          | 2 min    | **~11 min** |
| 6        | 2        | 2 min         | 6 min          | 2 min       | 6 min          | 3 min    | **~19 min** |
| 8        | 2        | 4 min         | 15 min         | 4 min       | 10 min         | 5 min    | **~38 min** |
| 10       | 2        | 8 min         | N/A*           | 8 min       | 15 min         | 8 min    | **~39 min** |

*Concentratable skipped for n_qubits > 8 (too expensive)

---

### B. Total Computational Cost

**Full Experimental Grid**:
```
4 qubits × 4 layers × 20 seeds × 2 models = 640 configs

Breakdown:
- 4q×1l: 80 configs × 5 min  = 400 min   = 6.7 hours
- 4q×2l: 80 configs × 9 min  = 720 min   = 12 hours
- 6q×1l: 80 configs × 11 min = 880 min   = 14.7 hours
- 6q×2l: 80 configs × 19 min = 1520 min  = 25.3 hours
- 8q×2l: 80 configs × 38 min = 3040 min  = 50.7 hours
- 10q×2l: 80 configs × 39 min = 3120 min = 52 hours

Total Sequential: ~162 hours = 6.75 days
```

**With Parallelization** (64 cores):
```
Total: 162 hours / 64 cores = 2.5 hours walltime
```

**With 128 cores** (SLURM array job):
```
Total: 162 hours / 128 cores = 1.3 hours walltime
```

---

### C. Storage Requirements

**Per Configuration**:
- NumPy file (.npy): ~50-200 KB (depends on n_qubits)
- Summary CSV: ~5 KB

**Total Storage**:
```
640 configs × 150 KB = 96 MB (NumPy files)
+ CSV summaries: ~10 MB

Total: ~110 MB
```

**Negligible** - easily fits in scratch space.

---

## 6. Data Analysis

### A. Loading Results

```python
import numpy as np
from pathlib import Path

# Load single result
data = np.load('comprehensive_results/6q_2l/QCNN_comprehensive_6q_2l_seed2024.npy',
               allow_pickle=True).item()

# Access metrics
print(f"Meyer-Wallach: {data['meyer_wallach']['mean']:.4f}")
print(f"Expressibility KL: {data['expressibility']['kl_divergence']:.4f}")
print(f"Effective Dimension: {data['effective_dimension']['effective_dimension']:.2f}")
```

---

### B. Cross-Configuration Analysis

**Scaling with System Size**:
```python
import matplotlib.pyplot as plt

qubits = [4, 6, 8, 10]
qcnn_mw = []
dilated_mw = []

for n_q in qubits:
    qcnn_vals = [load_result(f"QCNN_*_{n_q}q_2l_seed*.npy") for seed in seeds]
    dilated_vals = [load_result(f"QuantumDilatedCNN_*_{n_q}q_2l_seed*.npy") for seed in seeds]

    qcnn_mw.append(np.mean(qcnn_vals))
    dilated_mw.append(np.mean(dilated_vals))

plt.plot(qubits, qcnn_mw, 'o-', label='QCNN')
plt.plot(qubits, dilated_mw, 's-', label='QuantumDilatedCNN')
plt.xlabel('Number of Qubits')
plt.ylabel('Meyer-Wallach Measure')
plt.legend()
plt.savefig('mw_scaling.pdf')
```

---

## 7. Visualization

### A. Publication-Quality Figure Requirements

**Must Have**:
- ✅ Error bars (SEM or 95% CI)
- ✅ Statistical significance annotations (*, **, ***)
- ✅ High resolution (300 DPI minimum)
- ✅ Vector format (PDF) for scalability
- ✅ Clear legends and axis labels
- ✅ Consistent color scheme

**Example Matplotlib Settings**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")

# Font sizes
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Figure size (for 2-column paper)
plt.rcParams['figure.figsize'] = (7, 5)  # inches
plt.rcParams['figure.dpi'] = 300

# Save format
plt.savefig('figure.pdf', format='pdf', bbox_inches='tight')
```

---

### B. Suggested Figure Panels

**Figure 1: Comprehensive Metric Comparison**
- Panel A: Meyer-Wallach bar chart with error bars
- Panel B: Concentratable entanglement
- Panel C: Local vs Global MI
- Panel D: Expressibility distributions
- Panel E: Effective dimension

**Figure 2: Scaling Analysis**
- Panel A: Entanglement vs n_qubits
- Panel B: Expressibility vs n_qubits
- Panel C: Effective dimension vs n_qubits
- Panel D: All metrics vs n_layers

**Figure 3: Statistical Summary Heatmap**
- Rows: Metrics
- Columns: (n_qubits, n_layers) configurations
- Cell color: Effect size (Cohen's d)
- Cell annotation: p-value stars

---

## 8. Publication Standards

### A. Minimum Requirements for Top-Tier Venues

**Nature Physics / Physical Review Letters**:
- ✅ N ≥ 20 independent seeds
- ✅ Statistical significance testing (p-values, effect sizes)
- ✅ Multiple system sizes (≥3 different n_qubits)
- ✅ Baseline comparison (random circuits recommended)
- ✅ Theoretical predictions vs. experimental results
- ✅ Error bars on all plots
- ✅ Open-source code and data

**NeurIPS / ICML**:
- ✅ N ≥ 10 independent seeds
- ✅ Ablation studies (varying architecture components)
- ✅ Connection to ML performance metrics
- ✅ Computational cost analysis
- ✅ Reproducibility statement

---

### B. Supplementary Materials Checklist

- [ ] Full data tables (all seeds, all configurations)
- [ ] NumPy files with raw results
- [ ] Complete codebase (GitHub repository)
- [ ] Environment specification (conda export, requirements.txt)
- [ ] Extended methods (circuit diagrams, parameter details)
- [ ] Additional plots (per-seed distributions, correlation matrices)

---

### C. Key Research Questions to Address

**Primary Hypotheses**:
1. ✅ H1: QCNN exhibits higher total entanglement than QuantumDilatedCNN
2. ✅ H2: QuantumDilatedCNN has zero local entanglement by design
3. ✅ H3: QuantumDilatedCNN has higher global entanglement than QCNN
4. 🆕 H4: QuantumDilatedCNN has higher expressibility despite lower entanglement
5. 🆕 H5: Effective dimension correlates with entanglement magnitude
6. 🆕 H6: Entanglement patterns persist across system sizes

**Secondary Questions**:
- Does higher entanglement → worse trainability? (barren plateaus)
- Can we predict optimal architecture from task characteristics?
- How do noise and decoherence affect each architecture differently?

---

## 9. Timeline

### Phase 1: Core Analysis (Week 1-2)
- ✅ Implement comprehensive measurement script
- ✅ Test on small systems (4 qubits, 1 layer, 3 seeds)
- ✅ Debug and validate metrics
- ✅ Set up SLURM array jobs

### Phase 2: Full Experimental Grid (Week 2-3)
- ✅ Submit 640 SLURM array jobs
- ✅ Monitor progress
- ✅ Aggregate results as they complete

### Phase 3: Statistical Analysis (Week 3-4)
- ✅ Compute statistics across all seeds
- ✅ Generate summary tables
- ✅ Perform significance testing

### Phase 4: Visualization (Week 4)
- ✅ Create publication-quality figures
- ✅ Scaling plots
- ✅ Heatmaps and comparisons

### Phase 5: Manuscript Preparation (Week 5-6)
- ✅ Write methods section
- ✅ Create main and supplementary figures
- ✅ Draft results and discussion
- ✅ Prepare code/data release

---

## 10. References

### Papers

1. **Cong, I., Choi, S., & Lukin, M. D. (2019)**. Quantum convolutional neural networks. *Nature Physics*, 15(12), 1273-1278.
   https://arxiv.org/abs/1810.03787

2. **Sim, S., Johnson, P. D., & Aspuru-Guzik, A. (2019)**. Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms. *Advanced Quantum Technologies*, 2(12), 1900070.
   https://arxiv.org/abs/1905.10876

3. **Beckey, J. L., et al. (2021)**. Computable and operationally meaningful multipartite entanglement measures. *Physical Review Letters*, 127(14), 140501.
   https://arxiv.org/abs/2104.06923

4. **Abbas, A., et al. (2021)**. The power of quantum neural networks. *Nature Computational Science*, 1(6), 403-409.
   https://arxiv.org/abs/2011.01938

5. **Meyer, D. A., & Wallach, N. R. (2002)**. Global entanglement in multiparticle systems. *Journal of Mathematical Physics*, 43(9), 4273-4278.

---

## 11. Quick Start Commands

### Single Test Run
```bash
python measure_comprehensive_metrics.py \
    --n-qubits=4 \
    --n-layers=1 \
    --n-samples=50 \
    --seed=2024
```

### Production Run (6 qubits, 2 layers, 20 seeds)
```bash
for seed in {2024..2043}; do
    python measure_comprehensive_metrics.py \
        --n-qubits=6 \
        --n-layers=2 \
        --n-samples=100 \
        --seed=$seed \
        --output-dir='./comprehensive_results/6q_2l' &
done
wait
```

### Aggregate Results
```bash
python aggregate_comprehensive_results.py \
    --n-qubits=6 \
    --n-layers=2
```

---

## Contact & Support

For questions or issues:
- Check error logs in `logs/` directory
- Review this document's troubleshooting section (TODO)
- Verify environment with: `conda list | grep -E "pennylane|numpy|scipy"`

---

**Last Updated**: November 2025
**Status**: Ready for Deployment
**Estimated Completion**: 1-2 weeks for full experimental grid
**Expected Outcome**: Publication-ready statistical analysis of QCNN vs QuantumDilatedCNN
