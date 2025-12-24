# Comprehensive Metrics Experiment - Quick Run Guide

**Date**: November 13, 2025
**Status**: Ready to run

---

## Experimental Configuration

### Parameters
- **Qubits**: 6, 8, 10, 12 (4 options)
- **Layers**: 1, 2, 3 (3 options)
- **Seeds**: 2024, 2025, 2026, 2027, 2028 (5 seeds)
- **Samples per run**: 100
- **Models**: QCNN, QuantumDilatedCNN

### Total Configurations
- **Array jobs**: 60 (4 qubits × 3 layers × 5 seeds)
- **Total result files**: 120 (60 configs × 2 models)

### Metrics Computed
1. **Meyer-Wallach entanglement** - Overall entangling capability
2. **Concentratable entanglement** - Multipartite entanglement
3. **Distance-dependent mutual information** - Local vs global correlations
4. **Expressibility** - KL/JS divergence from Haar distribution
5. **Effective dimension** - Model capacity via Fisher Information

---

## Quick Start (3 Steps)

### Step 1: Submit SLURM Job
```bash
sbatch run_comprehensive_metrics.sh
```

**Expected output:**
```
Submitted batch job 12345678
```

**Job details:**
- Array size: 0-59 (60 tasks)
- Time limit: 4 hours per task
- Resources: 1 GPU, 4 CPUs per task
- Output logs: `comprehensive_results/logs/`

### Step 2: Monitor Progress
```bash
# Check job status
squeue -u $USER | grep comprehensive

# Check how many completed
ls comprehensive_results/metrics_*/*_comprehensive_*.npy 2>/dev/null | wc -l
# Should eventually reach 120 files (60 configs × 2 models)

# Check recent logs
tail -20 comprehensive_results/logs/metrics_*.out
```

### Step 3: Aggregate Results (After jobs complete)
```bash
python aggregate_comprehensive_results.py
```

**Generates:**
- `comprehensive_all_results.csv` - Raw data for all configurations
- `summary_6q_1l.csv`, `summary_6q_2l.csv`, etc. - Individual summaries
- `comprehensive_summary_all_configs.csv` - Combined publication table

---

## Expected Runtime

### Per Configuration
| Qubits | Layers | Est. Time per Config |
|--------|--------|---------------------|
| 6      | 1      | ~5 minutes          |
| 6      | 2      | ~8 minutes          |
| 6      | 3      | ~12 minutes         |
| 8      | 1      | ~8 minutes          |
| 8      | 2      | ~15 minutes         |
| 8      | 3      | ~25 minutes         |
| 10     | 1      | ~12 minutes         |
| 10     | 2      | ~25 minutes         |
| 10     | 3      | ~45 minutes         |
| 12     | 1      | ~20 minutes         |
| 12     | 2      | ~40 minutes         |
| 12     | 3      | ~60 minutes         |

### Total Walltime
- **Longest config**: 12 qubits, 3 layers (~60 min)
- **Total if sequential**: ~15-20 hours
- **Total with SLURM parallel** (60 GPUs): ~1 hour
- **Realistic** (limited GPUs): ~6-12 hours

---

## File Structure

```
comprehensive_results/
├── logs/
│   ├── metrics_12345678_0.out
│   ├── metrics_12345678_0.err
│   ├── metrics_12345678_1.out
│   └── ... (120 log files)
│
├── metrics_6q_1l/
│   ├── QCNN_comprehensive_6q_1l_seed2024.npy
│   ├── QCNN_comprehensive_6q_1l_seed2025.npy
│   ├── ...
│   ├── QuantumDilatedCNN_comprehensive_6q_1l_seed2024.npy
│   └── summary_6q_1l_seed2024.csv
│
├── metrics_6q_2l/
│   └── ... (10 .npy files + CSVs)
│
├── ... (12 directories total)
│
├── comprehensive_all_results.csv           ← All raw data
├── summary_6q_1l.csv                       ← Statistical summaries
├── summary_6q_2l.csv
├── ... (12 summary files)
└── comprehensive_summary_all_configs.csv   ← Combined table

```

---

## Checking Results

### Quick Check (During Run)
```bash
# Count completed runs
echo "Completed: $(ls comprehensive_results/metrics_*/*_comprehensive_*.npy 2>/dev/null | wc -l) / 120"

# Check which configs are done
for dir in comprehensive_results/metrics_*/; do
    config=$(basename $dir)
    count=$(ls $dir/*_comprehensive_*.npy 2>/dev/null | wc -l)
    echo "$config: $count/10 files (2 models × 5 seeds)"
done
```

### View Sample Result
```python
import numpy as np

# Load a sample result
data = np.load('comprehensive_results/metrics_6q_3l/QCNN_comprehensive_6q_3l_seed2024.npy',
               allow_pickle=True).item()

print("Meyer-Wallach:", data['meyer_wallach']['mean'])
print("Expressibility (KL):", data['expressibility']['kl_divergence'])
print("Effective Dimension:", data['effective_dimension']['effective_dimension'])
```

---

## Troubleshooting

### Problem: Jobs fail immediately
**Check:**
```bash
# View error log
cat comprehensive_results/logs/metrics_*_0.err

# Common issues:
# 1. Conda environment not activated → Check path in script
# 2. Missing dependencies → conda activate ./conda-envs/qml_eeg; pip list
# 3. GPU memory → Reduce n_samples or use smaller qubits first
```

### Problem: Some configs missing
**Resubmit specific array indices:**
```bash
# If jobs 5, 10, 15 failed, resubmit only those
sbatch --array=5,10,15 run_comprehensive_metrics.sh
```

### Problem: Results look wrong
**Validate with small test:**
```bash
# Run single config manually
conda activate ./conda-envs/qml_eeg
python measure_comprehensive_metrics.py \
    --n-qubits=6 \
    --n-layers=1 \
    --n-samples=20 \
    --seed=2024 \
    --output-dir='./test_manual'
```

---

## After Completion

### 1. Verify All Results Present
```bash
python << 'EOF'
from pathlib import Path

expected = 4 * 3 * 5 * 2  # 4 qubits × 3 layers × 5 seeds × 2 models = 120
actual = len(list(Path('comprehensive_results').rglob('*_comprehensive_*.npy')))

print(f"Expected: {expected} files")
print(f"Found: {actual} files")
print(f"Status: {'✓ COMPLETE' if actual == expected else f'✗ MISSING {expected - actual} files'}")
EOF
```

### 2. Run Aggregation
```bash
python aggregate_comprehensive_results.py
```

### 3. View Summary Table
```bash
# View a specific config
cat comprehensive_results/summary_6q_3l.csv

# Or view all in Python
python << 'EOF'
import pandas as pd
df = pd.read_csv('comprehensive_results/comprehensive_summary_all_configs.csv')
print(df.to_string())
EOF
```

### 4. Publication-Ready Table
The aggregation script creates tables with:
- Mean ± SEM for each metric
- Statistical significance (p-values with stars)
- Effect sizes (Cohen's d)
- Difference (Δ) between models

**Example output:**
```
Metric              | QCNN          | Dilated       | Δ       | p-value   | Cohen's d
--------------------|---------------|---------------|---------|-----------|----------
Meyer-Wallach       | 0.6550 ± 0.008| 0.5020 ± 0.009| +0.1530 | <0.001*** | 1.21 (L)
Expressibility (KL) | 0.0450 ± 0.003| 0.0380 ± 0.004| +0.0070 | 0.021*    | 0.42 (S)
Effective Dimension | 12.3 ± 0.8    | 15.7 ± 1.2    | -3.4    | 0.003**   | 0.68 (M)
```

---

## Next Steps for Analysis

After aggregation, you can:

1. **Plot scaling trends** (entanglement vs qubits/layers)
2. **Compare architectures** (QCNN vs Dilated across all configs)
3. **Statistical analysis** (ANOVA, post-hoc tests)
4. **Publication figures** (heatmaps, line plots, violin plots)

---

## Files Created

1. **`run_comprehensive_metrics.sh`** - SLURM array job script (60 tasks)
2. **`measure_comprehensive_metrics.py`** - Main analysis script (already exists)
3. **`aggregate_comprehensive_results.py`** - Results aggregation script
4. **`COMPREHENSIVE_METRICS_RUN_GUIDE.md`** - This guide

---

## Quick Reference Commands

```bash
# Submit job
sbatch run_comprehensive_metrics.sh

# Check progress
squeue -u $USER
ls comprehensive_results/metrics_*/*_comprehensive_*.npy | wc -l

# Aggregate when done
python aggregate_comprehensive_results.py

# View results
cat comprehensive_results/comprehensive_summary_all_configs.csv
```

---

**Ready to run!** Execute `sbatch run_comprehensive_metrics.sh` when ready.

**Last Updated**: November 13, 2025
