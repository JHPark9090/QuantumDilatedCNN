# QCNN vs QuantumDilatedCNN Analysis

This folder contains all scripts and documentation for comparing QCNN and QuantumDilatedCNN architectures.

## Directory Structure

```
QuantumDilatedCNN_Analysis/
├── scripts/              # Python and SLURM scripts
│   ├── QCNN_Comparison.py
│   ├── QCNN_Comparison_CIFAR10.sh
│   ├── QCNN_Comparison_COCO.sh
│   ├── measure_comprehensive_metrics.py
│   ├── run_comprehensive_metrics.sh
│   └── aggregate_comprehensive_results.py
│
├── docs/                 # Documentation
│   ├── ROBUST_ENTANGLEMENT_ANALYSIS_PLAN.md
│   ├── COMPREHENSIVE_METRICS_RUN_GUIDE.md
│   ├── READY_TO_RUN_SUMMARY.md
│   └── COMPREHENSIVE_METRICS_TEST_RESULTS.md
│
└── results/              # Output directory
    ├── cifar10/          # CIFAR-10 comparison results
    ├── coco/             # COCO comparison results
    ├── entanglement/     # Entanglement analysis results
    └── comprehensive_metrics/  # All 6 metrics results
        ├── logs/         # SLURM job logs
        └── metrics_*q_*l/  # Results by configuration

```

## Quick Start

All scripts are configured to run from this directory with absolute paths.

### 1. Image Classification Experiments

```bash
cd /pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis

# CIFAR-10
sbatch scripts/QCNN_Comparison_CIFAR10.sh

# COCO
sbatch scripts/QCNN_Comparison_COCO.sh
```

### 2. Comprehensive Metrics (Entanglement, Expressibility, Effective Dimension)

```bash
cd /pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis

# Submit array job (60 configurations)
sbatch scripts/run_comprehensive_metrics.sh

# After completion, aggregate results
python scripts/aggregate_comprehensive_results.py
```

### 3. Monitor Progress

```bash
# Check jobs
squeue -u $USER

# Count completed metrics files (target: 120)
ls results/comprehensive_metrics/metrics_*/*.npy 2>/dev/null | wc -l

# View logs
tail -f results/comprehensive_metrics/logs/metrics_*.out
```

## Results Location

All results are saved in the `results/` directory:
- **CIFAR-10**: `results/cifar10/`
- **COCO**: `results/coco/`
- **Comprehensive metrics**: `results/comprehensive_metrics/`
- **Aggregated tables**: `results/comprehensive_metrics/*.csv`

## Documentation

See the `docs/` directory for detailed guides:
- `READY_TO_RUN_SUMMARY.md` - Quick reference
- `COMPREHENSIVE_METRICS_RUN_GUIDE.md` - Step-by-step guide
- `ROBUST_ENTANGLEMENT_ANALYSIS_PLAN.md` - Full experimental plan

## Notes

- All scripts use absolute paths and can be run from anywhere
- Conda environment is automatically activated by SLURM scripts
- Results are organized by experiment type

