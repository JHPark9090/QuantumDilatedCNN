# QuantumDilatedCNN Comparison Experiments: Complete Status & Cleanup Recommendations

**Date:** November 15, 2025
**Analysis:** Comprehensive review of all QCNN vs QuantumDilatedCNN comparison files and experiments

---

## Executive Summary

You have **two main experiment types** to compare QCNN vs QuantumDilatedCNN:

1. **Performance Experiments** - Classification accuracy on CIFAR-10/COCO
2. **Comprehensive Metrics** - Entanglement, expressibility, effective dimension (6 metrics)

**Current Status:**
- ❌ Performance experiments: **FAILING** (dimension error + expired account)
- ⏸️ Comprehensive metrics: **TEST ONLY** (4q/1l completed, full run not submitted)
- 🗂️ Organization: **MESSY** (files scattered across multiple locations)

---

## Experiment Type 1: Performance Comparison (CIFAR-10/COCO)

### Purpose
Compare QCNN vs QuantumDilatedCNN classification accuracy on image datasets

### Key Files
**Main script:** `QCNN_Comparison.py` (20 KB)
- Implements both QCNN and QuantumDilatedCNN models
- Tests on CIFAR-10 and COCO datasets
- Tracks accuracy, precision, recall, F1 score

**SLURM scripts:**
- `QCNN_Comparison_CIFAR10.sh` - CIFAR-10 experiments
- `QCNN_Comparison_COCO.sh` - COCO experiments
- `submit_qcnn_comparison.sh` - Master submission script

### Current Status: ❌ FAILING

**Jobs in queue:**
- Job 45178516: PD (nodes down/drained)
- Job 45191262: PD (priority queue)
- Job 45191266: PD (priority queue)

**Error:**
```
RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d,
but got input of size: [32, 3072]
```

**Root cause:** Input data is flattened (32, 3072) but conv2d expects image format (32, 3, 32, 32)

**Previous attempts:** 8+ failed runs (see qcnn_comparison_results/ logs)

### Critical Issues

1. **Code bug:** Data preprocessing doesn't reshape to image format
2. **Expired account:** Scripts use `m4138_g` (should be `m4727_g`)
3. **No successful runs:** CIFAR-10 and COCO never completed

---

## Experiment Type 2: Comprehensive Metrics Analysis

### Purpose
Measure 6 quantum metrics comparing QCNN vs QuantumDilatedCNN architectures

### Metrics Measured
1. **Meyer-Wallach Measure** - Overall entanglement capability
2. **Concentratable Entanglement** - Multipartite entanglement
3. **Distance-Dependent Mutual Information** - Local vs global correlations
4. **Expressibility (KL/JS)** - Hilbert space coverage
5. **Effective Dimension** - Model capacity
6. **Bipartite Entropy** - Included in distance analysis

### Key Files
**Implementation:** `measure_comprehensive_metrics.py` (27 KB, 565 lines)
- All 6 metrics implemented
- Tested and validated

**SLURM script:** `run_comprehensive_metrics.sh`
- Array job: 60 configurations
- Qubits: 6, 8, 10, 12
- Layers: 1, 2, 3
- Seeds: 2024-2028
- **Total:** 120 result files (60 configs × 2 models)

**Analysis:** `aggregate_comprehensive_results.py`
- Statistical testing (Welch's t-test)
- Effect sizes (Cohen's d)
- Publication-ready tables

### Current Status: ⏸️ TEST ONLY

**Completed:**
- ✅ Test run: 4 qubits, 1 layer, seed 2024 (successful)
- ✅ Validation: All 6 metrics working correctly

**Not completed:**
- ❌ Full experimental grid (6-12 qubits, 1-3 layers) NOT submitted
- ❌ Production run never executed

**Test results location:** `/pscratch/sd/j/junghoon/test_comprehensive_metrics/`
- 2 .npy files (QCNN + QuantumDilatedCNN)
- 1 summary CSV

### Critical Issues

1. **Expired account:** Script uses `m4138_g` (should be `m4727_g`)
2. **Never submitted:** Full grid (60 configs) never run
3. **Test only:** Only validation completed, no production data

---

## File Organization Issues

### Current Scattered Structure ❌

**Main directory** (`/pscratch/sd/j/junghoon/`):
```
❌ QCNN_Comparison.py               (duplicate)
❌ QCNN_Comparison_CIFAR10.sh       (duplicate)
❌ QCNN_Comparison_COCO.sh          (duplicate)
❌ submit_qcnn_comparison.sh        (old)
❌ run_comprehensive_metrics.sh     (duplicate)
❌ measure_comprehensive_metrics.py (duplicate)
❌ aggregate_comprehensive_results.py (duplicate)
❌ 13+ markdown documentation files (scattered)
❌ measure_entanglement.py          (old, superseded)
❌ plot_entanglement_results.py     (old)
❌ measure_local_global_entanglement.py (old)
❌ plot_local_global_entanglement.py (old)
❌ cleanup_old_scripts.sh
❌ setup_qcnn_analysis_folder.sh
```

**Organized folder** (`QuantumDilatedCNN_Analysis/`):
```
✅ scripts/                         (clean organization)
   ├── QCNN_Comparison.py
   ├── QCNN_Comparison_CIFAR10.sh
   ├── QCNN_Comparison_COCO.sh
   ├── measure_comprehensive_metrics.py
   ├── run_comprehensive_metrics.sh
   └── aggregate_comprehensive_results.py

✅ docs/                            (documentation)
   ├── ROBUST_ENTANGLEMENT_ANALYSIS_PLAN.md
   ├── COMPREHENSIVE_METRICS_RUN_GUIDE.md
   ├── READY_TO_RUN_SUMMARY.md
   └── COMPREHENSIVE_METRICS_TEST_RESULTS.md

✅ results/                         (organized output)
   ├── cifar10/
   ├── coco/
   └── comprehensive_metrics/
```

**Old results folders:**
```
❌ qcnn_comparison_results/         (failed runs, 8+ attempts)
❌ comprehensive_results/            (only logs/)
❌ test_comprehensive_metrics/       (test data only)
```

### Recommended Structure ✅

```
QuantumDilatedCNN_Analysis/         ← PRIMARY LOCATION
├── scripts/                        ← All Python/SLURM scripts
├── docs/                           ← All documentation
└── results/                        ← All outputs
    ├── cifar10/
    ├── coco/
    └── comprehensive_metrics/

quantum_hydra_mamba_repo/           ← SEPARATE (already organized)
└── (quantum hydra/mamba experiments)
```

---

## Detailed File Inventory

### Scripts (Main Directory)

| File | Status | Action |
|------|--------|--------|
| `QCNN_Comparison.py` | Duplicate | ❌ DELETE (use QuantumDilatedCNN_Analysis/scripts/) |
| `QCNN_Comparison_CIFAR10.sh` | Duplicate | ❌ DELETE |
| `QCNN_Comparison_COCO.sh` | Duplicate | ❌ DELETE |
| `submit_qcnn_comparison.sh` | Old | ❌ DELETE |
| `monitor_qcnn_comparison.sh` | Generated | ❌ DELETE |
| `run_comprehensive_metrics.sh` | Duplicate | ❌ DELETE |
| `measure_comprehensive_metrics.py` | Duplicate | ❌ DELETE |
| `aggregate_comprehensive_results.py` | Duplicate | ❌ DELETE |
| `measure_entanglement.py` | Old (superseded) | ❌ DELETE |
| `plot_entanglement_results.py` | Old | ❌ DELETE |
| `measure_local_global_entanglement.py` | Old | ❌ DELETE |
| `plot_local_global_entanglement.py` | Old | ❌ DELETE |
| `setup_qcnn_analysis_folder.sh` | Setup script | ❌ DELETE (already set up) |
| `cleanup_old_scripts.sh` | Cleanup script | ❌ DELETE (after using) |

### Documentation (Main Directory)

| File | Status | Action |
|------|--------|--------|
| `QCNN_COMPARISON_GUIDE.md` | Duplicate | ❌ DELETE |
| `COMPREHENSIVE_METRICS_RUN_GUIDE.md` | Duplicate | ❌ DELETE |
| `COMPREHENSIVE_METRICS_TEST_RESULTS.md` | Duplicate | ❌ DELETE |
| `READY_TO_RUN_SUMMARY.md` | Duplicate | ❌ DELETE |
| `ROBUST_ENTANGLEMENT_ANALYSIS_PLAN.md` | Duplicate | ❌ DELETE |
| `ENTANGLEMENT_ANALYSIS_SUMMARY.md` | Old | ❌ DELETE |
| `LOCAL_VS_GLOBAL_ENTANGLEMENT_ANALYSIS.md` | Old | ❌ DELETE |
| `FOLDER_SETUP_EXPLANATION.md` | Setup notes | ❌ DELETE |
| `QCNN_CORRECTIONS.md` | Old fixes | ⚠️ REVIEW THEN DELETE |
| `QCNN_HIERARCHICAL_README.md` | Old | ❌ DELETE |
| `ENTANGLEMENT_STRUCTURES_EXPLAINED.md` | Reference | ✅ KEEP or move to docs/ |
| `TrueQCNN_*.md` (3 files) | Different project | ⚠️ SEPARATE PROJECT |
| `VQC_vs_QCNN_CLARIFICATION.md` | Reference | ⚠️ REVIEW |

### Results Folders

| Folder | Contents | Action |
|--------|----------|--------|
| `qcnn_comparison_results/` | 8+ failed CIFAR-10/COCO runs | ❌ DELETE (all failures) |
| `comprehensive_results/` | Only empty logs/ | ❌ DELETE |
| `test_comprehensive_metrics/` | 4q/1l test only | ⚠️ ARCHIVE THEN DELETE |
| `QuantumDilatedCNN_Analysis/results/` | Organized structure | ✅ KEEP (primary location) |

---

## Critical Fixes Needed

### Fix 1: Update Account in All Scripts

**Files to update:**
- `QuantumDilatedCNN_Analysis/scripts/QCNN_Comparison_CIFAR10.sh`
- `QuantumDilatedCNN_Analysis/scripts/QCNN_Comparison_COCO.sh`
- `QuantumDilatedCNN_Analysis/scripts/run_comprehensive_metrics.sh`

**Change:**
```bash
#SBATCH --account=m4138_g    # ❌ EXPIRED
#SBATCH --account=m4727_g    # ✅ ACTIVE
```

### Fix 2: Fix QCNN_Comparison.py Data Preprocessing

**Problem:** Input to model is flattened `(batch, 3072)` but conv2d needs `(batch, 3, 32, 32)`

**Location:** `QuantumDilatedCNN_Analysis/scripts/QCNN_Comparison.py`

**Need to add:** Reshape logic before passing to model

### Fix 3: Clean Up Duplicate Files

**Execute cleanup** (see Cleanup Script section below)

---

## Recommended Actions

### Immediate (Before Running Experiments)

1. **Fix account** in all 3 SLURM scripts → `m4727_g`
2. **Fix data preprocessing** in `QCNN_Comparison.py`
3. **Delete duplicates** from main directory
4. **Archive or delete** old failed results

### Short-term (Running Experiments)

1. **Test locally first:**
   ```bash
   python QuantumDilatedCNN_Analysis/scripts/QCNN_Comparison.py \
       --dataset=cifar10 \
       --n-epochs=2 \
       --batch-size=4 \
       --models qcnn
   ```

2. **Submit comprehensive metrics** (if data preprocessing fixed):
   ```bash
   sbatch QuantumDilatedCNN_Analysis/scripts/run_comprehensive_metrics.sh
   ```

3. **Monitor and verify** one successful run before submitting all

---

## Cleanup Script

```bash
#!/bin/bash
# QCNN Experiments Cleanup Script

cd /pscratch/sd/j/junghoon

echo "Creating backup of important test results..."
mkdir -p QuantumDilatedCNN_Archive
cp -r test_comprehensive_metrics/ QuantumDilatedCNN_Archive/
cp COMPREHENSIVE_METRICS_TEST_RESULTS.md QuantumDilatedCNN_Archive/

echo "Deleting duplicate scripts..."
rm -f QCNN_Comparison.py
rm -f QCNN_Comparison_CIFAR10.sh
rm -f QCNN_Comparison_COCO.sh
rm -f submit_qcnn_comparison.sh
rm -f monitor_qcnn_comparison.sh
rm -f run_comprehensive_metrics.sh
rm -f measure_comprehensive_metrics.py
rm -f aggregate_comprehensive_results.py

echo "Deleting old/superseded scripts..."
rm -f measure_entanglement.py
rm -f plot_entanglement_results.py
rm -f measure_local_global_entanglement.py
rm -f plot_local_global_entanglement.py
rm -f setup_qcnn_analysis_folder.sh
rm -f cleanup_old_scripts.sh

echo "Deleting duplicate documentation..."
rm -f QCNN_COMPARISON_GUIDE.md
rm -f COMPREHENSIVE_METRICS_RUN_GUIDE.md
rm -f COMPREHENSIVE_METRICS_TEST_RESULTS.md
rm -f READY_TO_RUN_SUMMARY.md
rm -f ROBUST_ENTANGLEMENT_ANALYSIS_PLAN.md
rm -f ENTANGLEMENT_ANALYSIS_SUMMARY.md
rm -f LOCAL_VS_GLOBAL_ENTANGLEMENT_ANALYSIS.md
rm -f FOLDER_SETUP_EXPLANATION.md
rm -f QCNN_HIERARCHICAL_README.md

echo "Deleting failed results..."
rm -rf qcnn_comparison_results/
rm -rf comprehensive_results/
rm -rf test_comprehensive_metrics/

echo "Cleanup complete!"
echo ""
echo "Remaining in main directory:"
ls -1 *.md *.sh *.py 2>/dev/null | wc -l
echo "files"
echo ""
echo "Primary location:"
echo "cd QuantumDilatedCNN_Analysis/"
```

---

## File Locations After Cleanup

### Main Directory (`/pscratch/sd/j/junghoon/`)
```
quantum_hydra_mamba_repo/           # Quantum Hydra/Mamba experiments
QuantumDilatedCNN_Analysis/         # QCNN comparison experiments
QuantumDilatedCNN_Archive/                       # Backup of test results
[Other non-QCNN files]              # DNA, MNIST, EEG experiments
```

### QuantumDilatedCNN_Analysis/ (All QCNN Work)
```
scripts/
  ├── QCNN_Comparison.py            ← (needs data fix)
  ├── QCNN_Comparison_CIFAR10.sh    ← (needs account fix)
  ├── QCNN_Comparison_COCO.sh       ← (needs account fix)
  ├── measure_comprehensive_metrics.py
  ├── run_comprehensive_metrics.sh  ← (needs account fix)
  └── aggregate_comprehensive_results.py

docs/
  ├── ROBUST_ENTANGLEMENT_ANALYSIS_PLAN.md
  ├── COMPREHENSIVE_METRICS_RUN_GUIDE.md
  ├── READY_TO_RUN_SUMMARY.md
  └── COMPREHENSIVE_METRICS_TEST_RESULTS.md

results/
  ├── cifar10/        (empty, awaiting successful runs)
  ├── coco/           (empty, awaiting successful runs)
  └── comprehensive_metrics/  (awaiting full grid submission)
```

---

## Summary

**Problem:** Scattered files + expired account + code bugs = No successful runs

**Solution:**
1. ✅ Clean up duplicates
2. ✅ Fix account → m4727_g
3. ✅ Fix data preprocessing in QCNN_Comparison.py
4. ✅ Use QuantumDilatedCNN_Analysis/ as single source of truth

**After fixes:** Ready to run both experiment types successfully!

---

**Created:** November 15, 2025
**Purpose:** Complete audit before running production experiments
