# SLURM Configuration Fix Applied

**Date**: November 13, 2025
**Issue**: GPU shared queue requires 32 CPUs per GPU

---

## Error Encountered

```
sbatch: error: Logical queue gpu_shared_ss11 requires you to request 32.0 cores
per GPU, job requested 4 cores (adjusted for memory) for 1 GPUs
sbatch: error: Batch job submission failed: Unspecified error
```

---

## Root Cause

The Perlmutter `gpu_shared_ss11` queue (when using `--qos=shared` with `--constraint=gpu&hbm80g`) has a strict requirement:
- **Must request exactly 32 CPUs per GPU**
- Original script requested only 4 CPUs

---

## Fix Applied

### Changed Settings in `run_comprehensive_metrics.sh`

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `--cpus-per-task` | 4 | **32** | Required by gpu_shared queue |
| `--time` | 04:00:00 | **48:00:00** | Allow sufficient time for 12q/3l configs |

### Updated Script Header

**Before:**
```bash
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
```

**After:**
```bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
```

---

## Files Updated

1. ✅ `/pscratch/sd/j/junghoon/run_comprehensive_metrics.sh` (original)
2. ✅ `/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/scripts/run_comprehensive_metrics.sh` (analysis folder)

Both files now have identical correct settings.

---

## Verification

```bash
# Check SLURM settings in analysis folder script
head -11 QuantumDilatedCNN_Analysis/scripts/run_comprehensive_metrics.sh

# Should show:
# --time=48:00:00
# --cpus-per-task=32
```

---

## Ready to Resubmit

The script is now configured correctly for Perlmutter's GPU shared queue.

### Resubmit Options

**Option 1: Submit comprehensive metrics only**
```bash
cd /pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis
sbatch scripts/run_comprehensive_metrics.sh
```

**Option 2: Submit all experiments**
```bash
cd /pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis
bash run_all.sh
```

---

## Expected Behavior After Fix

✅ Job submission will succeed
✅ Array job will spawn 60 tasks (array indices 0-59)
✅ Each task runs with 32 CPUs and 1 GPU
✅ Maximum runtime: 48 hours per task

---

## Additional Notes

### Why 48 Hours?

The longest configuration (12 qubits, 3 layers) may take:
- ~60 minutes per run with 100 samples
- Includes expressibility (500 pairs) and effective dimension (Fisher matrix)
- 48 hours provides safe buffer for potential slowdowns

### Why 32 CPUs?

- Perlmutter GPU shared queue requirement (non-negotiable)
- Can potentially speed up some NumPy operations
- Most computation is on GPU, so CPUs are mostly idle

### Can I Use Fewer CPUs?

No, not with `--qos=shared` and `--constraint=gpu&hbm80g` combination. Options:
- ✅ Use 32 CPUs (current solution)
- ❌ Different QoS (may have different limits/availability)
- ❌ Different constraint (may not have 80GB HBM GPUs)

---

## Status

✅ **FIXED AND READY TO RUN**

You can now successfully submit the comprehensive metrics experiment.

---

**Fixed by**: Claude Code
**Date**: November 13, 2025
