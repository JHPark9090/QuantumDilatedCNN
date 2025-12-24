#!/bin/bash
################################################################################
# Verify SLURM Settings in All Scripts
################################################################################

ANALYSIS_DIR="/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis"

echo "========================================================================"
echo "VERIFYING SLURM SETTINGS IN ALL SCRIPTS"
echo "========================================================================"
echo ""

for script in QCNN_Comparison_CIFAR10.sh QCNN_Comparison_COCO.sh run_comprehensive_metrics.sh; do
    echo "==================================================================="
    echo "Script: $script"
    echo "==================================================================="

    filepath="$ANALYSIS_DIR/scripts/$script"

    if [ ! -f "$filepath" ]; then
        echo "  ✗ File not found: $filepath"
        echo ""
        continue
    fi

    # Extract key SLURM parameters
    echo "  SLURM Parameters:"
    cpus=$(grep "cpus-per-task" "$filepath" | head -1)
    time=$(grep -E "#SBATCH.*time=|#SBATCH.*-t " "$filepath" | head -1)
    qos=$(grep "qos" "$filepath" | head -1)
    gpus=$(grep -E "gpus|gpus-per-task" "$filepath" | head -1)
    mem=$(grep "#SBATCH.*mem" "$filepath" | head -1)

    if [ -n "$cpus" ]; then
        echo "    $cpus"
        # Check if it's 32
        if echo "$cpus" | grep -q "32"; then
            echo "      ✓ Correct (32 CPUs)"
        else
            echo "      ✗ WRONG! Should be 32"
        fi
    else
        echo "    ✗ No cpus-per-task found"
    fi

    if [ -n "$time" ]; then
        echo "    $time"
        if echo "$time" | grep -q "48:00:00"; then
            echo "      ✓ Correct (48 hours)"
        else
            echo "      ⚠ Not 48:00:00"
        fi
    else
        echo "    ✗ No time limit found"
    fi

    if [ -n "$qos" ]; then
        echo "    $qos"
    fi

    if [ -n "$gpus" ]; then
        echo "    $gpus"
    fi

    if [ -n "$mem" ]; then
        echo "    $mem"
        echo "      ⚠ WARNING: Memory specification may interfere with CPU allocation"
    else
        echo "    No memory specification (good)"
    fi

    echo ""
done

echo "========================================================================"
echo "RECOMMENDATION"
echo "========================================================================"
echo ""
echo "All scripts should have:"
echo "  - --cpus-per-task=32  (REQUIRED for gpu_shared queue)"
echo "  - --time=48:00:00 or -t 48:00:00  (for long runs)"
echo "  - NO --mem specification  (let SLURM auto-assign)"
echo ""
echo "If any script shows 'WRONG', it needs to be fixed."
echo ""
