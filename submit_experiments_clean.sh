#!/bin/bash
################################################################################
# Clean Submission of QCNN vs QuantumDilatedCNN Experiments
# This script ensures you're using the correctly configured SLURM scripts
################################################################################

ANALYSIS_DIR="/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis"

echo "========================================================================"
echo "CLEAN SUBMISSION - QCNN vs QuantumDilatedCNN Experiments"
echo "========================================================================"
echo ""

# Change to analysis directory
cd "$ANALYSIS_DIR" || {
    echo "✗ ERROR: Cannot change to $ANALYSIS_DIR"
    exit 1
}

echo "Working directory: $(pwd)"
echo ""

# Verify scripts exist and are correct
echo "Step 1: Verifying SLURM scripts..."
bash verify_slurm_settings.sh | grep -E "✓|✗|WRONG"
echo ""

# Ask for confirmation
echo "========================================================================"
echo "Ready to submit experiments"
echo "========================================================================"
echo ""
echo "This will submit:"
echo "  1. CIFAR-10 comparison (1 job)"
echo "  2. COCO comparison (1 job)"
echo "  3. Comprehensive metrics (60 array jobs)"
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Submission cancelled."
    exit 0
fi

echo ""
echo "========================================================================"
echo "Submitting jobs..."
echo "========================================================================"
echo ""

# Submit CIFAR-10
echo "[1/3] Submitting CIFAR-10 comparison..."
CIFAR_JOB=$(sbatch "$ANALYSIS_DIR/scripts/QCNN_Comparison_CIFAR10.sh" 2>&1)
if echo "$CIFAR_JOB" | grep -q "Submitted"; then
    CIFAR_ID=$(echo "$CIFAR_JOB" | awk '{print $4}')
    echo "      ✓ CIFAR-10 job submitted: $CIFAR_ID"
else
    echo "      ✗ CIFAR-10 submission FAILED:"
    echo "        $CIFAR_JOB"
fi
echo ""

# Submit COCO
echo "[2/3] Submitting COCO comparison..."
COCO_JOB=$(sbatch "$ANALYSIS_DIR/scripts/QCNN_Comparison_COCO.sh" 2>&1)
if echo "$COCO_JOB" | grep -q "Submitted"; then
    COCO_ID=$(echo "$COCO_JOB" | awk '{print $4}')
    echo "      ✓ COCO job submitted: $COCO_ID"
else
    echo "      ✗ COCO submission FAILED:"
    echo "        $COCO_JOB"
fi
echo ""

# Submit comprehensive metrics
echo "[3/3] Submitting comprehensive metrics (60 array jobs)..."
METRICS_JOB=$(sbatch "$ANALYSIS_DIR/scripts/run_comprehensive_metrics.sh" 2>&1)
if echo "$METRICS_JOB" | grep -q "Submitted"; then
    METRICS_ID=$(echo "$METRICS_JOB" | awk '{print $4}')
    echo "      ✓ Comprehensive metrics submitted: $METRICS_ID"
    echo "        (Array jobs: ${METRICS_ID}_0 through ${METRICS_ID}_59)"
else
    echo "      ✗ Comprehensive metrics submission FAILED:"
    echo "        $METRICS_JOB"
fi
echo ""

echo "========================================================================"
echo "Submission Summary"
echo "========================================================================"
echo ""
if echo "$CIFAR_JOB" | grep -q "Submitted"; then
    echo "  ✓ CIFAR-10:       $CIFAR_ID"
fi
if echo "$COCO_JOB" | grep -q "Submitted"; then
    echo "  ✓ COCO:           $COCO_ID"
fi
if echo "$METRICS_JOB" | grep -q "Submitted"; then
    echo "  ✓ Comprehensive:  $METRICS_ID (array 0-59)"
fi
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  squeue -j $METRICS_ID  # (for comprehensive metrics)"
echo ""
echo "Check logs in:"
echo "  $ANALYSIS_DIR/results/comprehensive_metrics/logs/"
echo ""
