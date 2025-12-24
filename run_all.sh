#!/bin/bash
################################################################################
# Run All QCNN vs QuantumDilatedCNN Experiments
################################################################################

ANALYSIS_DIR="/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis"
cd "$ANALYSIS_DIR"

echo "========================================================================"
echo "Submitting all QCNN vs QuantumDilatedCNN experiments"
echo "========================================================================"
echo ""

# Submit CIFAR-10
echo "1. Submitting CIFAR-10 comparison..."
CIFAR_JOB=$(sbatch scripts/QCNN_Comparison_CIFAR10.sh | awk '{print $4}')
echo "   Job ID: $CIFAR_JOB"

# Submit COCO
echo "2. Submitting COCO comparison..."
COCO_JOB=$(sbatch scripts/QCNN_Comparison_COCO.sh | awk '{print $4}')
echo "   Job ID: $COCO_JOB"

# Submit comprehensive metrics
echo "3. Submitting comprehensive metrics (60 array jobs)..."
METRICS_JOB=$(sbatch scripts/run_comprehensive_metrics.sh | awk '{print $4}')
echo "   Job ID: $METRICS_JOB"

echo ""
echo "========================================================================"
echo "All jobs submitted!"
echo "========================================================================"
echo ""
echo "Job IDs:"
echo "  CIFAR-10:            $CIFAR_JOB"
echo "  COCO:                $COCO_JOB"
echo "  Comprehensive:       $METRICS_JOB"
echo ""
echo "Monitor with: squeue -u $USER"
echo ""

