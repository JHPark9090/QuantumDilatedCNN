#!/bin/bash
#SBATCH -A m4727_g
#SBATCH -J QCNN_Comparison_CIFAR10
#SBATCH -C gpu&hbm80g
#SBATCH --qos shared
#SBATCH -t 24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --chdir='/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/scripts'
#SBATCH --output=/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/results/cifar10/QCNN_Comparison_CIFAR10_%j.out
#SBATCH -e /pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/results/cifar10/QCNN_Comparison_CIFAR10_%j.err
#SBATCH --mail-user=utopie9090@snu.ac.kr

set +x

echo "=========================================="
echo "QCNN vs Quantum Dilated CNN - CIFAR-10"
echo "=========================================="
echo ""
echo "Comparing two quantum CNN architectures:"
echo "  1. QCNN (Cong et al. 2019) - Nearest-neighbor entanglement"
echo "  2. Quantum Dilated CNN - Non-adjacent entanglement"
echo ""

cd /pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/scripts
module load python
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

# Create output directories
mkdir -p /pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/results/cifar10
mkdir -p /pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/results/cifar10/checkpoints

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Device: GPU (CUDA)"
echo ""

# Configuration:
# - Dataset: CIFAR-10 (10 classes, 32x32x3 images)
# - Qubits: 8
# - Layers: 2
# - Optimizer: Adam with ReduceLROnPlateau
# - Epochs: 50 (with early stopping, patience=10)
# - Batch size: 32
# - Seeds: 2024, 2025, 2026 (3 seeds for statistical robustness)
# - Resume: Enabled for fault tolerance

python /pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/scripts/QCNN_Comparison.py \
    --dataset=cifar10 \
    --n-qubits=8 \
    --n-layers=2 \
    --models qcnn dilated \
    --n-epochs=50 \
    --batch-size=32 \
    --lr=1E-3 \
    --wd=1E-4 \
    --seeds 2024 2025 2026 \
    --patience=10 \
    --resume \
    --output-dir='/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/results/cifar10' \
    --job-id='cifar10_comparison'
    # --wandb \
    # --wandb-project='QCNN_Comparison' \
    # --wandb-entity='QML_Research'

echo ""
echo "=========================================="
echo "CIFAR-10 Comparison Complete"
echo "End Time: $(date)"
echo "=========================================="
