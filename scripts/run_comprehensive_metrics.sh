#!/bin/bash
#SBATCH --job-name=comprehensive_metrics
#SBATCH --account=m4727_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --array=0-59
#SBATCH --output=/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/results/comprehensive_metrics/logs/metrics_%A_%a.out
#SBATCH --error=/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/results/comprehensive_metrics/logs/metrics_%A_%a.err

##############################################################################
# Comprehensive Quantum Circuit Metrics Analysis
##############################################################################
#
# This script runs comprehensive entanglement and expressibility analysis
# comparing QCNN vs QuantumDilatedCNN architectures.
#
# Configuration:
#   - Qubits: 6, 8, 10, 12
#   - Layers: 1, 2, 3
#   - Seeds: 2024, 2025, 2026, 2027, 2028
#   - Total configurations: 4 × 3 × 5 = 60
#
# Metrics computed:
#   1. Meyer-Wallach entanglement
#   2. Concentratable entanglement
#   3. Distance-dependent mutual information
#   4. Expressibility (KL/JS divergence from Haar)
#   5. Effective dimension (Fisher Information Matrix)
#
# Expected runtime: ~10-30 minutes per configuration
# Total time: ~6-12 hours for all 60 runs
#
##############################################################################

echo "========================================================================"
echo "COMPREHENSIVE QUANTUM METRICS ANALYSIS"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Experimental grid
QUBITS=(6 8 10 12)
LAYERS=(1 2 3)
SEEDS=(2024 2025 2026 2027 2028)

# Decode array task ID into configuration
# Formula: task_id = qubit_idx * (N_LAYERS * N_SEEDS) + layer_idx * N_SEEDS + seed_idx
N_QUBITS=${#QUBITS[@]}
N_LAYERS=${#LAYERS[@]}
N_SEEDS=${#SEEDS[@]}

qubit_idx=$(( SLURM_ARRAY_TASK_ID / (N_LAYERS * N_SEEDS) ))
remainder=$(( SLURM_ARRAY_TASK_ID % (N_LAYERS * N_SEEDS) ))
layer_idx=$(( remainder / N_SEEDS ))
seed_idx=$(( remainder % N_SEEDS ))

n_qubits=${QUBITS[$qubit_idx]}
n_layers=${LAYERS[$layer_idx]}
seed=${SEEDS[$seed_idx]}

echo "Configuration:"
echo "  Qubits: $n_qubits"
echo "  Layers: $n_layers"
echo "  Seed: $seed"
echo "  Samples: 100"
echo ""

# Activate conda environment
echo "Activating conda environment..."
module load python
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

# Change to scripts directory
cd /pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/scripts

# Create output directory
output_dir="/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/results/comprehensive_metrics/metrics_${n_qubits}q_${n_layers}l"
mkdir -p "$output_dir"
mkdir -p "/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/results/comprehensive_metrics/logs"

echo "Output directory: $output_dir"
echo ""

# Run comprehensive metrics analysis
echo "========================================================================"
echo "Running comprehensive metrics analysis..."
echo "========================================================================"
echo ""

python measure_comprehensive_metrics.py \
    --n-qubits=$n_qubits \
    --n-layers=$n_layers \
    --n-samples=100 \
    --seed=$seed \
    --output-dir="$output_dir" \
    --compute-expressibility \
    --compute-eff-dim

exit_code=$?

echo ""
echo "========================================================================"
echo "Job completed"
echo "========================================================================"
echo "Exit code: $exit_code"
echo "End time: $(date)"
echo ""

if [ $exit_code -eq 0 ]; then
    echo "✓ SUCCESS: Results saved to $output_dir"
    echo ""
    echo "Output files:"
    ls -lh "$output_dir"/*seed${seed}*.npy 2>/dev/null || echo "  (No .npy files found)"
    ls -lh "$output_dir"/*seed${seed}*.csv 2>/dev/null || echo "  (No .csv files found)"
else
    echo "✗ FAILED: Check error log for details"
fi

exit $exit_code
