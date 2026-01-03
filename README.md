# Quantum Dilated CNN vs QCNN: A Comparative Study

## Study Rationale

### Background

Quantum Convolutional Neural Networks (QCNNs) have emerged as a promising approach for quantum machine learning, particularly for image classification tasks. The original QCNN architecture proposed by Cong et al. (2019) uses **nearest-neighbor entanglement**, where quantum gates only connect adjacent qubits in a linear topology.

### Hypothesis

We hypothesize that **dilated (non-adjacent) entanglement patterns** can improve quantum circuit expressibility and classification performance by:

1. **Increasing global connectivity**: Dilated connections allow distant qubits to interact directly, potentially capturing longer-range correlations in data
2. **Reducing circuit depth**: Global information can propagate faster through dilated connections
3. **Improving entanglement distribution**: Non-local entanglement may lead to more uniform entanglement across the quantum state

### Research Questions

1. Does dilated entanglement improve classification accuracy compared to nearest-neighbor QCNN?
2. How do entanglement metrics (Meyer-Wallach, concentratable entanglement) differ between architectures?
3. Does dilated connectivity improve circuit expressibility (coverage of Hilbert space)?

### Architectures Compared

| Architecture | Entanglement Pattern | Reference |
|--------------|---------------------|-----------|
| **QCNN** | Nearest-neighbor (i, i+1) | Cong et al. (2019) |
| **QuantumDilatedCNN** | Dilated (i, i+2) | This study |

---

## Directory Structure

```
QuantumDilatedCNN_Analysis/
├── README.md                              # This file
├── run_all.sh                             # Master script to submit all jobs
├── verify_slurm_settings.sh               # Verify SLURM configurations
├── submit_experiments_clean.sh            # Safe submission wrapper
├── Quantum_Dilated_CNN.pdf                # Reference paper
│
├── scripts/                               # Python and SLURM scripts
│   ├── QCNN_Comparison.py                 # Main comparison script
│   ├── measure_comprehensive_metrics.py   # Quantum circuit metrics
│   ├── aggregate_comprehensive_results.py # Results aggregation
│   ├── Load_Image_Datasets.py             # Data loaders
│   ├── QCNN_Comparison_CIFAR10.sh         # CIFAR-10 SLURM job
│   ├── QCNN_Comparison_COCO.sh            # COCO SLURM job
│   └── run_comprehensive_metrics.sh       # Metrics SLURM array job
│
├── results/                               # Output directory
│   ├── cifar10/                           # CIFAR-10 results
│   │   └── checkpoints/                   # Training checkpoints
│   ├── coco/                              # COCO results
│   │   └── checkpoints/                   # Training checkpoints
│   ├── comprehensive_metrics/             # Circuit metrics results (production)
│   │   └── logs/                          # SLURM job logs
│   └── preliminary_results/               # Early experiments (4-6 qubits)
│       ├── entanglement_results/          # Meyer-Wallach, Concentratable
│       └── local_global_results/          # Distance-dependent MI
│
└── docs/                                  # Documentation
    ├── READY_TO_RUN_SUMMARY.md
    ├── COMPREHENSIVE_METRICS_RUN_GUIDE.md
    └── ...
```

---

## Python Files

### Model and Training Scripts

| File | Description |
|------|-------------|
| `scripts/QCNN_Comparison.py` | Main training script comparing QCNN vs QuantumDilatedCNN on image classification. Includes early stopping, checkpoint/resume, multiple seeds support, and learning rate scheduling. |
| `scripts/measure_comprehensive_metrics.py` | Computes 6 quantum circuit metrics for both architectures across multiple configurations. |
| `scripts/aggregate_comprehensive_results.py` | Aggregates results from all metric runs, computes statistics (mean, std, t-tests, Cohen's d). |
| `scripts/Load_Image_Datasets.py` | Data loaders for CIFAR-10, COCO, MNIST, Fashion-MNIST, and EEG datasets. |

### Model Architecture (in `QCNN_Comparison.py`)

```
ClassicalFeatureExtractor
├── Conv2d(in_channels, 16, 3x3) + ReLU + MaxPool
├── Conv2d(16, 32, 3x3) + ReLU + MaxPool
└── Linear(flattened_size, n_qubits) + Tanh

QCNN / QuantumDilatedCNN
├── ClassicalFeatureExtractor (dimension reduction)
├── AngleEmbedding (encode features into quantum state)
├── Convolutional Layers (U3 + IsingZZ/YY/XX gates)
├── Pooling Layers (mid-circuit measurement)
├── ArbitraryUnitary (final transformation)
└── Linear(1, num_classes) (classification head)
```

---

## Quantum Circuit Metrics

The `measure_comprehensive_metrics.py` script computes 6 key metrics that characterize quantum circuit behavior:

| Metric | Description | Range | Higher = |
|--------|-------------|-------|----------|
| **Meyer-Wallach** | Overall entangling capability | [0, 1] | More entangled |
| **Concentratable Entanglement** | Multipartite entanglement measure | [0, 1] | More entangled |
| **Distance-Dependent MI** | Mutual information by qubit distance | [0, 2] | More correlated |
| **Expressibility (KL)** | KL divergence from Haar distribution | [0, ∞) | Less expressive |
| **Expressibility (JS)** | Jensen-Shannon divergence | [0, 1] | Less expressive |
| **Effective Dimension** | Model capacity (Fisher Information) | [0, ∞) | Higher capacity |
| **Effective Volume**(*draft*) | Entangling gate volume contributing to an observable | [0, ∞) | More noise-sensitive /higher classical computational cost |

### Detailed Metric Explanations

#### 1. Meyer-Wallach Measure (Q)
**What it measures**: The average entanglement across all qubits in the system.

**Formula**: Q = 2(1 - (1/n)∑ᵢTr(ρᵢ²)), where ρᵢ is the reduced density matrix of qubit i.

**Interpretation**:
- Q = 0: Product state (no entanglement)
- Q = 1: Maximally entangled state
- Higher Q indicates the circuit creates more overall entanglement

**Reference**: Meyer & Wallach (2002), "Global entanglement in multiparticle systems"

#### 2. Concentratable Entanglement (CE)
**What it measures**: Multipartite entanglement that can be "concentrated" into Bell pairs.

**Formula**: CE = 1 - (1/2ⁿ)∑ₐTr(ρₐ²), summed over all possible subsystems α.

**Interpretation**:
- Captures genuine multipartite entanglement (not just bipartite)
- Higher CE means entanglement is distributed across many qubits simultaneously
- Important for quantum error correction and quantum algorithms

**Reference**: Beckey et al. (2021), "Computable and operationally meaningful multipartite entanglement measures"

#### 3. Distance-Dependent Mutual Information
**What it measures**: Quantum correlations between qubit pairs as a function of their distance.

**Formula**: I(i:j) = S(ρᵢ) + S(ρⱼ) - S(ρᵢⱼ), where S is von Neumann entropy.

**Interpretation**:
- **d=1 (adjacent)**: Measures nearest-neighbor correlations
- **d=2 (dilated)**: Measures next-nearest-neighbor correlations
- **d>2 (global)**: Measures long-range correlations

**Why this matters for our study**:
- QCNN uses nearest-neighbor gates → should show high MI at d=1
- QuantumDilatedCNN uses dilated gates → should show high MI at d=2

#### 4. Expressibility (KL/JS Divergence)
**What it measures**: How uniformly the circuit can cover the Hilbert space compared to Haar-random states.

**Formula**: Expr = D_KL(P_circuit || P_Haar), where P is the fidelity distribution.

**Interpretation**:
- Lower divergence = more expressive circuit
- Expressibility = 0 means the circuit can generate any quantum state
- High expressibility is desirable for variational quantum algorithms

**Reference**: Sim et al. (2019), "Expressibility and entangling capability of parameterized quantum circuits"

#### 5. Effective Dimension
**What it measures**: The model's capacity to fit different functions, based on Fisher Information.

**Formula**: d_eff = Tr(F) / ||F||_F, where F is the Fisher Information Matrix.

**Interpretation**:
- Higher effective dimension = more trainable parameters are "active"
- Related to avoiding barren plateaus in training
- Indicates how much of the parameter space is being utilized

**Reference**: Abbas et al. (2021), "The power of quantum neural networks"

#### 6. Bipartite Entropy
**What it measures**: Entanglement entropy when the system is divided into two parts at distance d.

**Formula**: S = -Tr(ρ_A log₂ ρ_A), where ρ_A is the reduced density matrix of subsystem A.

**Interpretation**:
- Maximum entropy = log₂(dim) indicates maximal entanglement across the bipartition
- Useful for understanding how entanglement scales with system size

#### 7. Effective Volume *(draft)*
> **Note (temporary):**  
> By computing V_eff of the circuit, we can establish a bridge between experimental signal-to-noise degradation and the classical computational hardness of simulating the same observable, in a way that allows hardware-level difficulty to be reflected within classical simulations.

**What it measures**: The number of entangling two-qubit gates that  contribute to the expectation value of a specific observable O.

**Formula**:  
$F_{\mathrm{eff}} \sim \exp(-\varepsilon V_{\mathrm{eff}})$  

where $\varepsilon$ is the dominant error per two-qubit entangling gate, $V_{\mathrm{eff}}$ is the effective volume, and $F_{\mathrm{eff}}$ is the observable-dependent effective fidelity.  

This formula represents the scaling relation between the effective fidelity of an observable and the effective volume.

> **Notes(temporary):**  
> - Unlike other metrics, I am not sure this quantity can be computed solely from the state vector.
> - In case the entanglement is non-local or genuinely multipartite, I assume it can be decomposed into a sequence of two-qubit entangling gates and count its contribution accordingly.

**Interpretation**:
- different measurements on the same circuit can have different V_eff
- Larger V_eff implies Higher sensitivity to hardware noise (lower effective fidelity) and Larger classical simulation cost due to increased effective area

**Reference**: Kechedzhi, K., et al. (2024). Effective quantum volume, fidelity and computational cost of noisy quantum processing experiments. Future Generation Computer Systems, 153, 431–441.

---

## Preliminary Results

Early experiments (4-6 qubits) validated the architectural differences between QCNN and QuantumDilatedCNN.

### Distance-Dependent Mutual Information (6 qubits, 2 layers)

| Distance | QCNN | QuantumDilatedCNN | Interpretation |
|----------|------|-------------------|----------------|
| d=1 (adjacent) | **0.225 ± 0.234** | **0.000 ± 0.000** | QCNN has local correlations; Dilated has none |
| d=2 (dilated) | 0.239 ± 0.232 | **0.578 ± 0.373** | Dilated shows strong d=2 correlations |
| d=3 | 0.108 ± 0.129 | 0.000 ± 0.000 | |
| d=4 | 0.123 ± 0.172 | **0.501 ± 0.325** | Dilated also shows d=4 correlations |
| d=5 | 0.000 ± 0.000 | 0.000 ± 0.000 | |

**Key Finding**: The entanglement pattern exactly matches the architectural design:
- **QCNN**: Entangles adjacent qubits (i, i+1) → high MI at d=1
- **QuantumDilatedCNN**: Entangles non-adjacent qubits (i, i+2) → high MI at d=2, d=4

### Meyer-Wallach Entanglement

| Configuration | QCNN | QuantumDilatedCNN |
|---------------|------|-------------------|
| 4 qubits, 1 layer | 0.462 ± 0.230 | 0.330 ± 0.157 |
| 6 qubits, 2 layers | 0.655 ± 0.120 | 0.502 ± 0.130 |

**Observation**: QCNN shows higher overall entanglement, but this is because nearest-neighbor gates create more total entangling operations. The key difference is in the *pattern* of entanglement, not the total amount.

### Comprehensive Metrics (4 qubits, 1 layer, seed=2024)

| Metric | QCNN | QuantumDilatedCNN |
|--------|------|-------------------|
| Meyer-Wallach | 0.478 ± 0.200 | 0.334 ± 0.165 |
| Concentratable | 0.299 ± 0.098 | 0.222 ± 0.075 |
| Local MI (d=1) | **0.421** | **0.000** |
| Dilated MI (d=2) | 0.276 | **0.840** |
| Expressibility (KL) | 1.06 | 1.14 |
| Effective Dimension | 3.28 | 3.10 |

### Implications for the Full Study

These preliminary results suggest:

1. **Hypothesis Validated**: Dilated entanglement creates qualitatively different correlation patterns
2. **Expressibility Similar**: Both architectures have comparable expressibility (~1.0-1.1 KL divergence)
3. **Trade-off Identified**: QCNN has higher total entanglement; Dilated has more structured long-range correlations

The full 60-configuration study (6-12 qubits, 1-3 layers, 5 seeds) will determine:
- How these patterns scale with qubit count
- Whether dilated correlations improve classification performance
- Statistical significance of the differences (t-tests, Cohen's d)

---

## Datasets

### CIFAR-10
- **Location**: `/pscratch/sd/j/junghoon/data/cifar-10-batches-py/`
- **Size**: 60,000 images (50k train, 10k test)
- **Classes**: 10
- **Resolution**: 32×32×3

### COCO
- **Location**: `/pscratch/sd/j/junghoon/data/coco/`
- **Size**: 118,287 training images
- **Classes**: 80
- **Resolution**: 224×224×3 (resized)

---

## SLURM Batch Scripts

| Script | Purpose | Time | Array |
|--------|---------|------|-------|
| `QCNN_Comparison_CIFAR10.sh` | Train QCNN & QuantumDilatedCNN on CIFAR-10 | 24h | No |
| `QCNN_Comparison_COCO.sh` | Train QCNN & QuantumDilatedCNN on COCO | 24h | No |
| `run_comprehensive_metrics.sh` | Compute circuit metrics (60 configs) | 24h | 0-59 |

### SLURM Configuration
```bash
Account:     m4727_g
Constraint:  gpu&hbm80g (80GB HBM GPU)
QOS:         shared
CPUs:        32 per task
GPUs:        1 per task
```

---

## Step-by-Step Instructions

### Prerequisites

1. **Activate the conda environment**:
   ```bash
   module load python
   conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg
   ```

2. **Verify datasets exist**:
   ```bash
   ls /pscratch/sd/j/junghoon/data/cifar-10-batches-py/
   ls /pscratch/sd/j/junghoon/data/coco/train2017/ | wc -l  # Should show 118287
   ```

### Option A: Run All Experiments (Recommended)

```bash
cd /pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis

# Submit all jobs at once
bash run_all.sh
```

This will submit:
- 1 job for CIFAR-10 comparison
- 1 job for COCO comparison
- 60 array jobs for comprehensive metrics

### Option B: Run Individual Experiments

#### Step 1: CIFAR-10 Classification Comparison

```bash
cd /pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis
sbatch scripts/QCNN_Comparison_CIFAR10.sh
```

**Configuration**:
- Qubits: 8, Layers: 2
- Epochs: 50 (with early stopping, patience=10)
- Seeds: 2024, 2025, 2026
- Batch size: 32

**Outputs**:
- `results/cifar10/all_results_cifar10_cifar10_comparison.csv`
- `results/cifar10/summary_cifar10_cifar10_comparison.csv`
- `results/cifar10/QCNN_cifar10_seed*_history.csv`
- `results/cifar10/QuantumDilatedCNN_cifar10_seed*_history.csv`

#### Step 2: COCO Classification Comparison

```bash
sbatch scripts/QCNN_Comparison_COCO.sh
```

**Configuration**:
- Qubits: 8, Layers: 2
- Epochs: 50 (with early stopping, patience=10)
- Seeds: 2024, 2025, 2026
- Batch size: 16

**Outputs**:
- `results/coco/all_results_coco_coco_comparison.csv`
- `results/coco/summary_coco_coco_comparison.csv`

#### Step 3: Comprehensive Circuit Metrics

```bash
sbatch scripts/run_comprehensive_metrics.sh
```

**Configuration**:
- Qubits: 6, 8, 10, 12
- Layers: 1, 2, 3
- Seeds: 2024, 2025, 2026, 2027, 2028
- Total: 4 × 3 × 5 = 60 configurations

**Outputs**:
- `results/comprehensive_metrics/metrics_*q_*l/QCNN_comprehensive_*q_*l_seed*.npy`
- `results/comprehensive_metrics/metrics_*q_*l/QuantumDilatedCNN_comprehensive_*q_*l_seed*.npy`
- `results/comprehensive_metrics/metrics_*q_*l/summary_*q_*l_seed*.csv`

#### Step 4: Aggregate Metrics Results

After all metric jobs complete:

```bash
cd /pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/scripts
python aggregate_comprehensive_results.py \
    --base-dir=/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/results/comprehensive_metrics \
    --output-dir=/pscratch/sd/j/junghoon/QuantumDilatedCNN_Analysis/results/comprehensive_metrics
```

**Outputs**:
- `comprehensive_all_results.csv` - Raw data from all runs
- `comprehensive_summary_all_configs.csv` - Statistical summaries with p-values

---

## Monitoring Jobs

```bash
# View all your jobs
squeue -u $USER

# View specific job details
squeue -j <JOB_ID>

# View job logs (while running)
tail -f results/cifar10/QCNN_Comparison_CIFAR10_<JOB_ID>.out

# View completed job output
cat results/cifar10/QCNN_Comparison_CIFAR10_<JOB_ID>.out
```

---

## Resuming Interrupted Training

If a job is interrupted (timeout, node failure, etc.), simply resubmit:

```bash
sbatch scripts/QCNN_Comparison_CIFAR10.sh
```

The `--resume` flag is enabled by default, so training will automatically continue from the last checkpoint.

---

## Expected Results

### Classification Performance

| Model | Dataset | Expected Accuracy |
|-------|---------|-------------------|
| QCNN | CIFAR-10 | ~40-50% |
| QuantumDilatedCNN | CIFAR-10 | ~40-50% |
| QCNN | COCO | ~5-15% |
| QuantumDilatedCNN | COCO | ~5-15% |

*Note: Quantum models with limited qubits cannot match classical deep learning performance. The goal is to compare relative performance between quantum architectures.*

### Entanglement Metrics

We expect QuantumDilatedCNN to show:
- **Higher global MI (d>1)**: Due to dilated connections
- **Lower local MI (d=1)**: Reduced nearest-neighbor interactions
- **Similar Meyer-Wallach**: Overall entanglement should be comparable
- **Better expressibility**: Lower KL/JS divergence from Haar

---

## Key Parameters

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-qubits` | 8 | Number of qubits |
| `--n-layers` | 2 | Number of quantum layers |
| `--n-epochs` | 50 | Maximum epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--wd` | 1e-4 | Weight decay |
| `--seeds` | 2024 2025 2026 | Random seeds |
| `--patience` | 10 | Early stopping patience |
| `--resume` | True | Resume from checkpoint |

### Metrics Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-samples` | 100 | Random parameter samples |
| `--compute-expressibility` | True | Compute expressibility metrics |
| `--compute-eff-dim` | True | Compute effective dimension |


---

## Contact

- **Author**: Junghoon Park
- **Email**: utopie9090@snu.ac.kr
- **Institution**: Seoul National University

---

## Troubleshooting

### Import Error: scipy.constants
This is fixed in the current version. If you see this error, ensure you're using the latest `QCNN_Comparison.py`.

### CUDA Out of Memory
Reduce batch size:
```bash
python QCNN_Comparison.py --batch-size=16
```

### Job Timeout
Jobs are configured for 24 hours with early stopping (patience=10). If training is slow:
1. Reduce epochs: `--n-epochs=30`
2. Reduce seeds: `--seeds 2024 2025`
3. The checkpoint/resume system will continue from where it stopped

### Missing Checkpoint
Checkpoints are saved in `results/<dataset>/checkpoints/`. If missing, training starts from epoch 0.
