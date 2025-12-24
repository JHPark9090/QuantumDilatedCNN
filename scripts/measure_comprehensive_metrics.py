#!/usr/bin/env python3
"""
Comprehensive Quantum Circuit Analysis
========================================
Measures all key metrics for QCNN vs QuantumDilatedCNN comparison:

1. Meyer-Wallach Measure (Entangling Capability)
2. Concentratable Entanglement
3. Distance-Dependent Mutual Information (Local vs Global)
4. Bipartite Entanglement Entropy
5. Expressibility (Sim et al. 2019)
6. Effective Dimension (Abbas et al. 2021)

Supports:
- Multiple random seeds for statistical robustness
- Multiple qubit/layer configurations
- Parallel execution via SLURM
- Statistical analysis output

References:
- Sim et al. (2019): https://arxiv.org/abs/1905.10876
- Beckey et al. (2021): https://arxiv.org/abs/2104.06923
- Abbas et al. (2021): https://arxiv.org/abs/2011.01938
"""

import os, random, argparse
from pathlib import Path
from typing import Tuple, List, Dict, Callable
import numpy as np
import pandas as pd
import scipy.constants  # Must import before pennylane (lazy loading fix)
import torch
import pennylane as qml
from tqdm import tqdm
from itertools import combinations
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


def set_all_seeds(seed: int = 42) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    qml.numpy.random.seed(seed)


# ============================================================================
# Entanglement Measures (from existing scripts)
# ============================================================================

def von_neumann_entropy(rho: np.ndarray) -> float:
    """Compute von Neumann entropy: S = -Tr(ρ log ρ)"""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))


def meyer_wallach_measure(state_vector: np.ndarray, n_qubits: int) -> float:
    """
    Meyer-Wallach entanglement measure Q.
    Q = 2 * (1 - (1/n) * sum_of_purities)
    Range: [0, 1], higher = more entangled
    """
    psi = state_vector.reshape([2] * n_qubits)
    sum_purities = 0.0

    for qubit_idx in range(n_qubits):
        psi_reordered = np.moveaxis(psi, qubit_idx, 0)
        psi_matrix = psi_reordered.reshape(2, -1)
        rho = psi_matrix @ psi_matrix.conj().T
        purity = np.trace(rho @ rho).real
        sum_purities += purity

    Q = 2 * (1 - sum_purities / n_qubits)
    return Q


def concentratable_entanglement(state_vector: np.ndarray, n_qubits: int) -> float:
    """
    Concentratable Entanglement (Beckey et al. 2021).
    C(|psi>) = 1 - (1/2^n) * sum_{alpha} Tr(rho_alpha^2)
    Range: [0, 1], higher = more concentratable entanglement
    """
    psi = state_vector.reshape([2] * n_qubits)
    sum_purities = 0.0

    for subset_size in range(1, n_qubits + 1):
        for subset in combinations(range(n_qubits), subset_size):
            purity = compute_subset_purity(psi, n_qubits, subset)
            sum_purities += purity

    C = 1 - sum_purities / (2 ** n_qubits)
    return C


def compute_subset_purity(psi: np.ndarray, n_qubits: int, subset: Tuple[int]) -> float:
    """Compute purity Tr(rho^2) for a subset of qubits."""
    qubits_to_trace = [i for i in range(n_qubits) if i not in subset]
    axes_order = list(subset) + qubits_to_trace
    psi_reordered = np.moveaxis(psi, axes_order, range(n_qubits))

    subset_dim = 2 ** len(subset)
    complement_dim = 2 ** len(qubits_to_trace)
    psi_matrix = psi_reordered.reshape(subset_dim, complement_dim)

    rho = psi_matrix @ psi_matrix.conj().T
    purity = np.trace(rho @ rho).real
    return purity


def partial_trace(state_vector: np.ndarray, n_qubits: int, keep_qubits: List[int]) -> np.ndarray:
    """Compute partial trace, keeping only specified qubits."""
    psi = state_vector.reshape([2] * n_qubits)
    trace_qubits = [i for i in range(n_qubits) if i not in keep_qubits]
    axes_order = list(keep_qubits) + trace_qubits
    psi_reordered = np.moveaxis(psi, axes_order, range(n_qubits))

    keep_dim = 2 ** len(keep_qubits)
    trace_dim = 2 ** len(trace_qubits)
    psi_matrix = psi_reordered.reshape(keep_dim, trace_dim)

    rho = psi_matrix @ psi_matrix.conj().T
    return rho


def mutual_information(state_vector: np.ndarray, n_qubits: int,
                       qubit_i: int, qubit_j: int) -> float:
    """Mutual information I(i:j) = S(ρ_i) + S(ρ_j) - S(ρ_{ij})"""
    rho_i = partial_trace(state_vector, n_qubits, [qubit_i])
    rho_j = partial_trace(state_vector, n_qubits, [qubit_j])
    rho_ij = partial_trace(state_vector, n_qubits, [qubit_i, qubit_j])

    S_i = von_neumann_entropy(rho_i)
    S_j = von_neumann_entropy(rho_j)
    S_ij = von_neumann_entropy(rho_ij)

    return S_i + S_j - S_ij


def distance_dependent_entanglement(state_vector: np.ndarray, n_qubits: int) -> Dict[int, List[float]]:
    """Compute pairwise entanglement as function of qubit distance."""
    distances = {}

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            dist = j - i
            mi = mutual_information(state_vector, n_qubits, i, j)

            if dist not in distances:
                distances[dist] = []
            distances[dist].append(mi)

    return distances


# ============================================================================
# NEW: Expressibility (Sim et al. 2019)
# ============================================================================

def state_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Compute fidelity between two pure states: F = |<psi1|psi2>|^2"""
    return np.abs(np.vdot(state1, state2)) ** 2


def haar_fidelity_distribution(n_qubits: int, n_samples: int = 1000) -> np.ndarray:
    """
    Theoretical fidelity distribution for Haar-random states.

    For pure states, the probability density is:
    p(F) = (2^n - 1) * (1 - F)^(2^n - 2)

    where F is fidelity, n is number of qubits.
    """
    # Generate samples from theoretical distribution
    dim = 2 ** n_qubits
    # Beta distribution with parameters (1, dim - 1)
    fidelities = np.random.beta(1, dim - 1, n_samples)
    return fidelities


def expressibility(circuit_function, n_qubits: int, params_list: List[Dict],
                  n_pairs: int = 1000) -> Dict[str, float]:
    """
    Measure expressibility via KL divergence from Haar distribution.

    Lower expressibility value = more uniform coverage of Hilbert space

    Args:
        circuit_function: Function that builds the quantum circuit
        n_qubits: Number of qubits
        params_list: List of random parameter configurations
        n_pairs: Number of random pairs to sample for fidelity distribution

    Returns:
        dict with 'kl_divergence', 'js_divergence', 'wasserstein_distance'
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        circuit_function(params)
        return qml.state()

    # Sample fidelities from circuit
    circuit_fidelities = []

    for _ in tqdm(range(n_pairs), desc="Computing expressibility", leave=False):
        # Randomly sample two parameter sets
        params1 = random.choice(params_list)
        params2 = random.choice(params_list)

        state1 = circuit(params1)
        state2 = circuit(params2)

        fid = state_fidelity(state1, state2)
        circuit_fidelities.append(fid)

    circuit_fidelities = np.array(circuit_fidelities)

    # Get Haar distribution samples
    haar_fidelities = haar_fidelity_distribution(n_qubits, n_samples=n_pairs)

    # Compute histograms
    bins = np.linspace(0, 1, 50)
    circuit_hist, _ = np.histogram(circuit_fidelities, bins=bins, density=True)
    haar_hist, _ = np.histogram(haar_fidelities, bins=bins, density=True)

    # Normalize to probability distributions
    circuit_hist = circuit_hist / (circuit_hist.sum() + 1e-10)
    haar_hist = haar_hist / (haar_hist.sum() + 1e-10)

    # Add small epsilon to avoid log(0)
    circuit_hist = circuit_hist + 1e-10
    haar_hist = haar_hist + 1e-10

    # Compute divergences
    kl_div = np.sum(circuit_hist * np.log(circuit_hist / haar_hist))
    js_div = jensenshannon(circuit_hist, haar_hist) ** 2  # Square for proper JS divergence
    wasserstein = wasserstein_distance(circuit_fidelities, haar_fidelities)

    return {
        'kl_divergence': kl_div,
        'js_divergence': js_div,
        'wasserstein_distance': wasserstein,
        'mean_fidelity': np.mean(circuit_fidelities),
        'std_fidelity': np.std(circuit_fidelities),
        'haar_mean_fidelity': np.mean(haar_fidelities),
        'circuit_fidelities': circuit_fidelities,
        'haar_fidelities': haar_fidelities
    }


# ============================================================================
# NEW: Effective Dimension (Abbas et al. 2021)
# ============================================================================

def effective_dimension_fisher(circuit_function, n_qubits: int,
                               params_list: List[Dict], n_samples: int = 100) -> Dict[str, float]:
    """
    Effective dimension via Fisher Information Matrix.

    d_eff = Tr(F) / ||F||_F

    where F is the Fisher Information Matrix.
    Higher d_eff = more expressivity, higher model capacity.

    Args:
        circuit_function: Function that builds quantum circuit
        n_qubits: Number of qubits
        params_list: List of parameter configurations
        n_samples: Number of samples to average over

    Returns:
        dict with 'effective_dimension', 'trace_fisher', 'frobenius_norm'
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    # Get parameter structure from first params
    sample_params = params_list[0]

    # Count total number of parameters
    n_params = 0
    param_shapes = {}

    for key, val in sample_params.items():
        if isinstance(val, np.ndarray):
            param_shapes[key] = val.shape
            n_params += val.size

    print(f"  Total parameters in circuit: {n_params}")

    # For large parameter spaces, use sampling-based estimation
    if n_params > 100:
        print(f"  Using sampling-based Fisher estimation (large parameter space)")
        return effective_dimension_sampling(circuit_function, n_qubits, params_list, n_samples)

    # Full Fisher matrix calculation (only for small circuits)
    fisher_traces = []
    fisher_norms = []

    for params in tqdm(random.sample(params_list, min(n_samples, len(params_list))),
                       desc="Computing effective dimension", leave=False):

        # Flatten parameters
        param_vector = []
        for key in sorted(sample_params.keys()):
            if isinstance(params[key], np.ndarray):
                param_vector.extend(params[key].flatten())
        param_vector = np.array(param_vector)

        # Compute Fisher Information Matrix via parameter-shift rule
        fisher_matrix = compute_fisher_matrix_qml(circuit_function, params, n_qubits, n_params)

        # Eigenvalue decomposition for numerical stability
        eigenvalues = np.linalg.eigvalsh(fisher_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative

        trace_F = np.sum(eigenvalues)
        norm_F = np.sqrt(np.sum(eigenvalues ** 2))

        fisher_traces.append(trace_F)
        fisher_norms.append(norm_F)

    mean_trace = np.mean(fisher_traces)
    mean_norm = np.mean(fisher_norms)

    # Effective dimension
    d_eff = mean_trace / (mean_norm + 1e-10)

    return {
        'effective_dimension': d_eff,
        'trace_fisher': mean_trace,
        'frobenius_norm': mean_norm,
        'normalized_dimension': d_eff / n_params  # Normalized by total params
    }


def compute_fisher_matrix_qml(circuit_function, params: Dict, n_qubits: int,
                              n_params: int) -> np.ndarray:
    """
    Compute Fisher Information Matrix using parameter-shift rule.

    For a parametrized quantum state |ψ(θ)>, the quantum Fisher information is:
    F_ij = 4 * Re(<∂_i ψ|∂_j ψ> - <∂_i ψ|ψ><ψ|∂_j ψ>)

    Using parameter-shift: |∂_i ψ> ≈ (|ψ(θ + π/2 e_i)> - |ψ(θ - π/2 e_i)>) / 2
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params_input):
        circuit_function(params_input)
        return qml.state()

    # Get central state
    state_0 = circuit(params)

    # Flatten parameters for easier manipulation
    param_keys = sorted([k for k in params.keys() if isinstance(params[k], np.ndarray)])

    # Store gradients
    gradients = []

    shift = np.pi / 2

    for key in param_keys:
        param_array = params[key]
        param_flat = param_array.flatten()

        for idx in range(len(param_flat)):
            # Create shifted parameters
            params_plus = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in params.items()}
            params_minus = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in params.items()}

            # Apply shift
            params_plus[key].flat[idx] += shift
            params_minus[key].flat[idx] -= shift

            # Get shifted states
            state_plus = circuit(params_plus)
            state_minus = circuit(params_minus)

            # Compute gradient
            grad = (state_plus - state_minus) / 2
            gradients.append(grad)

    gradients = np.array(gradients)

    # Compute Fisher matrix: F_ij = 4 * Re(<∂_i ψ|∂_j ψ> - <∂_i ψ|ψ><ψ|∂_j ψ>)
    fisher = np.zeros((n_params, n_params))

    for i in range(n_params):
        for j in range(i, n_params):
            term1 = np.vdot(gradients[i], gradients[j])
            term2 = np.vdot(gradients[i], state_0) * np.vdot(state_0, gradients[j])
            fisher[i, j] = 4 * np.real(term1 - term2)
            fisher[j, i] = fisher[i, j]  # Symmetric

    return fisher


def effective_dimension_sampling(circuit_function, n_qubits: int,
                                 params_list: List[Dict], n_samples: int = 100) -> Dict[str, float]:
    """
    Sampling-based effective dimension estimation.
    Uses output variance across parameter space.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        circuit_function(params)
        return qml.expval(qml.PauliZ(0))

    outputs = []

    for params in tqdm(random.sample(params_list, min(n_samples, len(params_list))),
                       desc="Sampling outputs", leave=False):
        out = circuit(params)
        outputs.append(out)

    outputs = np.array(outputs)

    # Estimate effective dimension from output variance
    output_variance = np.var(outputs)

    # For reference: max variance for single qubit measurement is 1
    # Effective dimension scales with achievable variance
    d_eff_estimate = output_variance * 10  # Heuristic scaling

    return {
        'effective_dimension': d_eff_estimate,
        'output_variance': output_variance,
        'output_mean': np.mean(outputs),
        'output_std': np.std(outputs),
        'method': 'sampling'
    }


# ============================================================================
# Circuit Building Functions
# ============================================================================

def apply_qcnn_convolution(weights, wires):
    """Nearest-neighbor convolutional layer."""
    n_wires = len(wires)
    for p in [0, 1]:
        for indx, w in enumerate(wires):
            if indx % 2 == p and indx < n_wires - 1:
                qml.U3(*weights[indx, :3], wires=w)
                qml.U3(*weights[indx + 1, 3:6], wires=wires[indx + 1])
                qml.IsingZZ(weights[indx, 6], wires=[w, wires[indx + 1]])
                qml.IsingYY(weights[indx, 7], wires=[w, wires[indx + 1]])
                qml.IsingXX(weights[indx, 8], wires=[w, wires[indx + 1]])
                qml.U3(*weights[indx, 9:12], wires=w)
                qml.U3(*weights[indx + 1, 12:], wires=wires[indx + 1])


def apply_dilated_convolution(weights, wires):
    """Non-adjacent entanglement pattern."""
    n_wires = len(wires)

    if n_wires == 8:
        entanglement_pairs = [(0, 2), (1, 3), (4, 6), (5, 7)]
    elif n_wires == 6:
        entanglement_pairs = [(0, 2), (1, 3), (2, 4), (3, 5)]
    elif n_wires == 4:
        entanglement_pairs = [(0, 2), (1, 3)]
    elif n_wires == 2:
        entanglement_pairs = [(0, 1)]
    else:
        entanglement_pairs = [(i, i+2) for i in range(n_wires-2)]

    processed_qubits = set()

    for q1, q2 in entanglement_pairs:
        if q1 in wires and q2 in wires:
            qml.U3(*weights[q1, :3], wires=q1)
            qml.U3(*weights[q2, 3:6], wires=q2)
            qml.IsingZZ(weights[q1, 6], wires=[q1, q2])
            qml.IsingYY(weights[q1, 7], wires=[q1, q2])
            qml.IsingXX(weights[q1, 8], wires=[q1, q2])
            qml.U3(*weights[q1, 9:12], wires=q1)
            qml.U3(*weights[q2, 12:], wires=q2)
            processed_qubits.add(q1)
            processed_qubits.add(q2)

    for w in wires:
        if w not in processed_qubits:
            for i in range(5):
                qml.U3(*weights[w, i*3:(i+1)*3], wires=w)


def build_qcnn_circuit(n_qubits: int, n_layers: int):
    """Build QCNN circuit (nearest-neighbor)."""
    def circuit(params):
        conv_params = params['conv']
        features = params['features']
        wires = list(range(n_qubits))

        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        for layer in range(n_layers):
            apply_qcnn_convolution(conv_params[layer], wires)
            wires = wires[::2]

    return circuit


def build_dilated_qcnn_circuit(n_qubits: int, n_layers: int):
    """Build QuantumDilatedCNN circuit (non-adjacent)."""
    def circuit(params):
        conv_params = params['conv']
        features = params['features']
        wires = list(range(n_qubits))

        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        for layer in range(n_layers):
            apply_dilated_convolution(conv_params[layer], wires)
            wires = wires[::2]

    return circuit


def generate_random_params(n_qubits: int, n_layers: int, n_samples: int) -> List[Dict]:
    """Generate random parameters."""
    params_list = []
    for _ in range(n_samples):
        features = np.random.uniform(-np.pi, np.pi, n_qubits)
        conv_params = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 15))

        params_list.append({
            'features': features,
            'conv': conv_params
        })

    return params_list


# ============================================================================
# Main Analysis Function
# ============================================================================

def comprehensive_analysis(circuit_function, n_qubits: int, params_list: List[Dict],
                          compute_expressibility: bool = True,
                          compute_eff_dim: bool = True) -> Dict:
    """
    Comprehensive analysis of quantum circuit.

    Returns all metrics in a single dictionary.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        circuit_function(params)
        return qml.state()

    results = {}

    # 1. Meyer-Wallach
    print("  1/6: Computing Meyer-Wallach measure...")
    mw_values = []
    for params in tqdm(params_list, desc="Meyer-Wallach", leave=False):
        state = circuit(params)
        mw = meyer_wallach_measure(state, n_qubits)
        mw_values.append(mw)

    results['meyer_wallach'] = {
        'mean': np.mean(mw_values),
        'std': np.std(mw_values),
        'values': np.array(mw_values)
    }

    # 2. Concentratable Entanglement (only for n_qubits <= 8)
    if n_qubits <= 8:
        print("  2/6: Computing concentratable entanglement...")
        ce_values = []
        for params in tqdm(params_list, desc="Concentratable", leave=False):
            state = circuit(params)
            ce = concentratable_entanglement(state, n_qubits)
            ce_values.append(ce)

        results['concentratable'] = {
            'mean': np.mean(ce_values),
            'std': np.std(ce_values),
            'values': np.array(ce_values)
        }
    else:
        print("  2/6: Skipping concentratable entanglement (n_qubits > 8)")
        results['concentratable'] = None

    # 3. Distance-dependent entanglement
    print("  3/6: Computing distance-dependent entanglement...")
    distance_ent_all = {d: [] for d in range(1, n_qubits)}

    for params in tqdm(params_list, desc="Distance entanglement", leave=False):
        state = circuit(params)
        dist_ent = distance_dependent_entanglement(state, n_qubits)
        for dist, values in dist_ent.items():
            distance_ent_all[dist].extend(values)

    results['distance_entanglement'] = {
        dist: {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
        for dist, values in distance_ent_all.items()
    }

    # 4. Expressibility
    if compute_expressibility:
        print("  4/6: Computing expressibility...")
        expr_results = expressibility(circuit_function, n_qubits, params_list, n_pairs=500)
        results['expressibility'] = expr_results
    else:
        print("  4/6: Skipping expressibility (disabled)")
        results['expressibility'] = None

    # 5. Effective Dimension
    if compute_eff_dim:
        print("  5/6: Computing effective dimension...")
        eff_dim_results = effective_dimension_fisher(circuit_function, n_qubits,
                                                     params_list, n_samples=50)
        results['effective_dimension'] = eff_dim_results
    else:
        print("  5/6: Skipping effective dimension (disabled)")
        results['effective_dimension'] = None

    print("  6/6: Analysis complete!")

    return results


# ============================================================================
# Main Execution
# ============================================================================

def main(args):
    set_all_seeds(args.seed)

    print("="*70)
    print("COMPREHENSIVE QUANTUM CIRCUIT ANALYSIS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Number of qubits: {args.n_qubits}")
    print(f"  Number of layers: {args.n_layers}")
    print(f"  Number of samples: {args.n_samples}")
    print(f"  Random seed: {args.seed}")
    print(f"  Compute expressibility: {args.compute_expressibility}")
    print(f"  Compute effective dimension: {args.compute_eff_dim}")
    print()

    # Generate parameters
    print(f"Generating {args.n_samples} random parameter samples...")
    params_samples = generate_random_params(args.n_qubits, args.n_layers, args.n_samples)

    results = {}

    # Analyze QCNN
    print("\n" + "="*70)
    print("Analyzing QCNN (Nearest-Neighbor Entanglement)")
    print("="*70)
    qcnn_circuit = build_qcnn_circuit(args.n_qubits, args.n_layers)
    results['QCNN'] = comprehensive_analysis(
        qcnn_circuit, args.n_qubits, params_samples,
        compute_expressibility=args.compute_expressibility,
        compute_eff_dim=args.compute_eff_dim
    )

    # Analyze QuantumDilatedCNN
    print("\n" + "="*70)
    print("Analyzing QuantumDilatedCNN (Dilated Entanglement)")
    print("="*70)
    dilated_circuit = build_dilated_qcnn_circuit(args.n_qubits, args.n_layers)
    results['QuantumDilatedCNN'] = comprehensive_analysis(
        dilated_circuit, args.n_qubits, params_samples,
        compute_expressibility=args.compute_expressibility,
        compute_eff_dim=args.compute_eff_dim
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save numpy files
    for model_name, model_results in results.items():
        filename = f"{model_name}_comprehensive_{args.n_qubits}q_{args.n_layers}l_seed{args.seed}.npy"
        np.save(output_dir / filename, model_results)

    # Create summary CSV
    summary_data = []

    for model_name in ['QCNN', 'QuantumDilatedCNN']:
        row = {
            'Model': model_name,
            'n_qubits': args.n_qubits,
            'n_layers': args.n_layers,
            'seed': args.seed,
            'Meyer-Wallach (mean)': results[model_name]['meyer_wallach']['mean'],
            'Meyer-Wallach (std)': results[model_name]['meyer_wallach']['std'],
        }

        if results[model_name]['concentratable'] is not None:
            row['Concentratable (mean)'] = results[model_name]['concentratable']['mean']
            row['Concentratable (std)'] = results[model_name]['concentratable']['std']

        # Local vs global
        local_mi = results[model_name]['distance_entanglement'][1]['mean']
        global_mi = np.mean([results[model_name]['distance_entanglement'][d]['mean']
                            for d in range(2, args.n_qubits)])
        row['Local MI (d=1)'] = local_mi
        row['Global MI (d>1)'] = global_mi

        if results[model_name]['expressibility'] is not None:
            row['Expressibility (KL)'] = results[model_name]['expressibility']['kl_divergence']
            row['Expressibility (JS)'] = results[model_name]['expressibility']['js_divergence']

        if results[model_name]['effective_dimension'] is not None:
            row['Effective Dimension'] = results[model_name]['effective_dimension']['effective_dimension']

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / f"summary_{args.n_qubits}q_{args.n_layers}l_seed{args.seed}.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print("\n", summary_df.to_string(index=False))
    print(f"\n\nResults saved to: {output_dir}")
    print(f"  - NumPy files: *_comprehensive_{args.n_qubits}q_{args.n_layers}l_seed{args.seed}.npy")
    print(f"  - Summary CSV: {summary_csv.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive quantum circuit analysis with all metrics"
    )

    parser.add_argument('--n-qubits', type=int, default=6,
                       help='Number of qubits (default: 6)')
    parser.add_argument('--n-layers', type=int, default=2,
                       help='Number of layers (default: 2)')
    parser.add_argument('--n-samples', type=int, default=100,
                       help='Number of random parameter samples (default: 100)')
    parser.add_argument('--seed', type=int, default=2025,
                       help='Random seed (default: 2025)')
    parser.add_argument('--output-dir', type=str, default='./comprehensive_results',
                       help='Output directory for results')

    # Toggle metrics
    parser.add_argument('--compute-expressibility', action='store_true', default=True,
                       help='Compute expressibility (default: True)')
    parser.add_argument('--no-expressibility', action='store_false', dest='compute_expressibility',
                       help='Skip expressibility computation')
    parser.add_argument('--compute-eff-dim', action='store_true', default=True,
                       help='Compute effective dimension (default: True)')
    parser.add_argument('--no-eff-dim', action='store_false', dest='compute_eff_dim',
                       help='Skip effective dimension computation')

    args = parser.parse_args()

    main(args)
