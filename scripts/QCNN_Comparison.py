#!/usr/bin/env python3
"""
QCNN vs Quantum Dilated CNN Comparison
========================================
Compares performance of:
1. QCNN (Cong et al. 2019) - nearest-neighbor entanglement
2. Quantum Dilated CNN - non-adjacent entanglement for global connectivity

Supports CIFAR-10 and COCO datasets with configurable parameters.
"""

import os, random, copy, time
import argparse
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
import scipy.constants  # Must import before pennylane (lazy loading fix)
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
import pennylane as qml

# Import data loaders (using local Load_Image_Datasets.py with proper COCO support)
from Load_Image_Datasets import load_data

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging disabled.")


def set_all_seeds(seed: int = 42) -> None:
    """Seed every RNG we rely on (Python, NumPy, Torch, PennyLane, CUDNN)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    qml.numpy.random.seed(seed)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training when validation loss doesn't improve for 'patience' epochs.
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.mode == 'min':
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop


def save_checkpoint(model, optimizer, scheduler, epoch: int, best_val_acc: float,
                   history: list, checkpoint_path: Path, model_name: str,
                   dataset: str, seed: int) -> None:
    """Save training checkpoint for resume capability."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_acc': best_val_acc,
        'history': history,
        'model_name': model_name,
        'dataset': dataset,
        'seed': seed,
    }
    checkpoint_file = checkpoint_path / f"{model_name}_{dataset}_seed{seed}_checkpoint.pt"
    torch.save(checkpoint, checkpoint_file)
    print(f"  Checkpoint saved: {checkpoint_file.name}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: Path,
                   model_name: str, dataset: str, seed: int, device: torch.device) -> Tuple[int, float, list]:
    """Load training checkpoint for resume capability."""
    checkpoint_file = checkpoint_path / f"{model_name}_{dataset}_seed{seed}_checkpoint.pt"

    if not checkpoint_file.exists():
        print(f"  No checkpoint found at {checkpoint_file}")
        return 0, 0.0, []

    print(f"  Loading checkpoint: {checkpoint_file.name}")
    checkpoint = torch.load(checkpoint_file, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc']
    history = checkpoint['history']

    print(f"  Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")
    return start_epoch, best_val_acc, history


class ClassicalFeatureExtractor(nn.Module):
    """
    A classical CNN backbone to extract features from images.
    Replaces the initial linear layer for dimension reduction.
    """
    def __init__(self, input_channels, input_height, input_width, output_features):
        super(ClassicalFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Halves dimensions
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Halves dimensions again
        )

        # Calculate the size of the flattened features after conv layers
        dummy_input = torch.zeros(1, input_channels, input_height, input_width)
        with torch.no_grad():
            flattened_size = self.conv_layers(dummy_input).view(1, -1).size(1)

        self.fc_out = nn.Linear(flattened_size, output_features)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        x = torch.tanh(self.fc_out(x)) # Apply tanh as in original fc layer
        return x


class QCNN(nn.Module):
    """
    Original QCNN from Cong et al. (2019) with nearest-neighbor entanglement.
    https://arxiv.org/abs/1810.03787
    """
    def __init__(self, n_qubits=8, n_layers=2, input_channels=3, input_height=32, input_width=32, num_classes=10):
        super(QCNN, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_classes = num_classes

        # Classical feature extraction backbone
        self.feature_extractor = ClassicalFeatureExtractor(input_channels, input_height, input_width, n_qubits)

        # Quantum parameters
        self.conv_params = nn.Parameter(torch.randn(n_layers, n_qubits, 15))
        self.pool_params = nn.Parameter(torch.randn(n_layers, n_qubits // 2, 3))
        self.last_params = nn.Parameter(torch.randn(15))

        # Output layer for multi-class
        self.fc_out = nn.Linear(1, num_classes)

        # Quantum device (PennyLane default.qubit simulator)
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(self, conv_weights, pool_weights, last_weights, features):
        wires = list(range(self.n_qubits))

        # Variational Embedding
        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        for layer in range(self.n_layers):
            # Convolutional Layer
            self._apply_convolution(conv_weights[layer], wires)
            # Pooling Layer
            self._apply_pooling(pool_weights[layer], wires)
            wires = wires[::2]

        # Final unitary
        qml.ArbitraryUnitary(last_weights, wires)

        return qml.expval(qml.PauliZ(0))

    def forward(self, x):
        batch_size = x.shape[0]
        # x = x.view(batch_size, -1)  # Flatten - handled by feature extractor

        # Classical feature extraction
        reduced_x = self.feature_extractor(x)

        # Quantum circuit execution - now batched
        qnode = qml.qnode(self.dev, interface="torch")(self.circuit)
        quantum_out = qnode(self.conv_params, self.pool_params, self.last_params, reduced_x)
        quantum_out = quantum_out.unsqueeze(1).to(torch.float32)

        # Output layer
        logits = self.fc_out(quantum_out)

        return logits

    def _apply_convolution(self, weights, wires):
        """Nearest-neighbor convolutional layer (original QCNN)."""
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

    def _apply_pooling(self, pool_weights, wires):
        """Pooling using mid-circuit measurement."""
        n_wires = len(wires)
        assert n_wires >= 2, "Need at least two wires for pooling."
        for indx, w in enumerate(wires):
            if indx % 2 == 1 and indx < n_wires:
                measurement = qml.measure(w)
                qml.cond(measurement, qml.U3)(*pool_weights[indx // 2], wires=wires[indx - 1])


class QuantumDilatedCNN(nn.Module):
    """
    Quantum Dilated CNN with non-adjacent entanglement for global connectivity.
    Reduces local entanglement, increases global entanglement.
    """
    def __init__(self, n_qubits=8, n_layers=2, input_channels=3, input_height=32, input_width=32, num_classes=10):
        super(QuantumDilatedCNN, self).__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_classes = num_classes

        # Classical feature extraction backbone
        self.feature_extractor = ClassicalFeatureExtractor(input_channels, input_height, input_width, n_qubits)

        # Quantum parameters (same structure as QCNN for fair comparison)
        self.conv_params = nn.Parameter(torch.randn(n_layers, n_qubits, 15))
        self.pool_params = nn.Parameter(torch.randn(n_layers, n_qubits // 2, 3))
        self.last_params = nn.Parameter(torch.randn(15))

        # Output layer
        self.fc_out = nn.Linear(1, num_classes)

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def circuit(self, conv_weights, pool_weights, last_weights, features):
        wires = list(range(self.n_qubits))

        # Variational Embedding
        qml.AngleEmbedding(features, wires=wires, rotation='Y')

        for layer in range(self.n_layers):
            # Dilated Convolutional Layer
            self._apply_dilated_convolution(conv_weights[layer], wires)
            # Pooling Layer (same as QCNN)
            self._apply_pooling(pool_weights[layer], wires)
            wires = wires[::2]

        # Final unitary
        qml.ArbitraryUnitary(last_weights, wires)

        return qml.expval(qml.PauliZ(0))

    def forward(self, x):
        batch_size = x.shape[0]
        # x = x.view(batch_size, -1)  # Flatten - handled by feature extractor

        # Classical feature extraction
        reduced_x = self.feature_extractor(x)

        # Quantum circuit execution - now batched
        qnode = qml.qnode(self.dev, interface="torch")(self.circuit)
        quantum_out = qnode(self.conv_params, self.pool_params, self.last_params, reduced_x)
        quantum_out = quantum_out.unsqueeze(1).to(torch.float32)

        # Output layer
        logits = self.fc_out(quantum_out)

        return logits

    def _apply_dilated_convolution(self, weights, wires):
        """
        Non-adjacent entanglement pattern for global connectivity.
        Uses dilation to skip neighbors and connect distant qubits.
        """
        n_wires = len(wires)

        # Define dilated entanglement pairs based on number of wires
        if n_wires == 8:
            entanglement_pairs = [(0, 2), (1, 3), (4, 6), (5, 7)]
        elif n_wires == 4:
            entanglement_pairs = [(0, 2), (1, 3)]
        elif n_wires == 2:
            entanglement_pairs = [(0, 1)]  # Fall back to nearest neighbor
        else:
            # General case: stride of 2
            entanglement_pairs = [(i, i+2) for i in range(n_wires-2)]

        processed_qubits = set()

        # Apply entangling blocks to dilated pairs
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

        # Apply single-qubit gates to remaining qubits
        for w in wires:
            if w not in processed_qubits:
                for i in range(5):  # 5 U3 gates = 15 parameters
                    qml.U3(*weights[w, i*3:(i+1)*3], wires=w)

    def _apply_pooling(self, pool_weights, wires):
        """Pooling using mid-circuit measurement (same as QCNN)."""
        n_wires = len(wires)
        assert n_wires >= 2, "Need at least two wires for pooling."
        for indx, w in enumerate(wires):
            if indx % 2 == 1 and indx < n_wires:
                measurement = qml.measure(w)
                qml.cond(measurement, qml.U3)(*pool_weights[indx // 2], wires=wires[indx - 1])


def reshape_data_for_conv(data):
    """
    Reshape flattened image data to (B, C, H, W) format for Conv2d.
    LoadData_MultiChip flattens images, but QCNN needs image format.
    """
    batch_size = data.size(0)
    total_pixels = data.size(1)

    # CIFAR-10: 3*32*32 = 3072
    if total_pixels == 3072:
        return data.view(batch_size, 3, 32, 32)
    # COCO: 3*224*224 = 150528
    elif total_pixels == 150528:
        return data.view(batch_size, 3, 224, 224)
    # Already in correct format (B, C, H, W)
    elif data.dim() == 4:
        return data
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}. Expected flattened CIFAR-10 (3072) or COCO (150528) or 4D tensor.")


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_idx, (data, target) in enumerate(tqdm(loader, desc="Training")):
        data, target = data.to(device), target.to(device)

        # Reshape flattened data to image format for Conv2d
        data = reshape_data_for_conv(data)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in tqdm(loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)

            # Reshape flattened data to image format for Conv2d
            data = reshape_data_for_conv(data)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    # Additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    return avg_loss, accuracy, precision, recall, f1


def train_single_seed(args, seed: int, device: torch.device) -> Dict:
    """Train models for a single seed and return results."""
    set_all_seeds(seed)

    # Create output directory for this seed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Dataset configuration
    if args.dataset.lower() == 'cifar10':
        input_channels, input_height, input_width, num_classes = 3, 32, 32, 10
    elif args.dataset.lower() == 'coco':
        input_channels, input_height, input_width, num_classes = 3, 224, 224, 80
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Load data
    train_loader, val_loader, test_loader = load_data(
        dataset_name=args.dataset.lower(),
        batch_size=args.batch_size,
        num_workers=4
    )

    # Initialize models on the correct device
    models = {}
    if 'qcnn' in args.models:
        models['QCNN'] = QCNN(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            num_classes=num_classes
        ).to(device)

    if 'dilated' in args.models:
        models['QuantumDilatedCNN'] = QuantumDilatedCNN(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            num_classes=num_classes
        ).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    results = {}

    # Train each model
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name} (seed={seed})")
        print(f"{'='*60}")

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

        # Early stopping
        early_stopping = EarlyStopping(patience=args.patience, mode='max')

        # Resume from checkpoint if available
        start_epoch = 0
        best_val_acc = 0.0
        history = []

        if args.resume:
            start_epoch, best_val_acc, history = load_checkpoint(
                model, optimizer, scheduler, checkpoint_dir,
                model_name, args.dataset, seed, device
            )

        best_epoch = start_epoch

        # Initialize wandb
        if args.wandb and WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{model_name}_{args.dataset}_seed{seed}_{args.job_id}",
                config={
                    "model": model_name,
                    "dataset": args.dataset,
                    "n_qubits": args.n_qubits,
                    "n_layers": args.n_layers,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "wd": args.wd,
                    "epochs": args.n_epochs,
                    "seed": seed,
                    "patience": args.patience
                },
                resume="allow"
            )

        for epoch in range(start_epoch, args.n_epochs):
            print(f"\nEpoch {epoch+1}/{args.n_epochs}")

            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

            # Validate
            val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
                model, val_loader, criterion, device
            )

            # Update learning rate scheduler
            scheduler.step(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Log to wandb
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "lr": optimizer.param_groups[0]['lr']
                })

            # Save history
            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'lr': optimizer.param_groups[0]['lr']
            })

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'seed': seed
                }, output_dir / f"{model_name}_{args.dataset}_seed{seed}_best.pt")
                print(f"  New best model saved! Val Acc: {val_acc:.4f}")

            # Save checkpoint every epoch for resume capability
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc,
                          history, checkpoint_dir, model_name, args.dataset, seed)

            # Check early stopping
            if early_stopping(val_acc, epoch):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
                break

        # Test best model
        print(f"\nLoading best model from epoch {best_epoch}...")
        best_model_path = output_dir / f"{model_name}_{args.dataset}_seed{seed}_best.pt"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
            model, test_loader, criterion, device
        )

        print(f"\nTest Results for {model_name} (seed={seed}):")
        print(f"  Accuracy: {test_acc:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  F1 Score: {test_f1:.4f}")

        # Save results
        results[model_name] = {
            'seed': seed,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'history': history
        }

        # Save training history
        pd.DataFrame(history).to_csv(
            output_dir / f"{model_name}_{args.dataset}_seed{seed}_history.csv",
            index=False
        )

        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

    return results


def main(args):
    """Main function with multiple seeds support."""
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*60}")
    print("QCNN vs QuantumDilatedCNN Comparison")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Dataset: {args.dataset}")
    print(f"Models: {args.models}")
    print(f"Qubits: {args.n_qubits}, Layers: {args.n_layers}")
    print(f"Epochs: {args.n_epochs}, Batch Size: {args.batch_size}")
    print(f"Seeds: {args.seeds}")
    print(f"Early Stopping Patience: {args.patience}")
    print(f"Resume: {args.resume}")
    print(f"{'='*60}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Run training for each seed
    for seed in args.seeds:
        print(f"\n{'#'*60}")
        print(f"# SEED {seed}")
        print(f"{'#'*60}")

        seed_results = train_single_seed(args, seed, device)

        for model_name, result in seed_results.items():
            all_results.append({
                'model': model_name,
                'seed': seed,
                'best_epoch': result['best_epoch'],
                'best_val_acc': result['best_val_acc'],
                'test_acc': result['test_acc'],
                'test_precision': result['test_precision'],
                'test_recall': result['test_recall'],
                'test_f1': result['test_f1']
            })

    # Create summary across all seeds
    results_df = pd.DataFrame(all_results)

    # Compute statistics per model
    summary_stats = []
    for model_name in results_df['model'].unique():
        model_results = results_df[results_df['model'] == model_name]
        summary_stats.append({
            'Model': model_name,
            'Test Acc (mean)': model_results['test_acc'].mean(),
            'Test Acc (std)': model_results['test_acc'].std(),
            'Test F1 (mean)': model_results['test_f1'].mean(),
            'Test F1 (std)': model_results['test_f1'].std(),
            'Best Val Acc (mean)': model_results['best_val_acc'].mean(),
            'N Seeds': len(model_results)
        })

    summary_df = pd.DataFrame(summary_stats)

    # Save all results
    results_df.to_csv(output_dir / f"all_results_{args.dataset}_{args.job_id}.csv", index=False)
    summary_df.to_csv(output_dir / f"summary_{args.dataset}_{args.job_id}.csv", index=False)

    # Print summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY (All Seeds)")
    print(f"{'='*60}")
    print("\nPer-seed results:")
    print(results_df.to_string(index=False))
    print("\nAggregate statistics:")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QCNN vs Quantum Dilated CNN Comparison")

    # Model parameters
    parser.add_argument('--n-qubits', type=int, default=8, help='Number of qubits')
    parser.add_argument('--n-layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--models', nargs='+', default=['qcnn', 'dilated'],
                       choices=['qcnn', 'dilated'], help='Models to compare')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'coco'], help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')

    # Training parameters
    parser.add_argument('--n-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2024, 2025, 2026],
                       help='Random seeds for multiple runs')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--resume', action='store_true', default=False,
                       help='Resume training from checkpoint if available')

    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./qcnn_comparison_results',
                       help='Output directory')
    parser.add_argument('--job-id', type=str, default='comparison',
                       help='Job ID for naming')

    # Wandb parameters
    parser.add_argument('--wandb', action='store_true', default=False, help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='QCNN_Comparison',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default='QML_Research',
                       help='Wandb entity name')

    args = parser.parse_args()

    main(args)
