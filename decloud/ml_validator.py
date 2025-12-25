"""
DECLOUD ML Validator
====================

Machine learning validation for federated learning.
Evaluates trainer submissions by applying gradients and measuring improvement.
"""

import os
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    """Simple CNN for image classification (CIFAR-10/100, MNIST, etc.)"""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Model registry
MODELS = {
    "simple_cnn": SimpleCNN,
    "simple_cnn_10": lambda: SimpleCNN(num_classes=10),
    "simple_cnn_100": lambda: SimpleCNN(num_classes=100),
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetLoader:
    """Loads test datasets for validation"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cache = {}
    
    def get_loader(self, dataset_name: str, batch_size: int = 64) -> DataLoader:
        """Get DataLoader for a dataset"""
        
        if dataset_name in self._cache:
            return self._cache[dataset_name]
        
        print(f"📦 Loading {dataset_name} test dataset...")
        
        # Normalize dataset name
        name = dataset_name.lower().replace("_", "")
        
        if name in ("cifar10",):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=transform
            )
        
        elif name in ("cifar100",):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = datasets.CIFAR100(
                root=self.data_dir,
                train=False,
                download=True,
                transform=transform
            )
        
        elif name in ("mnist",):
            transform = transforms.Compose([
                transforms.Grayscale(3),  # Convert to 3 channels
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=True,
                transform=transform
            )
        
        elif name in ("fashionmnist",):
            transform = transforms.Compose([
                transforms.Grayscale(3),
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = datasets.FashionMNIST(
                root=self.data_dir,
                train=False,
                download=True,
                transform=transform
            )
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self._cache[dataset_name] = loader
        
        print(f"   ✓ Loaded {len(dataset)} test samples")
        return loader


# ═══════════════════════════════════════════════════════════════════════════════
# IPFS DOWNLOADER
# ═══════════════════════════════════════════════════════════════════════════════

class IPFSDownloader:
    """Downloads files from IPFS"""
    
    GATEWAYS = [
        "https://ipfs.io/ipfs/",
        "https://cloudflare-ipfs.com/ipfs/",
        "https://gateway.pinata.cloud/ipfs/",
        "https://dweb.link/ipfs/",
    ]
    
    def __init__(self, data_dir: str = "./data", local_store: str = None):
        self.data_dir = data_dir
        self.local_store = local_store or os.path.join(data_dir, "ipfs_local")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download(self, cid: str, output_path: str, timeout: int = 60) -> bool:
        """
        Download file from IPFS.
        First checks local store, then tries IPFS gateways.
        """
        # Check local store first
        if self.local_store:
            local_path = os.path.join(self.local_store, cid)
            if os.path.exists(local_path):
                with open(local_path, "rb") as f:
                    content = f.read()
                with open(output_path, "wb") as f:
                    f.write(content)
                print(f"   ✓ Loaded from local: {cid[:20]}...")
                return True
        
        # Try IPFS gateways
        for gateway in self.GATEWAYS:
            try:
                url = f"{gateway}{cid}"
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    print(f"   ✓ Downloaded: {cid[:20]}...")
                    return True
            except Exception:
                continue
        
        print(f"   ✗ Failed to download: {cid[:20]}...")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# ML VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainerResult:
    """Result of trainer evaluation"""
    trainer: str
    accuracy: float
    improvement: float  # vs baseline
    contribution_bps: int


class MLValidator:
    """
    Validates trainer submissions using ML evaluation.
    
    Workflow:
    1. Load baseline model
    2. Evaluate baseline accuracy
    3. For each trainer:
       - Load their gradients
       - Apply to model copy
       - Measure new accuracy
       - Calculate improvement
    4. Distribute rewards proportionally
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_loader = TestDatasetLoader(data_dir)
        self.ipfs = IPFSDownloader(data_dir)
        
        print(f"🖥️  ML Validator initialized (device: {self.device})")
    
    def load_model(self, path: str, model_class: str = "simple_cnn") -> nn.Module:
        """Load a model from file"""
        if model_class in MODELS:
            model_fn = MODELS[model_class]
            model = model_fn() if callable(model_fn) else model_fn
        else:
            model = SimpleCNN()
        
        model = model.to(self.device)
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        return model
    
    def evaluate(self, model: nn.Module, dataset_name: str) -> float:
        """Evaluate model accuracy on test dataset"""
        model.eval()
        loader = self.dataset_loader.get_loader(dataset_name)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def load_gradients(self, path: str) -> Dict[str, torch.Tensor]:
        """Load gradient dict from file"""
        return torch.load(path, map_location=self.device, weights_only=True)
    
    def apply_gradients(
        self,
        model: nn.Module,
        gradients: Dict[str, torch.Tensor],
        learning_rate: float = 0.1
    ) -> nn.Module:
        """Apply gradients to model (gradient descent step)"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in gradients:
                    param.data -= learning_rate * gradients[name]
        return model
    
    def evaluate_submissions(
        self,
        model_cid: str,
        gradient_submissions: List[Tuple[str, str]],  # [(trainer_pubkey, gradient_cid), ...]
        dataset_name: str,
        learning_rate: float = 0.1,
    ) -> List[TrainerResult]:
        """
        Evaluate all trainer submissions.
        
        Args:
            model_cid: IPFS CID of baseline model
            gradient_submissions: List of (trainer_pubkey, gradient_cid)
            dataset_name: Dataset to evaluate on
            learning_rate: Learning rate for gradient application
            
        Returns:
            List of TrainerResult with accuracies and improvements
        """
        # Download and load baseline model
        model_path = os.path.join(self.data_dir, f"model_{model_cid[:16]}.pt")
        if not os.path.exists(model_path):
            if not self.ipfs.download(model_cid, model_path):
                raise RuntimeError(f"Failed to download model: {model_cid}")
        
        baseline_model = self.load_model(model_path)
        baseline_acc = self.evaluate(baseline_model, dataset_name)
        print(f"\n📊 Baseline accuracy: {baseline_acc:.2f}%")
        
        results = []
        
        for trainer_pubkey, gradient_cid in gradient_submissions:
            print(f"\n🔬 Evaluating {trainer_pubkey[:16]}...")
            
            # Download gradients
            grad_path = os.path.join(
                self.data_dir,
                f"grad_{gradient_cid[:16]}.pt"
            )
            
            try:
                if not os.path.exists(grad_path):
                    if not self.ipfs.download(gradient_cid, grad_path):
                        raise RuntimeError("Download failed")
                
                # Load fresh model copy
                model = self.load_model(model_path)
                
                # Load and apply gradients
                gradients = self.load_gradients(grad_path)
                model = self.apply_gradients(model, gradients, learning_rate)
                
                # Evaluate
                new_acc = self.evaluate(model, dataset_name)
                improvement = new_acc - baseline_acc
                
                print(f"   ✓ Accuracy: {new_acc:.2f}% ({improvement:+.2f}%)")
                
                results.append(TrainerResult(
                    trainer=trainer_pubkey,
                    accuracy=new_acc,
                    improvement=improvement,
                    contribution_bps=0,
                ))
                
            except Exception as e:
                print(f"   ❌ FAILED: {e}")
                print(f"   💩 Trainer gets 0 contribution for invalid submission")
                
                results.append(TrainerResult(
                    trainer=trainer_pubkey,
                    accuracy=0.0,
                    improvement=-100.0,  # Максимальный штраф
                    contribution_bps=0,
                ))
        
        return results
    
    def calculate_contributions(
        self,
        results: List[TrainerResult]
    ) -> List[TrainerResult]:
        """
        Calculate contribution percentages based on improvements.
        
        - Trainers with failed submissions (improvement=-100) get 0
        - Trainers with positive improvement get proportional share
        - If no positive improvements, non-failed split equally
        - Total must sum to 10000 bps (100%)
        """
        if not results:
            return []
        
        # Separate failed (couldn't download/apply gradients) from valid
        failed = [r for r in results if r.improvement <= -99]  # -100 = failed
        valid = [r for r in results if r.improvement > -99]
        
        # Mark failed as 0
        for r in failed:
            r.contribution_bps = 0
            print(f"   💩 {r.trainer[:16]}... = 0% (invalid submission)")
        
        if not valid:
            # Everyone failed - return empty (don't complete round)
            print("   ⚠️  All submissions were invalid! Round will NOT be completed.")
            return []  # Empty = don't complete
        
        # Calculate for valid trainers
        positive = [r for r in valid if r.improvement > 0]
        
        if not positive:
            # No improvements but valid submissions - split equally among valid
            bps_each = 10000 // len(valid)
            remainder = 10000 - (bps_each * len(valid))
            for i, r in enumerate(valid):
                r.contribution_bps = bps_each + (1 if i < remainder else 0)
            return results
        
        # Proportional to improvement (only positive get rewards)
        total_improvement = sum(r.improvement for r in positive)
        total_bps = 0
        
        for r in valid:
            if r.improvement > 0:
                bps = int((r.improvement / total_improvement) * 10000)
            else:
                bps = 0  # Didn't improve = no reward
            r.contribution_bps = bps
            total_bps += bps
        
        # Adjust for rounding
        if total_bps != 10000:
            diff = 10000 - total_bps
            for r in results:
                if r.contribution_bps > 0:
                    r.contribution_bps += diff
                    break
        
        return results
    
    def validate_round(
        self,
        model_cid: str,
        gradient_submissions: List[Tuple[str, str]],
        dataset_name: str,
    ) -> Dict[str, int]:
        """
        Full validation workflow.
        
        Returns:
            Dict mapping trainer pubkey to contribution_bps
        """
        print(f"\n{'═'*60}")
        print(f"  🔬 ML Validation")
        print(f"{'═'*60}")
        print(f"  Model:   {model_cid[:30]}...")
        print(f"  Dataset: {dataset_name}")
        print(f"  Trainers: {len(gradient_submissions)}")
        print(f"{'═'*60}")
        
        # Evaluate all submissions
        results = self.evaluate_submissions(
            model_cid,
            gradient_submissions,
            dataset_name
        )
        
        # Calculate contributions
        results = self.calculate_contributions(results)
        
        if not results:
            print(f"\n{'─'*60}")
            print("  ❌ No valid contributions - round will NOT be completed")
            print(f"{'─'*60}\n")
            return {}
        
        # Print summary
        print(f"\n{'─'*60}")
        print("  📊 Contribution Summary:")
        for r in sorted(results, key=lambda x: x.contribution_bps, reverse=True):
            pct = r.contribution_bps / 100
            bar = "█" * int(pct / 5)
            print(f"  {r.trainer[:20]}... {pct:5.1f}% {bar}")
        print(f"{'─'*60}\n")
        
        return {r.trainer: r.contribution_bps for r in results}