"""
Dataset management and loading for Decloud Validator
"""
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from config import DATASETS_DIR, DATASETS, config

# Try to import various dataset libraries
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

try:
    import torchvision
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    from sklearn import datasets as sklearn_datasets
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class DatasetInfo:
    """Information about a dataset"""
    name: str
    task_type: str  # classification, regression, etc.
    num_classes: Optional[int]
    input_shape: Tuple
    split: str  # train, test, validation
    size: int
    

# Dataset configurations
DATASET_CONFIGS = {
    # Image Classification (torchvision)
    "Cifar10": {"source": "torchvision", "name": "CIFAR10", "num_classes": 10, "task": "classification"},
    "Cifar100": {"source": "torchvision", "name": "CIFAR100", "num_classes": 100, "task": "classification"},
    "Mnist": {"source": "torchvision", "name": "MNIST", "num_classes": 10, "task": "classification"},
    "FashionMnist": {"source": "torchvision", "name": "FashionMNIST", "num_classes": 10, "task": "classification"},
    "Emnist": {"source": "torchvision", "name": "EMNIST", "num_classes": 47, "task": "classification", "split": "balanced"},
    "Kmnist": {"source": "torchvision", "name": "KMNIST", "num_classes": 10, "task": "classification"},
    "Svhn": {"source": "torchvision", "name": "SVHN", "num_classes": 10, "task": "classification"},
    "Caltech101": {"source": "torchvision", "name": "Caltech101", "num_classes": 101, "task": "classification"},
    "Caltech256": {"source": "torchvision", "name": "Caltech256", "num_classes": 257, "task": "classification"},
    "Food101": {"source": "torchvision", "name": "Food101", "num_classes": 101, "task": "classification"},
    "Flowers102": {"source": "torchvision", "name": "Flowers102", "num_classes": 102, "task": "classification"},
    "StanfordCars": {"source": "torchvision", "name": "StanfordCars", "num_classes": 196, "task": "classification"},
    "OxfordPets": {"source": "torchvision", "name": "OxfordIIITPet", "num_classes": 37, "task": "classification"},
    "Eurosat": {"source": "torchvision", "name": "EuroSAT", "num_classes": 10, "task": "classification"},
    
    # Text Classification (HuggingFace)
    "Imdb": {"source": "huggingface", "name": "imdb", "num_classes": 2, "task": "classification"},
    "Sst2": {"source": "huggingface", "name": "sst2", "num_classes": 2, "task": "classification"},
    "AgNews": {"source": "huggingface", "name": "ag_news", "num_classes": 4, "task": "classification"},
    "YelpReviews": {"source": "huggingface", "name": "yelp_review_full", "num_classes": 5, "task": "classification"},
    "AmazonPolarity": {"source": "huggingface", "name": "amazon_polarity", "num_classes": 2, "task": "classification"},
    "Dbpedia": {"source": "huggingface", "name": "dbpedia_14", "num_classes": 14, "task": "classification"},
    
    # Sklearn datasets
    "Iris": {"source": "sklearn", "name": "iris", "num_classes": 3, "task": "classification"},
    "Wine": {"source": "sklearn", "name": "wine", "num_classes": 3, "task": "classification"},
    "BreastCancer": {"source": "sklearn", "name": "breast_cancer", "num_classes": 2, "task": "classification"},
    "Diabetes": {"source": "sklearn", "name": "diabetes", "num_classes": None, "task": "regression"},
    
    # More HuggingFace
    "RottenTomatoes": {"source": "huggingface", "name": "rotten_tomatoes", "num_classes": 2, "task": "classification"},
    "Banking77": {"source": "huggingface", "name": "banking77", "num_classes": 77, "task": "classification"},
    "Snli": {"source": "huggingface", "name": "snli", "num_classes": 3, "task": "nli"},
    "Mnli": {"source": "huggingface", "name": "multi_nli", "num_classes": 3, "task": "nli"},
    "Squad": {"source": "huggingface", "name": "squad", "num_classes": None, "task": "qa"},
    "SquadV2": {"source": "huggingface", "name": "squad_v2", "num_classes": None, "task": "qa"},
}


class EmbeddingsDataset(TorchDataset):
    """Dataset wrapper for embeddings + labels"""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class DatasetManager:
    """Manages dataset installation and loading"""
    
    def __init__(self):
        self.cache_dir = DATASETS_DIR
    
    def list_available(self) -> List[str]:
        """List all available datasets"""
        return list(DATASETS.keys())
    
    def list_installed(self) -> List[str]:
        """List installed datasets"""
        return config.installed_datasets.copy()
    
    def is_installed(self, name: str) -> bool:
        """Check if dataset is installed"""
        return name in config.installed_datasets
    
    def install(self, name: str) -> bool:
        """Install a dataset"""
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}")
        
        if name in config.installed_datasets:
            return True
        
        cfg = DATASET_CONFIGS.get(name)
        if not cfg:
            # For unsupported datasets, just mark as installed
            # (user needs to provide data manually)
            config.add_dataset(name)
            return True
        
        try:
            if cfg["source"] == "torchvision":
                self._install_torchvision(name, cfg)
            elif cfg["source"] == "huggingface":
                self._install_huggingface(name, cfg)
            elif cfg["source"] == "sklearn":
                self._install_sklearn(name, cfg)
            
            config.add_dataset(name)
            return True
        except Exception as e:
            print(f"Failed to install {name}: {e}")
            return False
    
    def _install_torchvision(self, name: str, cfg: Dict):
        """Install torchvision dataset"""
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision not installed")
        
        dataset_cls = getattr(torchvision.datasets, cfg["name"])
        dataset_path = self.cache_dir / name
        
        # Download
        kwargs = {"root": str(dataset_path), "download": True}
        if "split" in cfg:
            kwargs["split"] = cfg["split"]
        else:
            kwargs["train"] = False  # Get test set for validation
        
        try:
            dataset_cls(**kwargs)
        except TypeError:
            # Some datasets don't have train parameter
            kwargs.pop("train", None)
            dataset_cls(**kwargs)
    
    def _install_huggingface(self, name: str, cfg: Dict):
        """Install HuggingFace dataset"""
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library not installed")
        
        dataset_path = self.cache_dir / name
        load_dataset(cfg["name"], cache_dir=str(dataset_path))
    
    def _install_sklearn(self, name: str, cfg: Dict):
        """Install sklearn dataset (just verification)"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed")
        
        # Sklearn datasets are built-in, just verify
        loader = getattr(sklearn_datasets, f"load_{cfg['name']}")
        loader()
    
    def uninstall(self, name: str) -> bool:
        """Uninstall a dataset"""
        if name not in config.installed_datasets:
            return True
        
        import shutil
        dataset_path = self.cache_dir / name
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        
        config.remove_dataset(name)
        return True
    
    def load_test_data(self, name: str, limit: Optional[int] = None) -> Tuple[Any, Any]:
        """Load test data for validation"""
        if name not in config.installed_datasets:
            raise ValueError(f"Dataset {name} not installed")
        
        cfg = DATASET_CONFIGS.get(name)
        if not cfg:
            raise ValueError(f"Dataset {name} not supported for auto-loading")
        
        if cfg["source"] == "torchvision":
            return self._load_torchvision_test(name, cfg, limit)
        elif cfg["source"] == "huggingface":
            return self._load_huggingface_test(name, cfg, limit)
        elif cfg["source"] == "sklearn":
            return self._load_sklearn_test(name, cfg, limit)
        
        raise ValueError(f"Unknown source: {cfg['source']}")
    
    def _load_torchvision_test(self, name: str, cfg: Dict, limit: Optional[int]) -> Tuple[Any, Any]:
        """Load torchvision test data"""
        dataset_cls = getattr(torchvision.datasets, cfg["name"])
        dataset_path = self.cache_dir / name
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        kwargs = {"root": str(dataset_path), "download": False, "transform": transform}
        if "split" in cfg:
            kwargs["split"] = cfg["split"]
        else:
            kwargs["train"] = False
        
        try:
            dataset = dataset_cls(**kwargs)
        except TypeError:
            kwargs.pop("train", None)
            kwargs["split"] = "test"
            dataset = dataset_cls(**kwargs)
        
        # Convert to arrays
        if limit:
            indices = list(range(min(limit, len(dataset))))
        else:
            indices = list(range(len(dataset)))
        
        data = []
        labels = []
        for i in indices:
            x, y = dataset[i]
            data.append(x.numpy())
            labels.append(y)
        
        return np.array(data), np.array(labels)
    
    def _load_huggingface_test(self, name: str, cfg: Dict, limit: Optional[int]) -> Tuple[Any, Any]:
        """Load HuggingFace test data"""
        dataset_path = self.cache_dir / name
        ds = load_dataset(cfg["name"], cache_dir=str(dataset_path))
        
        # Get test split
        if "test" in ds:
            test_ds = ds["test"]
        elif "validation" in ds:
            test_ds = ds["validation"]
        else:
            test_ds = ds["train"]
        
        if limit:
            test_ds = test_ds.select(range(min(limit, len(test_ds))))
        
        # Extract text and labels
        text_key = "text" if "text" in test_ds.column_names else test_ds.column_names[0]
        label_key = "label" if "label" in test_ds.column_names else "labels"
        
        texts = test_ds[text_key]
        labels = test_ds[label_key] if label_key in test_ds.column_names else [0] * len(texts)
        
        return texts, np.array(labels)
    
    def _load_sklearn_test(self, name: str, cfg: Dict, limit: Optional[int]) -> Tuple[Any, Any]:
        """Load sklearn test data"""
        loader = getattr(sklearn_datasets, f"load_{cfg['name']}")
        data = loader()
        
        X, y = data.data, data.target
        
        # Use last 20% as test
        split = int(len(X) * 0.8)
        X_test, y_test = X[split:], y[split:]
        
        if limit:
            X_test = X_test[:limit]
            y_test = y_test[:limit]
        
        return X_test, y_test
    
    def get_num_classes(self, name: str) -> Optional[int]:
        """Get number of classes for a dataset"""
        cfg = DATASET_CONFIGS.get(name)
        if cfg:
            return cfg.get("num_classes")
        return None
    
    def get_task_type(self, name: str) -> str:
        """Get task type for a dataset"""
        cfg = DATASET_CONFIGS.get(name)
        if cfg:
            return cfg.get("task", "classification")
        return "classification"


# Global instance
dataset_manager = DatasetManager()
