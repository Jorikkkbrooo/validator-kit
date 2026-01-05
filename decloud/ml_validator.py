"""
DECLOUD ML Validator - Full Dataset Support
============================================

Machine learning validation for federated learning.
Evaluates trainer submissions by applying gradients and measuring improvement.

Supports 99 datasets matching the smart contract enum.
"""

import os
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torchvision.models as models

try:
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
        AutoModelForSequenceClassification,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    """Simple CNN for small images (32x32)"""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
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
        return self.fc2(x)


class MediumCNN(nn.Module):
    """Medium CNN for larger images (64-128px)"""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512 * 4 * 4, 1024), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SimpleMLP(nn.Module):
    """MLP for tabular data"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(0.3)])
            prev = dim
        layers.append(nn.Linear(prev, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AudioCNN(nn.Module):
    """CNN for audio spectrograms"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


class TextCNN(nn.Module):
    """CNN for text classification"""
    
    def __init__(self, vocab_size: int = 30000, embed_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, 100, k) for k in [3, 4, 5]])
        self.fc = nn.Linear(300, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.cat([torch.relu(c(x)).max(2)[0] for c in self.convs], 1)
        return self.fc(self.dropout(x))


class TimeSeriesCNN(nn.Module):
    """CNN for time series"""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 5, seq_len: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(4),
        )
        self.fc = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.fc(self.conv(x).view(x.size(0), -1))


class GraphMLP(nn.Module):
    """MLP for graph node classification"""
    
    def __init__(self, input_dim: int, num_classes: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.model(x)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_model(architecture: str, num_classes: int, in_channels: int = 3, 
                 input_dim: int = None, pretrained: bool = False, **kwargs) -> nn.Module:
    """Create model by architecture name"""
    arch = architecture.lower()
    
    if arch == "simple_cnn":
        return SimpleCNN(num_classes, in_channels)
    elif arch == "medium_cnn":
        return MediumCNN(num_classes, in_channels)
    elif arch == "resnet18":
        m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        if in_channels != 3:
            m.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        return m
    elif arch == "resnet34":
        m = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif arch == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif arch == "efficientnet_b1":
        m = models.efficientnet_b1(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif arch == "vgg11":
        m = models.vgg11(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
        return m
    elif arch == "densenet121":
        m = models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    elif arch == "vit_b_16":
        m = models.vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    elif arch == "mlp":
        return SimpleMLP(input_dim or 4, num_classes)
    elif arch == "audio_cnn":
        return AudioCNN(num_classes)
    elif arch == "text_cnn":
        return TextCNN(vocab_size=kwargs.get("vocab_size", 30000), num_classes=num_classes)
    elif arch == "timeseries_cnn":
        return TimeSeriesCNN(kwargs.get("input_channels", 1), num_classes, kwargs.get("seq_len", 128))
    elif arch == "graph_mlp":
        return GraphMLP(input_dim or 100, num_classes)
    elif arch == "distilbert":
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed")
        return DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_classes)
    elif arch == "bert":
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed")
        return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CONFIGURATIONS - ALL 99 DATASETS FROM SMART CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_CONFIG = {
    # ═══════════════════════════════════════════════════════════════
    # IMAGE CLASSIFICATION (17)
    # ═══════════════════════════════════════════════════════════════
    "cifar10": {"num_classes": 10, "in_channels": 3, "default_arch": "simple_cnn", "task": "image"},
    "cifar100": {"num_classes": 100, "in_channels": 3, "default_arch": "resnet18", "task": "image"},
    "mnist": {"num_classes": 10, "in_channels": 1, "default_arch": "simple_cnn", "task": "image"},
    "fashionmnist": {"num_classes": 10, "in_channels": 1, "default_arch": "simple_cnn", "task": "image"},
    "emnist": {"num_classes": 47, "in_channels": 1, "default_arch": "simple_cnn", "task": "image"},
    "kmnist": {"num_classes": 10, "in_channels": 1, "default_arch": "simple_cnn", "task": "image"},
    "food101": {"num_classes": 101, "in_channels": 3, "default_arch": "resnet50", "task": "image"},
    "flowers102": {"num_classes": 102, "in_channels": 3, "default_arch": "resnet34", "task": "image"},
    "stanforddogs": {"num_classes": 120, "in_channels": 3, "default_arch": "resnet50", "task": "image"},
    "stanfordcars": {"num_classes": 196, "in_channels": 3, "default_arch": "resnet50", "task": "image"},
    "oxfordpets": {"num_classes": 37, "in_channels": 3, "default_arch": "resnet34", "task": "image"},
    "catsvsdogs": {"num_classes": 2, "in_channels": 3, "default_arch": "resnet18", "task": "image"},
    "eurosat": {"num_classes": 10, "in_channels": 3, "default_arch": "medium_cnn", "task": "image"},
    "svhn": {"num_classes": 10, "in_channels": 3, "default_arch": "simple_cnn", "task": "image"},
    "caltech101": {"num_classes": 101, "in_channels": 3, "default_arch": "resnet34", "task": "image"},
    "caltech256": {"num_classes": 257, "in_channels": 3, "default_arch": "resnet50", "task": "image"},
    "stl10": {"num_classes": 10, "in_channels": 3, "default_arch": "medium_cnn", "task": "image"},
    
    # ═══════════════════════════════════════════════════════════════
    # TEXT - SENTIMENT (8)
    # ═══════════════════════════════════════════════════════════════
    "imdb": {"num_classes": 2, "default_arch": "distilbert", "task": "text", "max_length": 256},
    "sst2": {"num_classes": 2, "default_arch": "distilbert", "task": "text", "max_length": 128},
    "sst5": {"num_classes": 5, "default_arch": "distilbert", "task": "text", "max_length": 128},
    "yelpreviews": {"num_classes": 5, "default_arch": "distilbert", "task": "text", "max_length": 256},
    "amazonpolarity": {"num_classes": 2, "default_arch": "distilbert", "task": "text", "max_length": 256},
    "rottentomatoes": {"num_classes": 2, "default_arch": "distilbert", "task": "text", "max_length": 128},
    "financialsentiment": {"num_classes": 3, "default_arch": "distilbert", "task": "text", "max_length": 128},
    "tweetsentiment": {"num_classes": 3, "default_arch": "distilbert", "task": "text", "max_length": 128},
    
    # ═══════════════════════════════════════════════════════════════
    # TEXT - CLASSIFICATION (4)
    # ═══════════════════════════════════════════════════════════════
    "agnews": {"num_classes": 4, "default_arch": "distilbert", "task": "text", "max_length": 256},
    "dbpedia": {"num_classes": 14, "default_arch": "distilbert", "task": "text", "max_length": 256},
    "yahooanswers": {"num_classes": 10, "default_arch": "distilbert", "task": "text", "max_length": 256},
    "twentynewsgroups": {"num_classes": 20, "default_arch": "distilbert", "task": "text", "max_length": 512},
    
    # ═══════════════════════════════════════════════════════════════
    # TEXT - SPAM & TOXICITY (4)
    # ═══════════════════════════════════════════════════════════════
    "smsspam": {"num_classes": 2, "default_arch": "distilbert", "task": "text", "max_length": 128},
    "hatespeech": {"num_classes": 3, "default_arch": "distilbert", "task": "text", "max_length": 128},
    "civilcomments": {"num_classes": 2, "default_arch": "distilbert", "task": "text", "max_length": 256},
    "toxicity": {"num_classes": 2, "default_arch": "distilbert", "task": "text", "max_length": 256},
    
    # ═══════════════════════════════════════════════════════════════
    # TEXT - INTENT (3)
    # ═══════════════════════════════════════════════════════════════
    "clincintent": {"num_classes": 150, "default_arch": "distilbert", "task": "text", "max_length": 128},
    "banking77": {"num_classes": 77, "default_arch": "distilbert", "task": "text", "max_length": 128},
    "snipsintent": {"num_classes": 7, "default_arch": "distilbert", "task": "text", "max_length": 64},
    
    # ═══════════════════════════════════════════════════════════════
    # TEXT - NER (2)
    # ═══════════════════════════════════════════════════════════════
    "conll2003": {"num_classes": 9, "default_arch": "distilbert", "task": "ner", "max_length": 128},
    "wnut17": {"num_classes": 13, "default_arch": "distilbert", "task": "ner", "max_length": 128},
    
    # ═══════════════════════════════════════════════════════════════
    # TEXT - QA (5)
    # ═══════════════════════════════════════════════════════════════
    "squad": {"num_classes": 2, "default_arch": "distilbert", "task": "qa", "max_length": 384},
    "squadv2": {"num_classes": 2, "default_arch": "distilbert", "task": "qa", "max_length": 384},
    "triviaqa": {"num_classes": 2, "default_arch": "distilbert", "task": "qa", "max_length": 512},
    "boolq": {"num_classes": 2, "default_arch": "distilbert", "task": "text", "max_length": 256},
    "commonsenseqa": {"num_classes": 5, "default_arch": "distilbert", "task": "text", "max_length": 128},
    
    # ═══════════════════════════════════════════════════════════════
    # TEXT - SIMILARITY (5)
    # ═══════════════════════════════════════════════════════════════
    "stsb": {"num_classes": 1, "default_arch": "distilbert", "task": "regression", "max_length": 128},
    "mrpc": {"num_classes": 2, "default_arch": "distilbert", "task": "text", "max_length": 128},
    "qqp": {"num_classes": 2, "default_arch": "distilbert", "task": "text", "max_length": 128},
    "snli": {"num_classes": 3, "default_arch": "distilbert", "task": "text", "max_length": 128},
    "mnli": {"num_classes": 3, "default_arch": "distilbert", "task": "text", "max_length": 128},
    
    # ═══════════════════════════════════════════════════════════════
    # TEXT - SUMMARIZATION (3)
    # ═══════════════════════════════════════════════════════════════
    "cnndailymail": {"num_classes": 0, "default_arch": "distilbert", "task": "summarization", "max_length": 512},
    "xsum": {"num_classes": 0, "default_arch": "distilbert", "task": "summarization", "max_length": 512},
    "samsum": {"num_classes": 0, "default_arch": "distilbert", "task": "summarization", "max_length": 512},
    
    # ═══════════════════════════════════════════════════════════════
    # AUDIO - SPEECH (3)
    # ═══════════════════════════════════════════════════════════════
    "speechcommands": {"num_classes": 35, "default_arch": "audio_cnn", "task": "audio"},
    "librispeech": {"num_classes": 0, "default_arch": "audio_cnn", "task": "asr"},
    "commonvoice": {"num_classes": 0, "default_arch": "audio_cnn", "task": "asr"},
    
    # ═══════════════════════════════════════════════════════════════
    # AUDIO - MUSIC (4)
    # ═══════════════════════════════════════════════════════════════
    "gtzan": {"num_classes": 10, "default_arch": "audio_cnn", "task": "audio"},
    "esc50": {"num_classes": 50, "default_arch": "audio_cnn", "task": "audio"},
    "urbansound8k": {"num_classes": 10, "default_arch": "audio_cnn", "task": "audio"},
    "nsynth": {"num_classes": 11, "default_arch": "audio_cnn", "task": "audio"},
    
    # ═══════════════════════════════════════════════════════════════
    # AUDIO - EMOTION (3)
    # ═══════════════════════════════════════════════════════════════
    "ravdess": {"num_classes": 8, "default_arch": "audio_cnn", "task": "audio"},
    "cremad": {"num_classes": 6, "default_arch": "audio_cnn", "task": "audio"},
    "iemocap": {"num_classes": 4, "default_arch": "audio_cnn", "task": "audio"},
    
    # ═══════════════════════════════════════════════════════════════
    # TABULAR (10)
    # ═══════════════════════════════════════════════════════════════
    "iris": {"num_classes": 3, "input_dim": 4, "default_arch": "mlp", "task": "tabular"},
    "wine": {"num_classes": 3, "input_dim": 13, "default_arch": "mlp", "task": "tabular"},
    "diabetes": {"num_classes": 2, "input_dim": 8, "default_arch": "mlp", "task": "tabular"},
    "breastcancer": {"num_classes": 2, "input_dim": 30, "default_arch": "mlp", "task": "tabular"},
    "californiahousing": {"num_classes": 1, "input_dim": 8, "default_arch": "mlp", "task": "tabular_regression"},
    "adultincome": {"num_classes": 2, "input_dim": 14, "default_arch": "mlp", "task": "tabular"},
    "bankmarketing": {"num_classes": 2, "input_dim": 16, "default_arch": "mlp", "task": "tabular"},
    "creditdefault": {"num_classes": 2, "input_dim": 23, "default_arch": "mlp", "task": "tabular"},
    "titanic": {"num_classes": 2, "input_dim": 7, "default_arch": "mlp", "task": "tabular"},
    "heartdisease": {"num_classes": 2, "input_dim": 13, "default_arch": "mlp", "task": "tabular"},
    
    # ═══════════════════════════════════════════════════════════════
    # MEDICAL - IMAGES (7)
    # ═══════════════════════════════════════════════════════════════
    "chestxray": {"num_classes": 2, "in_channels": 1, "default_arch": "resnet18", "task": "image"},
    "skincancer": {"num_classes": 7, "in_channels": 3, "default_arch": "efficientnet_b1", "task": "image"},
    "diabeticretinopathy": {"num_classes": 5, "in_channels": 3, "default_arch": "resnet50", "task": "image"},
    "braintumor": {"num_classes": 4, "in_channels": 1, "default_arch": "resnet34", "task": "image"},
    "malaria": {"num_classes": 2, "in_channels": 3, "default_arch": "medium_cnn", "task": "image"},
    "bloodcells": {"num_classes": 4, "in_channels": 3, "default_arch": "medium_cnn", "task": "image"},
    "covidxray": {"num_classes": 3, "in_channels": 1, "default_arch": "resnet18", "task": "image"},
    
    # ═══════════════════════════════════════════════════════════════
    # MEDICAL - TEXT (2)
    # ═══════════════════════════════════════════════════════════════
    "pubmedqa": {"num_classes": 3, "default_arch": "distilbert", "task": "text", "max_length": 512},
    "medqa": {"num_classes": 4, "default_arch": "distilbert", "task": "text", "max_length": 512},
    
    # ═══════════════════════════════════════════════════════════════
    # TIME SERIES (4)
    # ═══════════════════════════════════════════════════════════════
    "electricity": {"num_classes": 1, "input_channels": 1, "default_arch": "timeseries_cnn", "task": "timeseries"},
    "weather": {"num_classes": 1, "input_channels": 21, "default_arch": "timeseries_cnn", "task": "timeseries"},
    "stockprices": {"num_classes": 1, "input_channels": 5, "default_arch": "timeseries_cnn", "task": "timeseries"},
    "ecgheartbeat": {"num_classes": 5, "input_channels": 1, "default_arch": "timeseries_cnn", "task": "timeseries"},
    
    # ═══════════════════════════════════════════════════════════════
    # CODE (4)
    # ═══════════════════════════════════════════════════════════════
    "codesearchnet": {"num_classes": 2, "default_arch": "distilbert", "task": "text", "max_length": 256},
    "humaneval": {"num_classes": 0, "default_arch": "distilbert", "task": "code_generation"},
    "mbpp": {"num_classes": 0, "default_arch": "distilbert", "task": "code_generation"},
    "spider": {"num_classes": 0, "default_arch": "distilbert", "task": "text2sql"},
    
    # ═══════════════════════════════════════════════════════════════
    # GRAPHS (3)
    # ═══════════════════════════════════════════════════════════════
    "cora": {"num_classes": 7, "input_dim": 1433, "default_arch": "graph_mlp", "task": "graph"},
    "citeseer": {"num_classes": 6, "input_dim": 3703, "default_arch": "graph_mlp", "task": "graph"},
    "qm9": {"num_classes": 1, "input_dim": 11, "default_arch": "mlp", "task": "graph_regression"},
    
    # ═══════════════════════════════════════════════════════════════
    # SECURITY (3)
    # ═══════════════════════════════════════════════════════════════
    "nslkdd": {"num_classes": 5, "input_dim": 41, "default_arch": "mlp", "task": "tabular"},
    "creditcardfraud": {"num_classes": 2, "input_dim": 30, "default_arch": "mlp", "task": "tabular"},
    "phishing": {"num_classes": 2, "input_dim": 30, "default_arch": "mlp", "task": "tabular"},
    
    # ═══════════════════════════════════════════════════════════════
    # RECOMMENDATION (2)
    # ═══════════════════════════════════════════════════════════════
    "movielens1m": {"num_classes": 5, "input_dim": 32, "default_arch": "mlp", "task": "tabular"},
    "movielens100k": {"num_classes": 5, "input_dim": 32, "default_arch": "mlp", "task": "tabular"},
    
    # ═══════════════════════════════════════════════════════════════
    # MULTILINGUAL (3)
    # ═══════════════════════════════════════════════════════════════
    "xnli": {"num_classes": 3, "default_arch": "distilbert", "task": "text", "max_length": 128},
    "amazonreviewsmulti": {"num_classes": 5, "default_arch": "distilbert", "task": "text", "max_length": 256},
    "sberquad": {"num_classes": 2, "default_arch": "distilbert", "task": "qa", "max_length": 384},
    
    # ═══════════════════════════════════════════════════════════════
    # CUSTOM (1)
    # ═══════════════════════════════════════════════════════════════
    "custom": {"num_classes": 10, "default_arch": "simple_cnn", "task": "custom"},
}


def get_dataset_config(name: str) -> Dict:
    """Get dataset configuration"""
    key = name.lower().replace("_", "").replace("-", "")
    if key not in DATASET_CONFIG:
        print(f"⚠️  Unknown dataset '{name}', using default")
        return {"num_classes": 10, "in_channels": 3, "default_arch": "simple_cnn", "task": "image"}
    return DATASET_CONFIG[key]


def list_supported_datasets() -> Dict[str, Dict]:
    """List all supported datasets"""
    return DATASET_CONFIG.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# IPFS CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class IPFSClient:
    """Simple IPFS client"""
    
    def __init__(self, gateway: str = "https://ipfs.io/ipfs/"):
        self.gateway = gateway
        self.local_store = None
    
    def download(self, cid: str, output_path: str) -> bool:
        if self.local_store:
            local = os.path.join(self.local_store, cid)
            if os.path.exists(local):
                import shutil
                shutil.copy(local, output_path)
                return True
        try:
            r = requests.get(f"{self.gateway}{cid}", timeout=60)
            r.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(r.content)
            return True
        except Exception as e:
            print(f"IPFS error: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# ML VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainerResult:
    trainer: str
    accuracy: float
    improvement: float
    contribution_bps: int


class MLValidator:
    """ML Validator for federated learning"""
    
    def __init__(self, data_dir: str = "./validator_data", device: str = None):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ipfs = IPFSClient()
        print(f"🔬 MLValidator initialized on {self.device}")
    
    def load_model(self, path: str, dataset: str, arch: str) -> nn.Module:
        config = get_dataset_config(dataset)
        model = create_model(arch, config.get("num_classes", 10), config.get("in_channels", 3), config.get("input_dim"))
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return model.to(self.device).eval()
    
    def evaluate(self, model: nn.Module, test_loader: DataLoader = None) -> float:
        if not test_loader:
            return 10.0
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        return 100.0 * correct / total if total else 0.0
    
    def apply_gradients(self, model: nn.Module, grads: Dict[str, torch.Tensor], lr: float = 1.0):
        with torch.no_grad():
            sd = model.state_dict()
            for k in sd:
                if k in grads:
                    sd[k] += lr * grads[k].to(self.device)
            model.load_state_dict(sd)
        return model
    
    def validate_round(self, model_cid: str, submissions: List[Tuple[str, str]], 
                       dataset: str, arch: str = None, test_loader: DataLoader = None) -> Dict[str, int]:
        """Full validation workflow"""
        config = get_dataset_config(dataset)
        arch = arch or config.get("default_arch", "simple_cnn")
        
        model_path = os.path.join(self.data_dir, f"model_{model_cid[:16]}.pt")
        if not os.path.exists(model_path):
            self.ipfs.download(model_cid, model_path)
        
        baseline = self.load_model(model_path, dataset, arch)
        baseline_acc = self.evaluate(baseline, test_loader)
        print(f"\n📊 Baseline: {baseline_acc:.2f}%")
        
        results = []
        for trainer, grad_cid in submissions:
            grad_path = os.path.join(self.data_dir, f"grad_{grad_cid[:16]}.pt")
            try:
                if not os.path.exists(grad_path):
                    self.ipfs.download(grad_cid, grad_path)
                model = self.load_model(model_path, dataset, arch)
                grads = torch.load(grad_path, map_location=self.device, weights_only=True)
                model = self.apply_gradients(model, grads)
                acc = self.evaluate(model, test_loader)
                imp = acc - baseline_acc
                print(f"   {trainer[:16]}... {acc:.2f}% ({imp:+.2f}%)")
                results.append(TrainerResult(trainer, acc, imp, 0))
            except Exception as e:
                print(f"   {trainer[:16]}... FAILED: {e}")
                results.append(TrainerResult(trainer, 0, -100, 0))
        
        # Calculate contributions
        valid = [r for r in results if r.improvement > -99]
        positive = [r for r in valid if r.improvement > 0]
        
        if not positive:
            bps = 10000 // len(valid) if valid else 0
            for r in valid:
                r.contribution_bps = bps
        else:
            total = sum(r.improvement for r in positive)
            for r in valid:
                r.contribution_bps = int((r.improvement / total) * 10000) if r.improvement > 0 else 0
            diff = 10000 - sum(r.contribution_bps for r in results)
            for r in results:
                if r.contribution_bps > 0:
                    r.contribution_bps += diff
                    break
        
        return {r.trainer: r.contribution_bps for r in results}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def list_supported_architectures() -> List[str]:
    return ["simple_cnn", "medium_cnn", "resnet18", "resnet34", "resnet50",
            "efficientnet_b0", "efficientnet_b1", "mobilenet_v2", "vgg11", "densenet121",
            "vit_b_16", "mlp", "audio_cnn", "text_cnn", "timeseries_cnn", "graph_mlp",
            "distilbert", "bert"]


def count_datasets_by_task() -> Dict[str, int]:
    counts = {}
    for c in DATASET_CONFIG.values():
        t = c.get("task", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return counts


def print_dataset_summary():
    counts = count_datasets_by_task()
    print(f"\n{'═'*50}")
    print(f"  📊 DATASETS: {sum(counts.values())} total")
    print(f"{'═'*50}")
    for t, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {t:<20} {c:>3}")
    print(f"{'═'*50}\n")