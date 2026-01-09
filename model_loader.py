"""
Dynamic model loader for any architecture
Supports loading models from IPFS with config.json, head.safetensors, embeddings.safetensors
"""
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import hashlib

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors

from config import MODELS_CACHE


class DynamicHead(nn.Module):
    """
    Dynamic head module that builds architecture from config
    Supports: Linear, Conv2d, BatchNorm, ReLU, Dropout, etc.
    """
    
    SUPPORTED_LAYERS = {
        "Linear": nn.Linear,
        "Conv1d": nn.Conv1d,
        "Conv2d": nn.Conv2d,
        "BatchNorm1d": nn.BatchNorm1d,
        "BatchNorm2d": nn.BatchNorm2d,
        "LayerNorm": nn.LayerNorm,
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "SiLU": nn.SiLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid,
        "Softmax": nn.Softmax,
        "Dropout": nn.Dropout,
        "Dropout2d": nn.Dropout2d,
        "MaxPool1d": nn.MaxPool1d,
        "MaxPool2d": nn.MaxPool2d,
        "AvgPool1d": nn.AvgPool1d,
        "AvgPool2d": nn.AvgPool2d,
        "AdaptiveAvgPool1d": nn.AdaptiveAvgPool1d,
        "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
        "Flatten": nn.Flatten,
        "Embedding": nn.Embedding,
        "LSTM": nn.LSTM,
        "GRU": nn.GRU,
        "MultiheadAttention": nn.MultiheadAttention,
        "TransformerEncoderLayer": nn.TransformerEncoderLayer,
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        
        self._build_from_config(config)
    
    def _build_from_config(self, config: Dict[str, Any]):
        """Build network from config"""
        layers_config = config.get("layers", [])
        
        for layer_cfg in layers_config:
            layer_type = layer_cfg.get("type")
            params = layer_cfg.get("params", {})
            
            if layer_type not in self.SUPPORTED_LAYERS:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            
            layer_cls = self.SUPPORTED_LAYERS[layer_type]
            
            # Handle special cases
            if layer_type in ["ReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "Flatten"]:
                layer = layer_cls(**params) if params else layer_cls()
            else:
                layer = layer_cls(**params)
            
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ModelPackage:
    """
    Loaded model package with head and embeddings
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        head: nn.Module,
        embeddings: Optional[torch.Tensor] = None,
    ):
        self.config = config
        self.head = head
        self.embeddings = embeddings
        self.device = "cpu"
    
    def to(self, device: str) -> "ModelPackage":
        """Move to device"""
        self.device = device
        self.head = self.head.to(device)
        if self.embeddings is not None:
            self.embeddings = self.embeddings.to(device)
        return self
    
    def eval(self) -> "ModelPackage":
        """Set to evaluation mode"""
        self.head.eval()
        return self
    
    @torch.no_grad()
    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Run prediction on embeddings"""
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device)
        return self.head(embeddings)
    
    @torch.no_grad()
    def predict_batch(
        self,
        embeddings: np.ndarray,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Run prediction on batch of embeddings"""
        self.head.eval()
        results = []
        
        for i in range(0, len(embeddings), batch_size):
            batch = torch.tensor(embeddings[i:i+batch_size], dtype=torch.float32)
            batch = batch.to(self.device)
            output = self.head(batch)
            results.append(output.cpu().numpy())
        
        return np.concatenate(results, axis=0)


class ModelLoader:
    """
    Loads model packages from local files or IPFS
    """
    
    def __init__(self, cache_dir: Path = MODELS_CACHE):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_from_directory(self, path: Path) -> ModelPackage:
        """Load model from local directory"""
        config_path = path / "config.json"
        head_path = path / "head.safetensors"
        embeddings_path = path / "embeddings.safetensors"
        
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {path}")
        
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Build head from config
        head_config = config.get("head", config)  # Support both formats
        head = DynamicHead(head_config)
        
        # Load head weights if available
        if head_path.exists():
            state_dict = load_safetensors(str(head_path))
            # Map weights to layers
            self._load_weights_to_head(head, state_dict)
        
        # Load embeddings if available
        embeddings = None
        if embeddings_path.exists():
            embeddings_dict = load_safetensors(str(embeddings_path))
            # Assume single tensor or combine all
            if len(embeddings_dict) == 1:
                embeddings = list(embeddings_dict.values())[0]
            else:
                embeddings = embeddings_dict.get("embeddings")
        
        return ModelPackage(config, head, embeddings)
    
    def _load_weights_to_head(self, head: DynamicHead, state_dict: Dict[str, torch.Tensor]):
        """Load weights into dynamic head"""
        # Try direct loading first
        try:
            head.load_state_dict(state_dict, strict=False)
            return
        except Exception:
            pass
        
        # Manual mapping for common patterns
        layer_idx = 0
        for name, tensor in state_dict.items():
            # Try to find matching layer
            for i, layer in enumerate(head.layers):
                if hasattr(layer, "weight"):
                    layer_state = layer.state_dict()
                    if "weight" in layer_state:
                        if layer_state["weight"].shape == tensor.shape:
                            layer.weight.data = tensor
                            if "bias" in state_dict.get(name.replace("weight", "bias"), {}):
                                layer.bias.data = state_dict[name.replace("weight", "bias")]
                            break
    
    def load_from_cache(self, cid: str) -> Optional[ModelPackage]:
        """Load model from cache by CID"""
        cache_path = self.cache_dir / cid
        if cache_path.exists():
            return self.load_from_directory(cache_path)
        return None
    
    def cache_model(self, cid: str, data: Dict[str, bytes]) -> Path:
        """Cache model data"""
        cache_path = self.cache_dir / cid
        cache_path.mkdir(parents=True, exist_ok=True)
        
        for filename, content in data.items():
            file_path = cache_path / filename
            with open(file_path, "wb") as f:
                f.write(content)
        
        return cache_path
    
    def get_cache_path(self, cid: str) -> Path:
        """Get cache path for CID"""
        return self.cache_dir / cid


# Simple architecture configs for common cases
SIMPLE_ARCHITECTURES = {
    "mlp_classifier": {
        "layers": [
            {"type": "Linear", "params": {"in_features": 768, "out_features": 256}},
            {"type": "ReLU", "params": {}},
            {"type": "Dropout", "params": {"p": 0.1}},
            {"type": "Linear", "params": {"in_features": 256, "out_features": 10}},
        ]
    },
    "simple_linear": {
        "layers": [
            {"type": "Linear", "params": {"in_features": 768, "out_features": 10}},
        ]
    },
    "deep_classifier": {
        "layers": [
            {"type": "Linear", "params": {"in_features": 768, "out_features": 512}},
            {"type": "BatchNorm1d", "params": {"num_features": 512}},
            {"type": "ReLU", "params": {}},
            {"type": "Dropout", "params": {"p": 0.2}},
            {"type": "Linear", "params": {"in_features": 512, "out_features": 256}},
            {"type": "BatchNorm1d", "params": {"num_features": 256}},
            {"type": "ReLU", "params": {}},
            {"type": "Dropout", "params": {"p": 0.2}},
            {"type": "Linear", "params": {"in_features": 256, "out_features": 10}},
        ]
    },
}


def create_simple_head(arch_name: str, num_classes: int, input_dim: int = 768) -> DynamicHead:
    """Create a simple head from predefined architecture"""
    if arch_name not in SIMPLE_ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch_name}")
    
    config = SIMPLE_ARCHITECTURES[arch_name].copy()
    config = {"layers": [layer.copy() for layer in config["layers"]]}
    
    # Update dimensions
    for layer in config["layers"]:
        params = layer.get("params", {})
        if "in_features" in params and params["in_features"] == 768:
            params["in_features"] = input_dim
        if "out_features" in params and params["out_features"] == 10:
            params["out_features"] = num_classes
        if "num_features" in params:
            # Keep intermediate dimensions
            pass
    
    return DynamicHead(config)


# Global loader instance
model_loader = ModelLoader()
