"""
DECLOUD Validator Kit
=====================

Decentralized Cloud for Distributed Learning

A toolkit for validators to participate in DECLOUD network:
- Download and manage datasets
- Validate training rounds
- Earn rewards for honest validation

Quick Start:
    from decloud import Validator
    
    validator = Validator()
    validator.login("your_private_key")
    validator.download_datasets()
    validator.start()

Documentation: https://docs.decloud.network
"""

__version__ = "0.1.0"
__author__ = "DECLOUD Team"

from .validator import Validator
from .dataset_manager import DatasetManager
from .config import Config
from .monitor import Monitor, LiveValidator
from .ml_validator import MLValidator

__all__ = [
    "Validator",
    "DatasetManager", 
    "Config",
    "Monitor",
    "LiveValidator",
    "MLValidator",
]