"""
Configuration management for Decloud Validator
"""
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# Paths
HOME_DIR = Path.home()
CONFIG_DIR = HOME_DIR / ".decloud-validator"
CONFIG_FILE = CONFIG_DIR / "config.json"
DATASETS_DIR = CONFIG_DIR / "datasets"
CACHE_DIR = CONFIG_DIR / "cache"
MODELS_CACHE = CACHE_DIR / "models"

# Solana
PROGRAM_ID = "HvQ8c3BBCsibJpH74UDXbvqEidJymzFYyGjNjRn7MYwC"
TREASURY = "FzuCxi65QyFXAGbHcXB28RXqyBZSZ5KXLQxeofx1P9K2"

# RPC Endpoints
RPC_ENDPOINTS = {
    "devnet": "https://api.devnet.solana.com",
    "mainnet": "https://api.mainnet-beta.solana.com",
    "testnet": "https://api.testnet.solana.com",
}

# IPFS Gateways (fallback order)
IPFS_GATEWAYS = [
    "https://ipfs.io/ipfs/",
    "https://gateway.pinata.cloud/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/",
    "https://dweb.link/ipfs/",
]

# Dataset enum mapping (must match contract)
DATASETS = {
    "Cifar10": 0, "Cifar100": 1, "Mnist": 2, "FashionMnist": 3, "Emnist": 4,
    "Kmnist": 5, "Food101": 6, "Flowers102": 7, "StanfordDogs": 8, "StanfordCars": 9,
    "OxfordPets": 10, "CatsVsDogs": 11, "Eurosat": 12, "Svhn": 13, "Caltech101": 14,
    "Caltech256": 15, "Imdb": 16, "Sst2": 17, "Sst5": 18, "YelpReviews": 19,
    "AmazonPolarity": 20, "RottenTomatoes": 21, "FinancialSentiment": 22, "TweetSentiment": 23,
    "AgNews": 24, "Dbpedia": 25, "YahooAnswers": 26, "TwentyNewsgroups": 27,
    "SmsSpam": 28, "HateSpeech": 29, "CivilComments": 30, "Toxicity": 31,
    "ClincIntent": 32, "Banking77": 33, "SnipsIntent": 34, "Conll2003": 35,
    "Wnut17": 36, "Squad": 37, "SquadV2": 38, "TriviaQa": 39, "BoolQ": 40,
    "CommonsenseQa": 41, "Stsb": 42, "Mrpc": 43, "Qqp": 44, "Snli": 45,
    "Mnli": 46, "CnnDailymail": 47, "Xsum": 48, "Samsum": 49, "SpeechCommands": 50,
    "Librispeech": 51, "CommonVoice": 52, "Gtzan": 53, "Esc50": 54, "Urbansound8k": 55,
    "Nsynth": 56, "Ravdess": 57, "CremaD": 58, "Iemocap": 59, "Iris": 60,
    "Wine": 61, "Diabetes": 62, "BreastCancer": 63, "CaliforniaHousing": 64,
    "AdultIncome": 65, "BankMarketing": 66, "CreditDefault": 67, "Titanic": 68,
    "HeartDisease": 69, "ChestXray": 70, "SkinCancer": 71, "DiabeticRetinopathy": 72,
    "BrainTumor": 73, "Malaria": 74, "BloodCells": 75, "CovidXray": 76,
    "PubmedQa": 77, "MedQa": 78, "Electricity": 79, "Weather": 80, "StockPrices": 81,
    "EcgHeartbeat": 82, "CodeSearchNet": 83, "Humaneval": 84, "Mbpp": 85,
    "Spider": 86, "Cora": 87, "Citeseer": 88, "Qm9": 89, "NslKdd": 90,
    "CreditCardFraud": 91, "Phishing": 92, "Movielens1m": 93, "Movielens100k": 94,
    "Xnli": 95, "AmazonReviewsMulti": 96, "Sberquad": 97,
}

# Reverse mapping
DATASET_ID_TO_NAME = {v: k for k, v in DATASETS.items()}


class Config:
    """Validator configuration"""
    
    def __init__(self):
        self.private_key: Optional[str] = None
        self.network: str = "mainnet"
        self.installed_datasets: List[str] = []
        self.auto_validate: bool = True
        self.poll_interval: int = 30  # seconds
        self.validation_batch_size: int = 1000
        self.max_concurrent_validations: int = 3
        
        self._ensure_dirs()
        self._load()
    
    def _ensure_dirs(self):
        """Create necessary directories"""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_CACHE.mkdir(parents=True, exist_ok=True)
    
    def _load(self):
        """Load config from file"""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                self.private_key = data.get("private_key")
                self.network = data.get("network", "devnet")
                self.installed_datasets = data.get("installed_datasets", [])
                self.auto_validate = data.get("auto_validate", True)
                self.poll_interval = data.get("poll_interval", 30)
                self.validation_batch_size = data.get("validation_batch_size", 1000)
                self.max_concurrent_validations = data.get("max_concurrent_validations", 3)
    
    def save(self):
        """Save config to file"""
        data = {
            "private_key": self.private_key,
            "network": self.network,
            "installed_datasets": self.installed_datasets,
            "auto_validate": self.auto_validate,
            "poll_interval": self.poll_interval,
            "validation_batch_size": self.validation_batch_size,
            "max_concurrent_validations": self.max_concurrent_validations,
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    
    @property
    def rpc_url(self) -> str:
        return RPC_ENDPOINTS.get(self.network, RPC_ENDPOINTS["devnet"])
    
    def is_dataset_installed(self, dataset_name: str) -> bool:
        return dataset_name in self.installed_datasets
    
    def add_dataset(self, dataset_name: str):
        if dataset_name not in self.installed_datasets:
            self.installed_datasets.append(dataset_name)
            self.save()
    
    def remove_dataset(self, dataset_name: str):
        if dataset_name in self.installed_datasets:
            self.installed_datasets.remove(dataset_name)
            self.save()


# Global config instance
config = Config()
