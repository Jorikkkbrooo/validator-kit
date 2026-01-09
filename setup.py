#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="decloud-validator",
    version="1.0.0",
    description="Decloud Validator - Automated validation for federated learning on Solana",
    author="Decloud Team",
    python_requires=">=3.9",
    py_modules=[
        "main",
        "config",
        "dataset_manager",
        "model_loader",
        "ipfs_client",
        "solana_client",
        "websocket_listener",
        "validator",
    ],
    install_requires=[
        "solana>=0.30.0",
        "solders>=0.18.0",
        "torch>=2.0.0",
        "safetensors>=0.4.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "aiohttp>=3.9.0",
        "aiofiles>=23.0.0",
        "websockets>=12.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "pyyaml>=6.0.0",
        "python-dotenv>=1.0.0",
        "base58>=2.1.0",
    ],
    entry_points={
        "console_scripts": [
            "decloud-validator=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
