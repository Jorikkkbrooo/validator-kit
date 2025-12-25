#!/usr/bin/env python3
"""
DECLOUD Validator Kit
=====================

Install:
    pip install decloud-validator

Usage:
    decloud datasets download --minimal
    decloud validate start --private-key <key>
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="decloud-validator",
    version="0.1.0",
    author="DECLOUD Team",
    author_email="team@decloud.network",
    description="Validator toolkit for DECLOUD - Decentralized Cloud for AI Training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/decloud-network/validator-kit",
    project_urls={
        "Documentation": "https://docs.decloud.network",
        "Bug Tracker": "https://github.com/decloud-network/validator-kit/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.9",
    install_requires=[
        "datasets>=2.14.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "audio": [
            "torchaudio>=2.0.0",
            "librosa>=0.10.0",
        ],
        "full": [
            "transformers>=4.30.0",
            "scikit-learn>=1.3.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "decloud=decloud.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "decloud": ["idl/*.json"],
        "": ["idl.json"],
    },
    data_files=[
        ("", ["idl.json"]),
    ],
)