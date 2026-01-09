# Decloud Validator Kit

Automated validation for federated learning rounds on Solana blockchain.

## Features

- 🔐 Secure wallet integration with private key
- 📦 Dataset installation and management
- 🤖 Automatic pre-validation and post-validation
- 💰 Easy reward claiming
- 🔄 Background auto-validation loop
- 🎯 Support for any model architecture

## Installation

```bash
# Clone or extract the validator kit
cd decloud-validator-kit

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Setup

```bash
python main.py setup
```

This will prompt you for:
- Your Solana wallet private key (base58 encoded)
- Network selection (devnet/mainnet/testnet)

### 2. Install Datasets

```bash
# List available datasets
python main.py dataset list --available

# Install specific datasets
python main.py dataset install Cifar10
python main.py dataset install Mnist
python main.py dataset install Imdb

# Install common datasets at once
python main.py dataset install-all
```

### 3. Start Validating

```bash
# Show status
python main.py status

# View active rounds
python main.py rounds

# Start auto-validation
python main.py start
```

## Commands Reference

### Setup & Configuration

| Command | Description |
|---------|-------------|
| `setup` | Interactive setup wizard |
| `network -n <network>` | Change network (devnet/mainnet/testnet) |
| `status` | Show validator status |
| `balance` | Show wallet balance |

### Dataset Management

| Command | Description |
|---------|-------------|
| `dataset list` | List installed datasets |
| `dataset list --available` | List all available datasets |
| `dataset install <name>` | Install a dataset |
| `dataset uninstall <name>` | Remove a dataset |
| `dataset install-all` | Install common datasets |
| `dataset install-all -c image` | Install all image datasets |

### Validation

| Command | Description |
|---------|-------------|
| `start` | Start auto-validation loop |
| `rounds` | Show active rounds |
| `round-info <id>` | Show detailed round info |
| `prevalidate <id>` | Manually prevalidate a round |
| `postvalidate <id> <trainer>` | Manually postvalidate a gradient |

### Rewards

| Command | Description |
|---------|-------------|
| `claim <round_id>` | Claim rewards from finalized round |

## Supported Datasets

### Image Classification
- Cifar10, Cifar100
- Mnist, FashionMnist, Emnist, Kmnist
- Food101, Flowers102
- StanfordDogs, StanfordCars, OxfordPets
- Eurosat, Svhn, Caltech101, Caltech256

### Text Classification
- Imdb, Sst2, Sst5
- AgNews, Dbpedia, YahooAnswers
- YelpReviews, AmazonPolarity, RottenTomatoes
- Banking77, SnipsIntent

### Tabular
- Iris, Wine, Diabetes, BreastCancer
- Titanic, HeartDisease
- CreditDefault, BankMarketing

### Medical
- ChestXray, SkinCancer
- DiabeticRetinopathy, BrainTumor
- Malaria, BloodCells

## Model Package Format

Models are stored on IPFS with the following structure:

```
model_package/
├── config.json         # Architecture definition
├── head.safetensors    # Model weights
└── embeddings.safetensors  # Pre-computed embeddings (optional)
```

### config.json Example

```json
{
  "head": {
    "layers": [
      {"type": "Linear", "params": {"in_features": 768, "out_features": 256}},
      {"type": "ReLU", "params": {}},
      {"type": "Dropout", "params": {"p": 0.1}},
      {"type": "Linear", "params": {"in_features": 256, "out_features": 10}}
    ]
  }
}
```

### Supported Layer Types

- Linear, Conv1d, Conv2d
- BatchNorm1d, BatchNorm2d, LayerNorm
- ReLU, GELU, SiLU, Tanh, Sigmoid
- Dropout, Dropout2d
- MaxPool1d/2d, AvgPool1d/2d, AdaptiveAvgPool1d/2d
- Flatten, Embedding
- LSTM, GRU
- MultiheadAttention, TransformerEncoderLayer

## How Validation Works

### Pre-validation
1. Validator checks active rounds matching installed datasets
2. Downloads model from IPFS
3. Runs model on test set
4. Submits accuracy to blockchain

### Post-validation
1. When trainers submit gradients, validators check them
2. Downloads gradient model from IPFS
3. Validates against same test set
4. Submits accuracy (improvement is calculated on-chain)

### Rewards
- 10% of round reward goes to validators
- Split equally among all validations
- 2% treasury fee on claims

## Configuration

Config file location: `~/.decloud-validator/config.json`

```json
{
  "private_key": "...",
  "network": "devnet",
  "installed_datasets": ["Cifar10", "Mnist"],
  "auto_validate": true,
  "poll_interval": 30,
  "validation_batch_size": 1000,
  "max_concurrent_validations": 3
}
```

## Troubleshooting

### "No private key configured"
Run `python main.py setup` to configure your wallet.

### "Dataset not installed"
Install the required dataset with `python main.py dataset install <name>`.

### "Prevalidation closed"
A trainer has already submitted a gradient, so prevalidation is no longer accepted.

### Low balance warning
You need SOL to pay transaction fees. Get devnet SOL from a faucet.

## Security Notes

- Private key is stored locally in `~/.decloud-validator/config.json`
- Never share your private key
- Use a dedicated validator wallet, not your main wallet
- Consider using environment variables for production

## Development

```bash
# Run with debug output
python main.py --help

# Test validation locally
python -c "from validator import Validator; v = Validator('your-key'); print(v.get_balance())"
```

## License

MIT License
