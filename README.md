# DECLOUD Validator Kit

<p align="center">
  <img src="https://img.shields.io/badge/DECLOUD-Validator-blue?style=for-the-badge" alt="DECLOUD Validator"/>
  <img src="https://img.shields.io/badge/Solana-Mainnet-green?style=for-the-badge" alt="Solana Mainnet"/>
  <img src="https://img.shields.io/badge/Python-3.9+-yellow?style=for-the-badge" alt="Python 3.9+"/>
</p>

**DECLOUD** is a decentralized cloud platform for distributed machine learning on Solana.

Validators earn rewards by evaluating training contributions from trainers. This toolkit provides everything you need to become a validator.

## 🚀 Quick Start

### Installation

```bash
pip install decloud-validator
```

Or install from source:

```bash
git clone https://github.com/jorikkkbrooo/validator-kit
cd validator-kit
pip install -e .
```

### Prerequisites

- Python 3.9+
- Node.js 18+
- Solana CLI (optional)

Install Node.js dependencies:

```bash
npm install @solana/web3.js @coral-xyz/anchor bs58
```

### Setup

```bash
# Download validation datasets (minimal ~500MB)
decloud datasets download --minimal

# Start validating
decloud validate start --private-key <YOUR_BASE58_PRIVATE_KEY>
```

## 📦 Datasets

DECLOUD supports 80+ datasets across multiple domains:

| Category | Count | Examples |
|----------|-------|----------|
| Images Classification | 16 | CIFAR-10, MNIST, ImageNet |
| Text Sentiment | 8 | IMDB, SST-2, Yelp |
| Text Classification | 7 | AG News, DBpedia |
| Audio | 10 | Speech Commands, GTZAN |
| Tabular | 10 | Iris, Titanic, Adult Income |
| Medical | 9 | Chest X-Ray, Skin Cancer |
| Code | 4 | HumanEval, MBPP, Spider |
| And more... | 20+ | Graphs, Time Series, Security |

### Download Datasets

```bash
# Download minimal pack (~500MB)
decloud datasets download --minimal

# Download all datasets (~50GB, skips very large ones)
decloud datasets download

# Download specific category
decloud datasets download --category images_classification

# Download specific dataset
decloud datasets download --dataset Cifar10

# Show download status
decloud datasets list --status
```

## 🔧 Configuration

```bash
# Show current config
decloud config show

# Set values
decloud config set network devnet
decloud config set min_reward 0.01
decloud config set auto_claim true
```

Configuration file: `~/.decloud/config.json`

Environment variables:
- `DECLOUD_PRIVATE_KEY` - Validator private key
- `DECLOUD_RPC_URL` - Solana RPC URL
- `DECLOUD_NETWORK` - Network (devnet/mainnet)
- `DECLOUD_DATA_DIR` - Dataset storage directory

## 💻 Python API

```python
from decloud import Validator, DatasetManager, Config

# Initialize
config = Config.load()
validator = Validator(config)

# Login
validator.login("your_base58_private_key")
print(f"Balance: {validator.balance} SOL")

# Download datasets
datasets = DatasetManager()
datasets.download_minimal()

# Get available rounds
rounds = validator.get_available_rounds()
for r in rounds:
    print(f"Round #{r.id}: {r.reward_amount/1e9} SOL")

# Claim and validate a round
validator.claim_round(round_id=1)
validator.start_training(round_id=1)
# ... wait for submissions ...
validator.evaluate_and_complete(round_id=1)

# Or run in auto mode
validator.start()  # Runs forever, auto-claims and validates
```

### Custom Evaluation

```python
def my_evaluator(submission):
    """
    Custom evaluator function.
    Returns a score between 0 and 1.
    """
    # Download and evaluate gradients
    gradients = download_gradients(submission.gradient_cid)
    score = evaluate_gradients(gradients)
    return score

validator.start(evaluator=my_evaluator)
```

## 📊 Validation Flow

```
1. Creator creates round (deposits SOL)
         ↓
2. Validator claims round
         ↓
3. Trainers join (max 10)
         ↓
4. Validator starts training
         ↓
5. Trainers submit gradients
         ↓
6. Validator evaluates submissions
   - Set contribution % for each trainer
         ↓
7. Validator completes validation
   - Receives 10% fee automatically
         ↓
8. Trainers claim their rewards
```

## 🔒 Security

- Never share your private key
- Use environment variables in production
- Private keys are NOT saved to config file by default
- Consider using a separate wallet for validation

## 🌐 Networks

| Network | RPC URL | Status |
|---------|---------|--------|
| Mainnet | `https://api.mainnet-beta.solana.com` | ✅ Active |
| Devnet | `https://api.devnet.solana.com` | ✅ Testing |

## 📚 CLI Reference

```bash
# General
decloud info                    # Show system info
decloud --version               # Show version

# Authentication
decloud login <private_key>     # Login with base58 key
decloud login -f keypair.json   # Login from file

# Datasets
decloud datasets list           # List all datasets
decloud datasets list --status  # Show download status
decloud datasets download       # Download all
decloud datasets download -m    # Download minimal
decloud datasets download -c <category>
decloud datasets download -d <dataset>

# Validation
decloud validate start          # Start validator loop
decloud validate start -k <key> # With private key

# Rounds
decloud rounds list             # List all rounds

# Configuration
decloud config show             # Show config
decloud config set <key> <val>  # Set value
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- Website: https://decloud.network
- Documentation: https://depincloud.com/docs
- Twitter: https://x.com/CloudDepin

---

<p align="center">
  <b>DECLOUD</b> - Decentralized Cloud for AI Training
  <br/>
  <i>Built on Solana</i>
</p>