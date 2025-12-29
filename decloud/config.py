"""
DECLOUD Configuration
=====================

Configuration management for DECLOUD Validator.
"""

import os
import json
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, asdict


# Network presets
NETWORKS = {
    "devnet": {
        "rpc_url": "https://api.devnet.solana.com",
        "ws_url": "wss://api.devnet.solana.com",
    },
    "mainnet-beta": {
        "rpc_url": "https://api.mainnet-beta.solana.com",
        "ws_url": "wss://api.mainnet-beta.solana.com",
    },
    "localhost": {
        "rpc_url": "http://localhost:8899",
        "ws_url": "ws://localhost:8900",
    },
}

# Default program ID from lib.rs
DEFAULT_PROGRAM_ID = "HvQ8c3BBCsibJpH74UDXbvqEidJymzFYyGjNjRn7MYwC"


@dataclass
class Config:
    """
    DECLOUD Validator Configuration
    
    Load from file:
        config = Config.load()
    
    From environment:
        config = Config.from_env()
    
    Create manually:
        config = Config(network="mainnet-beta", min_reward=0.01)
    """
    
    # Network settings
    network: str = "mainnet-beta"
    rpc_url: str = "https://api.mainnet-beta.solana.com"
    ws_url: str = "wss://api.mainnet-beta.solana.com"
    program_id: str = DEFAULT_PROGRAM_ID
    
    # Paths
    data_dir: str = "./data"
    idl_path: str = ""  # Will be auto-detected
    config_path: str = "~/.decloud/config.json"
    
    # Auth
    private_key: Optional[str] = None
    
    # Filters
    min_reward: float = 0.0  # SOL
    max_reward: float = float("inf")  # SOL
    allowed_datasets: Optional[List[str]] = None
    blocked_creators: Optional[List[str]] = None
    only_downloaded: bool = True
    
    # Automation
    auto_claim: bool = True
    auto_start: bool = True
    auto_validate: bool = True
    max_concurrent_rounds: int = 1
    dry_run: bool = False
    
    # Timing
    poll_interval: int = 3  # seconds
    claim_delay: float = 0.5  # seconds
    validation_timeout: int = 300  # seconds
    
    # Monitoring
    use_websocket: bool = True
    sound_alerts: bool = False
    
    def __post_init__(self):
        """Apply network presets if network changed"""
        if self.network in NETWORKS:
            preset = NETWORKS[self.network]
            if self.rpc_url == "https://api.mainnet-beta.solana.com":
                self.rpc_url = preset["rpc_url"]
            if self.ws_url == "wss://api.mainnet-beta.solana.com":
                self.ws_url = preset["ws_url"]
        
        # Expand paths
        self.config_path = os.path.expanduser(self.config_path)
        self.data_dir = os.path.expanduser(self.data_dir)
        
        # Auto-detect IDL path if not set
        if not self.idl_path:
            self.idl_path = self._find_idl().replace("\\", "/")
        else:
            self.idl_path = os.path.expanduser(self.idl_path)
    
    def _find_idl(self) -> str:
        """Find IDL file in common locations"""
        # Get package directory
        package_dir = os.path.dirname(os.path.abspath(__file__))
        kit_dir = os.path.dirname(package_dir)
        
        # Search paths in order of priority
        search_paths = [
            # 1. Next to the package (decloud-validator-kit/idl.json)
            os.path.join(kit_dir, "idl.json"),
            # 2. Inside package
            os.path.join(package_dir, "idl.json"),
            # 3. Current directory
            os.path.join(os.getcwd(), "idl.json"),
            # 4. Home directory
            os.path.expanduser("~/.decloud/idl.json"),
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        # Fallback - will error later if not found
        return os.path.join(kit_dir, "idl.json")
    
    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """
        Load configuration from file.
        
        Priority:
        1. Provided path
        2. DECLOUD_CONFIG env var
        3. ~/.decloud/config.json
        4. ./decloud.json
        """
        config_path = path or os.getenv("DECLOUD_CONFIG")
        
        if not config_path:
            for p in ["~/.decloud/config.json", "./decloud.json"]:
                expanded = os.path.expanduser(p)
                if os.path.exists(expanded):
                    config_path = expanded
                    break
        
        if config_path and os.path.exists(os.path.expanduser(config_path)):
            with open(os.path.expanduser(config_path)) as f:
                data = json.load(f)
                return cls(**data)
        
        # Return default config
        return cls()
    
    @classmethod
    def from_env(cls) -> "Config":
        """
        Create config from environment variables.
        
        Supported env vars:
        - DECLOUD_NETWORK
        - DECLOUD_RPC_URL
        - DECLOUD_WS_URL
        - DECLOUD_PROGRAM_ID
        - DECLOUD_PRIVATE_KEY
        - DECLOUD_DATA_DIR
        - DECLOUD_IDL_PATH
        - DECLOUD_MIN_REWARD
        - DECLOUD_MAX_REWARD
        - DECLOUD_DRY_RUN
        - DECLOUD_AUTO_CLAIM
        - DECLOUD_POLL_INTERVAL
        """
        config = cls.load()
        
        env_map = {
            "DECLOUD_NETWORK": "network",
            "DECLOUD_RPC_URL": "rpc_url",
            "DECLOUD_WS_URL": "ws_url",
            "DECLOUD_PROGRAM_ID": "program_id",
            "DECLOUD_PRIVATE_KEY": "private_key",
            "DECLOUD_DATA_DIR": "data_dir",
            "DECLOUD_IDL_PATH": "idl_path",
        }
        
        for env_var, attr in env_map.items():
            value = os.getenv(env_var)
            if value:
                setattr(config, attr, value)
        
        # Numeric values
        if os.getenv("DECLOUD_MIN_REWARD"):
            config.min_reward = float(os.getenv("DECLOUD_MIN_REWARD"))
        if os.getenv("DECLOUD_MAX_REWARD"):
            config.max_reward = float(os.getenv("DECLOUD_MAX_REWARD"))
        if os.getenv("DECLOUD_POLL_INTERVAL"):
            config.poll_interval = int(os.getenv("DECLOUD_POLL_INTERVAL"))
        
        # Boolean values
        if os.getenv("DECLOUD_DRY_RUN"):
            config.dry_run = os.getenv("DECLOUD_DRY_RUN").lower() in ("true", "1", "yes")
        if os.getenv("DECLOUD_AUTO_CLAIM"):
            config.auto_claim = os.getenv("DECLOUD_AUTO_CLAIM").lower() in ("true", "1", "yes")
        
        return config
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file"""
        save_path = path or self.config_path
        save_path = os.path.expanduser(save_path)
        
        # Create directory
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert to dict, remove None values
        data = {k: v for k, v in asdict(self).items() if v is not None}
        
        # Don't save infinity
        if data.get("max_reward") == float("inf"):
            del data["max_reward"]
        
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Config saved to {save_path}")
    
    def matches_round(self, round_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a round matches the configured filters.
        
        Args:
            round_data: Dict with reward_amount, dataset, creator
            
        Returns:
            (matches: bool, reason: str)
        """
        reward = round_data.get("reward_amount", 0)
        reward_sol = reward / 1e9 if reward > 1000 else reward  # Handle both lamports and SOL
        
        # Check reward range
        if reward_sol < self.min_reward:
            return False, f"reward {reward_sol:.4f} < min {self.min_reward}"
        
        if reward_sol > self.max_reward:
            return False, f"reward {reward_sol:.4f} > max {self.max_reward}"
        
        # Check dataset filter
        dataset = round_data.get("dataset", "")
        if self.allowed_datasets and dataset not in self.allowed_datasets:
            return False, f"dataset '{dataset}' not in allowed list"
        
        # Check blocked creators
        creator = round_data.get("creator", "")
        if self.blocked_creators and creator in self.blocked_creators:
            return False, f"creator '{creator[:16]}...' is blocked"
        
        return True, "matches"
    
    def print_config(self):
        """Print current configuration"""
        print("\n⚙️  DECLOUD Configuration")
        print("=" * 50)
        print(f"  Network:     {self.network}")
        print(f"  RPC URL:     {self.rpc_url}")
        print(f"  WS URL:      {self.ws_url}")
        print(f"  Program ID:  {self.program_id[:30]}...")
        print()
        print(f"  Data Dir:    {self.data_dir}")
        print(f"  IDL Path:    {self.idl_path}")
        print()
        print(f"  Min Reward:  {self.min_reward} SOL")
        print(f"  Max Reward:  {self.max_reward} SOL")
        print(f"  Datasets:    {self.allowed_datasets or 'all'}")
        print()
        print(f"  Auto Claim:     {self.auto_claim}")
        print(f"  Auto Start:     {self.auto_start}")
        print(f"  Auto Validate:  {self.auto_validate}")
        print(f"  Dry Run:        {self.dry_run}")
        print()
        print(f"  Poll Interval:  {self.poll_interval}s")
        print(f"  Use WebSocket:  {self.use_websocket}")
        print("=" * 50)
    
    def set_network(self, network: str):
        """Switch to a different network"""
        if network not in NETWORKS:
            raise ValueError(f"Unknown network: {network}. Available: {list(NETWORKS.keys())}")
        
        self.network = network
        self.rpc_url = NETWORKS[network]["rpc_url"]
        self.ws_url = NETWORKS[network]["ws_url"]