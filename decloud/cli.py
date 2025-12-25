#!/usr/bin/env python3
"""
DECLOUD Validator CLI
=====================

Command-line interface for DECLOUD validators.

Usage:
    decloud login <private_key>
    decloud datasets download
    decloud datasets list
    decloud validate start
    decloud rounds list
    decloud config show
"""

import os
import sys
import argparse
from typing import Optional

from .config import Config
from .dataset_manager import DatasetManager, DatasetCategory
from .validator import Validator


def print_banner():
    """Print DECLOUD banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     ██████╗ ███████╗ ██████╗██╗      ██████╗ ██╗   ██╗   ║
    ║     ██╔══██╗██╔════╝██╔════╝██║     ██╔═══██╗██║   ██║   ║
    ║     ██║  ██║█████╗  ██║     ██║     ██║   ██║██║   ██║   ║
    ║     ██║  ██║██╔══╝  ██║     ██║     ██║   ██║██║   ██║   ║
    ║     ██████╔╝███████╗╚██████╗███████╗╚██████╔╝╚██████╔╝   ║
    ║     ╚═════╝ ╚══════╝ ╚═════╝╚══════╝ ╚═════╝  ╚═════╝    ║
    ║                                                           ║
    ║           Decentralized Cloud for AI Training             ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def cmd_login(args):
    """Login command"""
    config = Config.load()
    validator = Validator(config)
    
    if args.keyfile:
        success = validator.login_from_file(args.keyfile)
    else:
        success = validator.login(args.private_key)
    
    if success:
        # Save private key to config (optional)
        if args.save:
            config.private_key = args.private_key
            config.save()
            print("✓ Credentials saved to config")
    
    return 0 if success else 1


def cmd_datasets_download(args):
    """Download datasets"""
    config = Config.load()
    manager = DatasetManager(config.data_dir)
    
    if args.minimal:
        print("\n📦 Downloading minimal dataset pack...")
        manager.download_minimal()
    elif args.category:
        try:
            cat = DatasetCategory(args.category)
            print(f"\n📦 Downloading {args.category} datasets...")
            manager.download_category(cat)
        except ValueError:
            print(f"❌ Unknown category: {args.category}")
            print("Available categories:")
            for c in DatasetCategory:
                print(f"  - {c.value}")
            return 1
    elif args.dataset:
        print(f"\n📦 Downloading {args.dataset}...")
        manager.download(args.dataset)
    else:
        print("\n📦 Downloading all datasets...")
        manager.download_all(skip_large=not args.include_large)
    
    return 0


def cmd_datasets_list(args):
    """List datasets"""
    config = Config.load()
    manager = DatasetManager(config.data_dir)
    
    if args.status:
        manager.print_status()
    else:
        categories = manager.list_categories()
        print(f"\n📊 DECLOUD Supported Datasets")
        print("=" * 50)
        
        for cat, datasets in categories.items():
            print(f"\n{cat} ({len(datasets)} datasets):")
            for name in datasets:
                print(f"  - {name}")
        
        print(f"\nTotal: {len(manager.list_datasets())} datasets")
        print(f"Size estimate: {manager.estimate_total_size()}")
    
    return 0


def cmd_validate_start(args):
    """Start validator with WebSocket monitoring"""
    config = Config.from_env()
    
    # Apply CLI args
    if args.min_reward:
        config.min_reward = args.min_reward
    if args.max_reward:
        config.max_reward = args.max_reward
    if args.dataset:
        config.allowed_datasets = [args.dataset]
    if args.no_websocket:
        config.use_websocket = False
    if args.dry_run:
        config.dry_run = True
    if args.no_auto_claim:
        config.auto_claim = False
    
    # Get private key
    private_key = args.private_key or config.private_key or os.getenv("DECLOUD_PRIVATE_KEY")
    
    if not private_key:
        print("❌ No private key provided")
        print("   Use: decloud validate start --private-key <key>")
        print("   Or:  export DECLOUD_PRIVATE_KEY=<key>")
        return 1
    
    validator = Validator(config)
    
    if not validator.login(private_key):
        return 1
    
    validator.start()
    
    return 0


def cmd_rounds_list(args):
    """List rounds"""
    config = Config.load()
    validator = Validator(config)
    
    rounds = validator.get_all_rounds()
    
    print(f"\n📋 DECLOUD Rounds")
    print("=" * 90)
    print(f"{'ID':<5} {'Status':<18} {'Dataset':<15} {'Reward':<12} {'Trainers':<10} {'Ready':<6}")
    print("-" * 90)
    
    for r in sorted(rounds, key=lambda x: x.id, reverse=True):
        reward = f"{r.reward_amount / 1e9:.4f} SOL"
        trainers = f"{r.submissions_count}/{r.trainers_count}"
        
        # Check if dataset is downloaded
        has_dataset = validator.datasets.is_downloaded(r.dataset)
        ready = "✓" if has_dataset else "○"
        
        print(f"{r.id:<5} {r.status:<18} {r.dataset:<15} {reward:<12} {trainers:<10} {ready:<6}")
    
    print("-" * 90)
    print(f"Total: {len(rounds)} rounds")
    
    # Show missing datasets hint
    missing = validator.get_missing_datasets()
    if missing:
        print(f"\n💡 Missing datasets for available rounds: {', '.join(missing[:5])}")
        print("   Run: decloud datasets download --dataset <name>")
    
    return 0


def cmd_config_show(args):
    """Show configuration"""
    config = Config.load()
    config.print_config()
    return 0


def cmd_config_set(args):
    """Set configuration value"""
    config = Config.load()
    
    key = args.key
    value = args.value
    
    if hasattr(config, key):
        # Type conversion
        field_type = type(getattr(config, key))
        if field_type == bool:
            value = value.lower() in ("true", "1", "yes")
        elif field_type == int:
            value = int(value)
        elif field_type == float:
            value = float(value)
        
        setattr(config, key, value)
        config.save()
        print(f"✓ Set {key} = {value}")
    else:
        print(f"❌ Unknown config key: {key}")
        return 1
    
    return 0


def cmd_info(args):
    """Show system info"""
    print_banner()
    
    config = Config.load()
    manager = DatasetManager(config.data_dir)
    
    # Check datasets
    downloaded = sum(1 for d in manager.list_datasets() if manager.is_downloaded(d))
    total = len(manager.list_datasets())
    
    print(f"  Network:   {config.network}")
    print(f"  RPC URL:   {config.rpc_url}")
    print(f"  Datasets:  {downloaded}/{total} downloaded")
    print(f"  Data Dir:  {config.data_dir}")
    print()
    
    return 0


def cmd_config_list(args):
    """List all config options"""
    config = Config.load()
    
    print("\n⚙️  Available Configuration Options:")
    print("=" * 60)
    
    fields = [
        ("Network", [
            ("network", "Network name (devnet/mainnet-beta)"),
            ("rpc_url", "RPC endpoint URL"),
            ("ws_url", "WebSocket endpoint URL"),
        ]),
        ("Filters", [
            ("min_reward", "Minimum reward in SOL"),
            ("max_reward", "Maximum reward in SOL"),
            ("only_downloaded", "Only claim if dataset downloaded"),
        ]),
        ("Automation", [
            ("auto_claim", "Auto-claim matching rounds"),
            ("auto_start", "Auto-start training"),
            ("auto_validate", "Auto-validate submissions"),
            ("max_concurrent_rounds", "Max concurrent rounds"),
            ("dry_run", "Monitor only, don't claim"),
        ]),
        ("Timing", [
            ("poll_interval", "Seconds between polls"),
            ("claim_delay", "Delay before claiming"),
            ("validation_timeout", "Validation timeout"),
        ]),
        ("Paths", [
            ("data_dir", "Dataset directory"),
            ("idl_path", "IDL file path"),
        ]),
    ]
    
    for section, items in fields:
        print(f"\n{section}:")
        for key, desc in items:
            value = getattr(config, key, "?")
            print(f"  {key:<25} = {str(value):<20} # {desc}")
    
    print("\n" + "=" * 60)
    print("Set with: decloud config set <key> <value>")
    
    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        prog="decloud",
        description="DECLOUD Validator CLI - Decentralized Cloud for AI Training"
    )
    parser.add_argument("--version", action="version", version="DECLOUD 0.1.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Login
    login_parser = subparsers.add_parser("login", help="Login with private key")
    login_parser.add_argument("private_key", nargs="?", help="Base58 private key")
    login_parser.add_argument("--keyfile", "-f", help="Path to keypair JSON file")
    login_parser.add_argument("--save", "-s", action="store_true", help="Save to config")
    login_parser.set_defaults(func=cmd_login)
    
    # Datasets
    datasets_parser = subparsers.add_parser("datasets", help="Manage datasets")
    datasets_sub = datasets_parser.add_subparsers(dest="datasets_cmd")
    
    # datasets download
    dl_parser = datasets_sub.add_parser("download", help="Download datasets")
    dl_parser.add_argument("--minimal", "-m", action="store_true", help="Download minimal pack")
    dl_parser.add_argument("--category", "-c", help="Download specific category")
    dl_parser.add_argument("--dataset", "-d", help="Download specific dataset")
    dl_parser.add_argument("--include-large", action="store_true", help="Include large datasets")
    dl_parser.set_defaults(func=cmd_datasets_download)
    
    # datasets list
    list_parser = datasets_sub.add_parser("list", help="List datasets")
    list_parser.add_argument("--status", "-s", action="store_true", help="Show download status")
    list_parser.set_defaults(func=cmd_datasets_list)
    
    # Validate
    validate_parser = subparsers.add_parser("validate", help="Validator operations")
    validate_sub = validate_parser.add_subparsers(dest="validate_cmd")
    
    # validate start
    start_parser = validate_sub.add_parser("start", help="Start validator with WebSocket monitoring")
    start_parser.add_argument("--private-key", "-k", help="Private key")
    start_parser.add_argument("--min-reward", type=float, help="Minimum reward filter (SOL)")
    start_parser.add_argument("--max-reward", type=float, help="Maximum reward filter (SOL)")
    start_parser.add_argument("--dataset", "-d", help="Filter by specific dataset")
    start_parser.add_argument("--dry-run", action="store_true", help="Monitor only, don't claim")
    start_parser.add_argument("--no-websocket", action="store_true", help="Use polling instead of WebSocket")
    start_parser.add_argument("--no-auto-claim", action="store_true", help="Disable auto-claiming")
    start_parser.set_defaults(func=cmd_validate_start)
    
    # Rounds
    rounds_parser = subparsers.add_parser("rounds", help="Round operations")
    rounds_sub = rounds_parser.add_subparsers(dest="rounds_cmd")
    
    # rounds list
    rlist_parser = rounds_sub.add_parser("list", help="List all rounds")
    rlist_parser.set_defaults(func=cmd_rounds_list)
    
    # Config
    config_parser = subparsers.add_parser("config", help="Configuration")
    config_sub = config_parser.add_subparsers(dest="config_cmd")
    
    # config show
    show_parser = config_sub.add_parser("show", help="Show configuration")
    show_parser.set_defaults(func=cmd_config_show)
    
    # config set
    set_parser = config_sub.add_parser("set", help="Set configuration value")
    set_parser.add_argument("key", help="Config key")
    set_parser.add_argument("value", help="Config value")
    set_parser.set_defaults(func=cmd_config_set)
    
    # config list (NEW!)
    clist_parser = config_sub.add_parser("list", help="List all config options")
    clist_parser.set_defaults(func=cmd_config_list)
    
    # Info
    info_parser = subparsers.add_parser("info", help="Show system info")
    info_parser.set_defaults(func=cmd_info)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        cmd_info(args)
        parser.print_help()
        return 0
    
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())