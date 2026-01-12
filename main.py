#!/usr/bin/env python3
"""
Decloud Validator CLI
Real-time WebSocket validation for federated learning rounds
"""
import sys
import asyncio
import getpass
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm

from config import config, DATASETS, RPC_ENDPOINTS
from dataset_manager import dataset_manager, DATASET_CONFIGS
from validator import Validator

console = Console()


def get_validator() -> Validator:
    """Get or create validator instance"""
    if not config.private_key:
        console.print("[red]No private key configured. Run 'decloud-validator setup' first.[/red]")
        sys.exit(1)
    return Validator(config.private_key)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Decloud Validator CLI
    
    Real-time WebSocket validation for federated learning on Solana.
    """
    pass


# ═══════════════════════════════════════════════════════════════
# Setup Commands
# ═══════════════════════════════════════════════════════════════

@cli.command()
def setup():
    """Interactive setup wizard"""
    console.print("\n[bold cyan]=== Decloud Validator Setup ===[/bold cyan]\n")
    
    console.print("[yellow]Enter your Solana wallet private key (base58 encoded)[/yellow]")
    console.print("[dim]This will be stored locally in ~/.decloud-validator/config.json[/dim]")
    
    private_key = getpass.getpass("Private Key: ")
    
    if not private_key:
        console.print("[red]Private key is required[/red]")
        return
    
    try:
        from solana_client import SolanaClient
        client = SolanaClient.from_private_key(private_key)
        console.print(f"[green]Wallet: {client.pubkey}[/green]")
    except Exception as e:
        console.print(f"[red]Invalid private key: {e}[/red]")
        return
    
    console.print("\n[yellow]Select network:[/yellow]")
    for i, net in enumerate(["devnet", "mainnet", "testnet"], 1):
        console.print(f"  {i}. {net}")
    
    network_choice = Prompt.ask("Network", choices=["1", "2", "3"], default="1")
    network = ["devnet", "mainnet", "testnet"][int(network_choice) - 1]
    
    config.private_key = private_key
    config.network = network
    config.save()
    
    console.print(f"\n[green]Configuration saved![/green]")
    console.print(f"[dim]Network: {network}[/dim]")
    
    try:
        balance = client.get_balance() / 1e9
        console.print(f"[dim]Balance: {balance:.4f} SOL[/dim]")
        if balance < 0.01:
            console.print("[yellow]Low balance! You need SOL for transaction fees.[/yellow]")
    except:
        console.print("[dim]Balance: (check failed)[/dim]")
    
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("  1. Install datasets: [bold]decloud-validator dataset install-all[/bold]")
    console.print("  2. Start validating: [bold]decloud-validator start[/bold]")


@cli.command()
@click.option("--network", "-n", type=click.Choice(["devnet", "mainnet", "testnet"]))
def network(network):
    """Change or show network"""
    if network:
        config.network = network
        config.save()
        console.print(f"[green]Network changed to {network}[/green]")
    else:
        console.print(f"Current network: [cyan]{config.network}[/cyan]")
        console.print(f"RPC: [dim]{config.rpc_url}[/dim]")


# ═══════════════════════════════════════════════════════════════
# Dataset Commands
# ═══════════════════════════════════════════════════════════════

@cli.group()
def dataset():
    """Dataset management commands"""
    pass


@dataset.command("list")
@click.option("--installed", "-i", is_flag=True, help="Show only installed datasets")
@click.option("--available", "-a", is_flag=True, help="Show all available datasets")
def dataset_list(installed, available):
    """List datasets"""
    if installed or not available:
        installed_list = dataset_manager.list_installed()
        if not installed_list:
            console.print("[yellow]No datasets installed[/yellow]")
            console.print("[dim]Run: decloud-validator dataset install-all[/dim]")
            return
        
        table = Table(title="Installed Datasets")
        table.add_column("Name", style="cyan")
        table.add_column("Classes", style="green")
        table.add_column("Task", style="yellow")
        
        for name in installed_list:
            cfg = DATASET_CONFIGS.get(name, {})
            table.add_row(name, str(cfg.get("num_classes", "?")), cfg.get("task", "?"))
        
        console.print(table)
    
    if available:
        console.print("\n[bold]All Available Datasets:[/bold]")
        installed_set = set(dataset_manager.list_installed())
        
        categories = {
            "Image Classification": ["Cifar10", "Cifar100", "Mnist", "FashionMnist", "Emnist", "Food101", "Flowers102"],
            "Text Classification": ["Imdb", "Sst2", "AgNews", "YelpReviews", "RottenTomatoes", "Banking77"],
            "Tabular": ["Iris", "Wine", "BreastCancer", "Diabetes", "Titanic"],
            "Medical": ["ChestXray", "SkinCancer", "Malaria", "BloodCells"],
        }
        
        for cat, datasets in categories.items():
            console.print(f"\n[cyan]{cat}:[/cyan]")
            for ds in datasets:
                status = "[green]v[/green]" if ds in installed_set else "[dim]o[/dim]"
                console.print(f"  {status} {ds}")


@dataset.command("install")
@click.argument("names", nargs=-1)
def dataset_install(names):
    """Install datasets (multiple allowed)"""
    if not names:
        console.print("[yellow]Usage: decloud-validator dataset install <name1> <name2> ...[/yellow]")
        return
    
    for name in names:
        if name not in DATASETS:
            console.print(f"[red]Unknown dataset: {name}[/red]")
            continue
        
        if dataset_manager.is_installed(name):
            console.print(f"[yellow]{name} already installed[/yellow]")
            continue
        
        console.print(f"[cyan]Installing {name}...[/cyan]")
        try:
            success = dataset_manager.install(name)
            if success:
                console.print(f"[green]v {name} installed[/green]")
            else:
                console.print(f"[red]x Failed to install {name}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@dataset.command("uninstall")
@click.argument("name")
def dataset_uninstall(name):
    """Uninstall a dataset"""
    if not dataset_manager.is_installed(name):
        console.print(f"[yellow]Dataset {name} is not installed[/yellow]")
        return
    
    if Confirm.ask(f"Remove dataset {name}?"):
        dataset_manager.uninstall(name)
        console.print(f"[green]v Dataset {name} removed[/green]")


@dataset.command("install-all")
@click.option("--category", "-c", help="Category: image, text, tabular, all")
def dataset_install_all(category):
    """Install multiple datasets"""
    categories = {
        "image": ["Cifar10", "Cifar100", "Mnist", "FashionMnist"],
        "text": ["Imdb", "Sst2", "AgNews", "RottenTomatoes"],
        "tabular": ["Iris", "Wine", "BreastCancer"],
        "all": ["Cifar10", "Cifar100", "Mnist", "FashionMnist", "Imdb", "Sst2", "AgNews", "Iris", "Wine"],
    }
    
    if category:
        if category not in categories:
            console.print(f"[red]Unknown category. Options: {list(categories.keys())}[/red]")
            return
        datasets = categories[category]
    else:
        datasets = ["Cifar10", "Mnist", "Imdb", "Iris"]
    
    for name in datasets:
        if not dataset_manager.is_installed(name):
            console.print(f"[cyan]Installing {name}...[/cyan]")
            try:
                dataset_manager.install(name)
                console.print(f"[green]v {name}[/green]")
            except Exception as e:
                console.print(f"[red]x {name}: {e}[/red]")
    
    console.print("\n[green]Done![/green]")


# ═══════════════════════════════════════════════════════════════
# Validation Commands
# ═══════════════════════════════════════════════════════════════

@cli.command()
def start():
    """Start validator (WebSocket real-time)"""
    validator = get_validator()
    
    if not config.installed_datasets:
        console.print("\n[yellow]No datasets installed![/yellow]")
        console.print("[dim]Run: decloud-validator dataset install-all[/dim]")
        return
    
    try:
        asyncio.run(validator.start())
    except KeyboardInterrupt:
        validator.stop()


@cli.command()
def status():
    """Show validator status"""
    validator = get_validator()
    validator.show_status()


@cli.command()
@click.option("--limit", "-l", default=10, help="Number of rounds to show")
def rounds(limit):
    """Show active rounds"""
    validator = get_validator()
    validator.show_rounds(limit=limit)


@cli.command()
@click.argument("round_id", type=int)
def prevalidate(round_id):
    """Manually prevalidate a round"""
    validator = get_validator()
    console.print(f"[cyan]Prevalidating round {round_id}...[/cyan]")
    result = asyncio.run(validator.prevalidate_round(round_id))
    
    if result.success:
        console.print(f"[green]v Success! Accuracy: {result.accuracy:.2f}%[/green]")
        console.print(f"[dim]TX: {result.tx_signature}[/dim]")
    else:
        console.print(f"[red]x Failed: {result.error}[/red]")


@cli.command()
@click.argument("round_id", type=int)
@click.argument("trainer")
def postvalidate(round_id, trainer):
    """Manually postvalidate a gradient"""
    validator = get_validator()
    console.print(f"[cyan]Postvalidating round {round_id}...[/cyan]")
    result = asyncio.run(validator.postvalidate_gradient(round_id, trainer))
    
    if result.success:
        console.print(f"[green]v Success! Accuracy: {result.accuracy:.2f}%[/green]")
        console.print(f"[dim]TX: {result.tx_signature}[/dim]")
    else:
        console.print(f"[red]x Failed: {result.error}[/red]")


# ═══════════════════════════════════════════════════════════════
# Rewards Commands
# ═══════════════════════════════════════════════════════════════

@cli.command("claim")
@click.argument("round_id", type=int)
def claim_reward(round_id):
    """Claim rewards from a finalized round"""
    validator = get_validator()
    console.print(f"[cyan]Claiming rewards from round {round_id}...[/cyan]")
    
    results = validator.claim_rewards(round_id)
    
    if "error" in results:
        console.print(f"[red]x {results['error']}[/red]")
        return
    
    if results["pre"]:
        if results["pre"]["success"]:
            console.print(f"[green]v Pre-validation reward claimed[/green]")
            console.print(f"[dim]TX: {results['pre']['tx']}[/dim]")
        else:
            console.print(f"[red]x Pre claim failed: {results['pre'].get('error')}[/red]")
    
    for post in results["post"]:
        if post["success"]:
            console.print(f"[green]v Post reward claimed[/green]")
        else:
            console.print(f"[red]x Post claim failed: {post.get('error')}[/red]")


@cli.command("balance")
def show_balance():
    """Show wallet balance"""
    validator = get_validator()
    try:
        balance = validator.get_balance()
        console.print(f"Balance: [green]{balance:.6f} SOL[/green]")
    except Exception as e:
        console.print(f"[red]Failed to get balance: {e}[/red]")


# ═══════════════════════════════════════════════════════════════
# Info Commands  
# ═══════════════════════════════════════════════════════════════

@cli.command("info")
@click.argument("round_id", type=int)
def round_info(round_id):
    """Show detailed round information"""
    validator = get_validator()
    
    info = validator.solana.get_round(round_id)
    if not info:
        console.print(f"[red]Round {round_id} not found[/red]")
        return
    
    table = Table(title=f"Round {round_id}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("ID", str(info.id))
    table.add_row("Creator", info.creator[:20] + "...")
    table.add_row("Model CID", info.model_cid[:30] + "..." if len(info.model_cid) > 30 else info.model_cid)
    table.add_row("Dataset", info.dataset)
    table.add_row("Reward", f"{info.reward_amount / 1e9:.4f} SOL")
    table.add_row("Status", info.status)
    table.add_row("Pre-validators", str(info.pre_count))
    table.add_row("Trainers", str(info.gradients_count))
    table.add_row("Total Validations", str(info.total_validations))
    
    if info.pre_count > 0:
        avg_pre = info.pre_accuracy_sum / info.pre_count / 100
        table.add_row("Avg Pre-accuracy", f"{avg_pre:.2f}%")
    
    console.print(table)
    
    has_pre = validator.solana.has_prevalidated(round_id)
    console.print(f"\nYour participation: {'[green]v Prevalidated[/green]' if has_pre else '[dim]Not prevalidated[/dim]'}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()