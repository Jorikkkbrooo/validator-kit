"""
Core validation logic for Decloud Validator
Event-driven WebSocket architecture
"""
import asyncio
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from solders.pubkey import Pubkey

from config import config, DATASETS
from dataset_manager import dataset_manager
from model_loader import model_loader, ModelPackage
from ipfs_client import ipfs_client
from solana_client import SolanaClient, RoundInfo, GradientInfo
from websocket_listener import (
    DecloudWebSocket,
    RoundCreatedEvent,
    GradientSubmittedEvent,
    PrevalidatedEvent,
    PostvalidatedEvent,
    RoundFinalizedEvent,
)

console = Console()


@dataclass
class ValidationResult:
    """Result of a validation"""
    round_id: int
    accuracy: float
    success: bool
    error: Optional[str] = None
    tx_signature: Optional[str] = None


@dataclass
class ValidatorStats:
    """Validator statistics"""
    prevalidations: int = 0
    postvalidations: int = 0
    errors: int = 0
    total_rounds_seen: int = 0
    uptime_start: float = 0


class Validator:
    """
    Decloud Validator - Event-driven validation
    """
    
    def __init__(self, private_key: str):
        self.solana = SolanaClient.from_private_key(private_key)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.running = False
        self.ws: Optional[DecloudWebSocket] = None
        
        # Stats
        self.stats = ValidatorStats()
        
        # Track what we've validated (in-memory cache)
        self.prevalidated_rounds: Set[int] = set()
        self.postvalidated: Dict[int, Set[str]] = defaultdict(set)
        
        # Pending work queues
        self.prevalidate_queue: asyncio.Queue = asyncio.Queue()
        self.postvalidate_queue: asyncio.Queue = asyncio.Queue()
        
        console.print(f"[green]✓ Validator initialized[/green]")
        console.print(f"[dim]  Wallet: {self.solana.pubkey}[/dim]")
        console.print(f"[dim]  Device: {self.device}[/dim]")
        console.print(f"[dim]  Network: {config.network}[/dim]")
    
    def get_balance(self) -> float:
        """Get SOL balance"""
        lamports = self.solana.get_balance()
        return lamports / 1e9
    
    # ═══════════════════════════════════════════════════════════════
    # Validation Logic
    # ═══════════════════════════════════════════════════════════════
    
    async def validate_model(
        self,
        model_cid: str,
        dataset_name: str,
        batch_size: int = 32,
        limit: Optional[int] = None,
    ) -> Tuple[float, int, int]:
        """
        Validate a model against a dataset
        Returns: (accuracy, correct_count, total_count)
        """
        console.print(f"[dim]  Downloading model {model_cid[:20]}...[/dim]")
        model_path = await ipfs_client.download_model_package(model_cid)
        if model_path is None:
            raise ValueError(f"Failed to download model: {model_cid}")
        
        console.print(f"[dim]  Loading model...[/dim]")
        model_pkg = model_loader.load_from_directory(model_path)
        model_pkg.to(self.device).eval()
        
        limit = limit or config.validation_batch_size
        console.print(f"[dim]  Loading test data ({dataset_name})...[/dim]")
        test_data, test_labels = dataset_manager.load_test_data(dataset_name, limit=limit)
        
        if model_pkg.embeddings is not None:
            embeddings = model_pkg.embeddings.numpy()
            if len(embeddings) != len(test_labels):
                raise ValueError(f"Embeddings count mismatch: {len(embeddings)} vs {len(test_labels)}")
        else:
            if isinstance(test_data, np.ndarray) and len(test_data.shape) == 2:
                embeddings = test_data
            else:
                raise ValueError("No embeddings in model package")
        
        console.print(f"[dim]  Running inference...[/dim]")
        predictions = model_pkg.predict_batch(embeddings, batch_size=batch_size)
        
        if len(predictions.shape) > 1:
            pred_labels = np.argmax(predictions, axis=1)
        else:
            pred_labels = (predictions > 0.5).astype(int)
        
        correct = (pred_labels == test_labels).sum()
        total = len(test_labels)
        accuracy = (correct / total) * 100
        
        return accuracy, int(correct), total
    
    # ═══════════════════════════════════════════════════════════════
    # Pre-validation
    # ═══════════════════════════════════════════════════════════════
    
    async def prevalidate_round(self, round_id: int) -> ValidationResult:
        """Pre-validate a round's base model"""
        
        if round_id in self.prevalidated_rounds:
            return ValidationResult(round_id, 0, False, "Already prevalidated (cached)")
        
        round_info = self.solana.get_round(round_id)
        if not round_info:
            return ValidationResult(round_id, 0, False, "Round not found")
        
        if round_info.status != "Active":
            return ValidationResult(round_id, 0, False, f"Round not active: {round_info.status}")
        
        if round_info.gradients_count > 0:
            return ValidationResult(round_id, 0, False, "Prevalidation closed")
        
        if not dataset_manager.is_installed(round_info.dataset):
            return ValidationResult(round_id, 0, False, f"Dataset not installed: {round_info.dataset}")
        
        if self.solana.has_prevalidated(round_id):
            self.prevalidated_rounds.add(round_id)
            return ValidationResult(round_id, 0, False, "Already prevalidated (chain)")
        
        try:
            console.print(f"[cyan]⚡ Prevalidating round {round_id} ({round_info.dataset})...[/cyan]")
            
            accuracy, correct, total = await self.validate_model(
                round_info.model_cid,
                round_info.dataset,
            )
            
            accuracy_bps = int(accuracy * 100)
            
            console.print(f"[dim]  Submitting to blockchain...[/dim]")
            tx = self.solana.prevalidate(round_id, accuracy_bps)
            
            self.prevalidated_rounds.add(round_id)
            self.stats.prevalidations += 1
            
            console.print(f"[green]✓ Prevalidated round {round_id}: {accuracy:.2f}%[/green]")
            console.print(f"[dim]  TX: {tx}[/dim]")
            
            return ValidationResult(round_id, accuracy, True, tx_signature=tx)
            
        except Exception as e:
            self.stats.errors += 1
            console.print(f"[red]✗ Prevalidation failed: {e}[/red]")
            return ValidationResult(round_id, 0, False, str(e))
    
    # ═══════════════════════════════════════════════════════════════
    # Post-validation
    # ═══════════════════════════════════════════════════════════════
    async def postvalidate_gradient(self, round_id: int, trainer_pubkey: str) -> ValidationResult:
        """Post-validate a gradient submission"""
        
        if trainer_pubkey in self.postvalidated.get(round_id, set()):
            return ValidationResult(round_id, 0, False, "Already postvalidated (cached)")
        
        trainer = Pubkey.from_string(trainer_pubkey)
        
        round_info = self.solana.get_round(round_id)
        if not round_info:
            return ValidationResult(round_id, 0, False, "Round not found")
        
        if round_info.status != "Active":
            return ValidationResult(round_id, 0, False, f"Round not active: {round_info.status}")
        
        if not dataset_manager.is_installed(round_info.dataset):
            return ValidationResult(round_id, 0, False, f"Dataset not installed: {round_info.dataset}")
        
        gradient_info = self.solana.get_gradient(round_id, trainer)
        if not gradient_info:
            return ValidationResult(round_id, 0, False, "Gradient not found")
        
        if self.solana.has_postvalidated(round_id, trainer):
            self.postvalidated[round_id].add(trainer_pubkey)
            return ValidationResult(round_id, 0, False, "Already postvalidated (chain)")
        
        try:
            console.print(f"[cyan]⚡ Postvalidating round {round_id} trainer {trainer_pubkey[:12]}...[/cyan]")
            
            # Download base model (for embeddings)
            console.print(f"[dim]  Downloading base model...[/dim]")
            base_path = await ipfs_client.download_model_package(round_info.model_cid)
            if not base_path:
                raise ValueError("Failed to download base model")
            base_pkg = model_loader.load_from_directory(base_path)
            
            # Download gradient (only head + config)
            console.print(f"[dim]  Downloading gradient...[/dim]")
            gradient_path = await ipfs_client.download_model_package(gradient_info.cid)
            if not gradient_path:
                raise ValueError("Failed to download gradient")
            gradient_pkg = model_loader.load_from_directory(gradient_path)
            gradient_pkg.to(self.device).eval()
            
            # Use embeddings from base model
            if base_pkg.embeddings is None:
                raise ValueError("No embeddings in base model")
            embeddings = base_pkg.embeddings.numpy()
            
            # Load test labels
            limit = config.validation_batch_size
            _, test_labels = dataset_manager.load_test_data(round_info.dataset, limit=limit)
            
            if len(embeddings) != len(test_labels):
                raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(test_labels)} labels")
            
            # Run inference with trained head
            console.print(f"[dim]  Running inference...[/dim]")
            predictions = gradient_pkg.predict_batch(embeddings, batch_size=32)
            
            if len(predictions.shape) > 1:
                pred_labels = np.argmax(predictions, axis=1)
            else:
                pred_labels = (predictions > 0.5).astype(int)
            
            correct = (pred_labels == test_labels).sum()
            total = len(test_labels)
            accuracy = (correct / total) * 100
            
            accuracy_bps = int(accuracy * 100)
            
            console.print(f"[dim]  Submitting to blockchain...[/dim]")
            tx = self.solana.postvalidate(round_id, trainer, accuracy_bps)
            
            self.postvalidated[round_id].add(trainer_pubkey)
            self.stats.postvalidations += 1
            
            console.print(f"[green]✓ Postvalidated round {round_id}: {accuracy:.2f}%[/green]")
            console.print(f"[dim]  TX: {tx}[/dim]")
            
            return ValidationResult(round_id, accuracy, True, tx_signature=tx)
            
        except Exception as e:
            self.stats.errors += 1
            console.print(f"[red]✗ Postvalidation failed: {e}[/red]")
            return ValidationResult(round_id, 0, False, str(e))
    
    # ═══════════════════════════════════════════════════════════════
    # Event Handlers
    # ═══════════════════════════════════════════════════════════════
    
    async def on_round_created(self, event: RoundCreatedEvent):
        """Handle new round created event"""
        self.stats.total_rounds_seen += 1
        
        console.print(f"\n[yellow]📢 New Round #{event.round_id}[/yellow]")
        console.print(f"[dim]   Dataset: {event.dataset}[/dim]")
        console.print(f"[dim]   Reward: {event.reward_amount / 1e9:.4f} SOL[/dim]")
        
        if dataset_manager.is_installed(event.dataset):
            await self.prevalidate_queue.put(event.round_id)
        else:
            console.print(f"[dim]   ⏭ Skipping (dataset not installed)[/dim]")
    
    async def on_gradient_submitted(self, event: GradientSubmittedEvent):
        """Handle gradient submitted event"""
        console.print(f"\n[yellow]📢 Gradient submitted to Round #{event.round_id}[/yellow]")
        console.print(f"[dim]   Trainer: {event.trainer[:16]}...[/dim]")
        
        round_info = self.solana.get_round(event.round_id)
        if round_info and dataset_manager.is_installed(round_info.dataset):
            await self.postvalidate_queue.put((event.round_id, event.trainer))
        else:
            console.print(f"[dim]   ⏭ Skipping[/dim]")
    
    async def on_prevalidated(self, event: PrevalidatedEvent):
        """Handle prevalidation event (from others)"""
        console.print(f"[dim]👁 Round #{event.round_id} prevalidated by {event.validator[:12]}... ({event.accuracy/100:.2f}%)[/dim]")
    
    async def on_postvalidated(self, event: PostvalidatedEvent):
        """Handle postvalidation event (from others)"""
        console.print(f"[dim]👁 Round #{event.round_id} postvalidated ({event.accuracy/100:.2f}%)[/dim]")
    
    async def on_round_finalized(self, event: RoundFinalizedEvent):
        """Handle round finalized event"""
        console.print(f"\n[green]🏁 Round #{event.round_id} finalized![/green]")
        
        if event.round_id in self.prevalidated_rounds:
            console.print(f"[yellow]   💰 You have rewards! Run: decloud-validator claim {event.round_id}[/yellow]")
    
    # ═══════════════════════════════════════════════════════════════
    # Worker Tasks
    # ═══════════════════════════════════════════════════════════════
    
    async def prevalidate_worker(self):
        """Worker that processes prevalidation queue"""
        while self.running:
            try:
                round_id = await asyncio.wait_for(self.prevalidate_queue.get(), timeout=1.0)
                await self.prevalidate_round(round_id)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                console.print(f"[red]Prevalidate worker error: {e}[/red]")
    
    async def postvalidate_worker(self):
        """Worker that processes postvalidation queue"""
        while self.running:
            try:
                item = await asyncio.wait_for(self.postvalidate_queue.get(), timeout=1.0)
                round_id, trainer = item
                await self.postvalidate_gradient(round_id, trainer)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                console.print(f"[red]Postvalidate worker error: {e}[/red]")
    
    # ═══════════════════════════════════════════════════════════════
    # Initial Scan
    # ═══════════════════════════════════════════════════════════════
    
    async def scan_existing_rounds(self):
        """Scan existing rounds on startup"""
        console.print("\n[cyan]🔍 Scanning existing rounds...[/cyan]")
        
        installed = set(config.installed_datasets)
        rounds = self.solana.get_active_rounds()
        
        console.print(f"[dim]  Found {len(rounds)} active rounds[/dim]")
        
        prevalidate_count = 0
        postvalidate_count = 0
        
        for round_info in rounds:
            if round_info.dataset not in installed:
                continue
            
            # Check prevalidation
            if round_info.gradients_count == 0:
                if not self.solana.has_prevalidated(round_info.id):
                    await self.prevalidate_queue.put(round_info.id)
                    prevalidate_count += 1
                else:
                    self.prevalidated_rounds.add(round_info.id)
            
            # Check postvalidation - scan all gradients for this round
            if round_info.gradients_count > 0:
                console.print(f"[dim]  Round {round_info.id}: {round_info.gradients_count} gradients, scanning...[/dim]")
                gradients = self.solana.get_all_gradients_for_round(round_info.id)
                console.print(f"[dim]  Found {len(gradients)} gradient accounts[/dim]")
                
                for gradient in gradients:
                    trainer_pubkey = gradient.trainer
                    
                    # Check if we already postvalidated this
                    if trainer_pubkey in self.postvalidated.get(round_info.id, set()):
                        continue
                    
                    # Check on chain
                    try:
                        trainer = Pubkey.from_string(trainer_pubkey)
                        if self.solana.has_postvalidated(round_info.id, trainer):
                            self.postvalidated[round_info.id].add(trainer_pubkey)
                            continue
                    except Exception as e:
                        console.print(f"[dim]  Error checking trainer {trainer_pubkey[:12]}: {e}[/dim]")
                        continue
                    
                    # Queue for postvalidation
                    await self.postvalidate_queue.put((round_info.id, trainer_pubkey))
                    postvalidate_count += 1
        
        console.print(f"[green]✓ Found {prevalidate_count} rounds to prevalidate[/green]")
        console.print(f"[green]✓ Found {postvalidate_count} gradients to postvalidate[/green]")
    
    # ═══════════════════════════════════════════════════════════════
    # Main Loop
    # ═══════════════════════════════════════════════════════════════
    
    async def start(self):
        """Start the validator"""
        self.running = True
        self.stats.uptime_start = time.time()
        
        self.show_status()
        
        if not config.installed_datasets:
            console.print("\n[red]✗ No datasets installed![/red]")
            console.print("[dim]Run: decloud-validator dataset install-all[/dim]")
            return
        
        console.print("\n[cyan]🔌 Connecting to Solana WebSocket...[/cyan]")
        self.ws = DecloudWebSocket()
        
        self.ws.on_round_created = self.on_round_created
        self.ws.on_gradient_submitted = self.on_gradient_submitted
        self.ws.on_prevalidated = self.on_prevalidated
        self.ws.on_postvalidated = self.on_postvalidated
        self.ws.on_round_finalized = self.on_round_finalized
        
        if not await self.ws.connect():
            console.print("[red]✗ Failed to connect WebSocket[/red]")
            return
        
        if not await self.ws.subscribe():
            console.print("[red]✗ Failed to subscribe[/red]")
            return
        
        await self.scan_existing_rounds()
        
        console.print("\n[green]🚀 Validator running! Listening for events...[/green]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
        
        workers = [
            asyncio.create_task(self.ws.listen()),
            asyncio.create_task(self.prevalidate_worker()),
            asyncio.create_task(self.postvalidate_worker()),
        ]
        
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            await self.ws.disconnect()
    
    def stop(self):
        """Stop the validator"""
        self.running = False
        console.print("\n[yellow]Stopping validator...[/yellow]")
    
    # ═══════════════════════════════════════════════════════════════
    # Rewards
    # ═══════════════════════════════════════════════════════════════
    
    def claim_rewards(self, round_id: int) -> Dict[str, Any]:
        """Claim all rewards for a round"""
        results = {"pre": None, "post": [], "total_claimed": 0}
        
        round_info = self.solana.get_round(round_id)
        if not round_info:
            return {"error": "Round not found"}
        
        if round_info.status != "Finalized":
            return {"error": f"Round not finalized: {round_info.status}"}
        
        pre = self.solana.get_pre_validation(round_id, self.solana.pubkey)
        if pre and not pre.reward_claimed:
            try:
                tx = self.solana.claim_validator_pre(round_id)
                results["pre"] = {"tx": tx, "success": True}
                console.print(f"[green]✓ Claimed pre-validation reward[/green]")
            except Exception as e:
                results["pre"] = {"error": str(e), "success": False}
        elif pre and pre.reward_claimed:
            results["pre"] = {"error": "Already claimed", "success": False}
        
        for trainer in self.postvalidated.get(round_id, set()):
            try:
                trainer_pk = Pubkey.from_string(trainer)
                post = self.solana.get_post_validation(round_id, trainer_pk, self.solana.pubkey)
                if post and not post.reward_claimed:
                    tx = self.solana.claim_validator_post(round_id, trainer_pk)
                    results["post"].append({"trainer": trainer, "tx": tx, "success": True})
                    console.print(f"[green]✓ Claimed post reward (trainer: {trainer[:12]}...)[/green]")
            except Exception as e:
                results["post"].append({"trainer": trainer, "error": str(e), "success": False})
        
        return results
    
    # ═══════════════════════════════════════════════════════════════
    # Status
    # ═══════════════════════════════════════════════════════════════
    
    def show_status(self):
        """Display validator status"""
        try:
            balance = self.get_balance()
        except:
            balance = -1
        
        installed = config.installed_datasets
        
        table = Table(title="🤖 Validator Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Wallet", str(self.solana.pubkey))
        table.add_row("Balance", f"{balance:.4f} SOL" if balance >= 0 else "Error")
        table.add_row("Network", config.network)
        table.add_row("Device", self.device)
        table.add_row("Datasets", str(len(installed)))
        table.add_row("Prevalidations", str(self.stats.prevalidations))
        table.add_row("Postvalidations", str(self.stats.postvalidations))
        
        console.print(table)
        
        if installed:
            console.print(f"\n[dim]Datasets: {', '.join(installed[:10])}{'...' if len(installed) > 10 else ''}[/dim]")
    
    def show_rounds(self, limit: int = 10):
        """Display active rounds"""
        rounds = self.solana.get_active_rounds()
        installed = set(config.installed_datasets)
        
        table = Table(title=f"Active Rounds ({len(rounds)} total)")
        table.add_column("ID", style="cyan")
        table.add_column("Dataset", style="yellow")
        table.add_column("Reward", style="green")
        table.add_column("Pre", style="blue")
        table.add_column("Gradients", style="magenta")
        table.add_column("Status", style="white")
        
        for round_info in rounds[:limit]:
            can_validate = round_info.dataset in installed
            has_prevalidated = round_info.id in self.prevalidated_rounds or self.solana.has_prevalidated(round_info.id)
            
            if has_prevalidated:
                status = "[green]✓ done[/green]"
            elif can_validate:
                status = "[yellow]⏳ pending[/yellow]"
            else:
                status = "[dim]⏭ skip[/dim]"
            
            table.add_row(
                str(round_info.id),
                round_info.dataset,
                f"{round_info.reward_amount / 1e9:.4f}",
                str(round_info.pre_count),
                str(round_info.gradients_count),
                status,
            )
        
        console.print(table)