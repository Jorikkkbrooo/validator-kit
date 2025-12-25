"""
DECLOUD Monitor
===============

Real-time monitoring of new rounds via WebSocket.
Listens to Solana program logs and parses events.
"""

import json
import time
import asyncio
import subprocess
from datetime import datetime
from typing import Callable, Optional, Dict, Any
from dataclasses import dataclass

from .config import Config


@dataclass
class RoundEvent:
    """Parsed round event from blockchain"""
    event_type: str  # "created", "claimed", "started", "completed", etc.
    round_id: int
    timestamp: datetime
    data: Dict[str, Any]


class Monitor:
    """
    Real-time monitor for DECLOUD rounds.
    
    Usage:
        monitor = Monitor(config)
        
        @monitor.on_new_round
        def handle_new(event):
            print(f"New round: {event.round_id}")
        
        monitor.start()
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._callbacks: Dict[str, list] = {
            "new_round": [],
            "round_claimed": [],
            "training_started": [],
            "submission": [],
            "completed": [],
            "any": [],
        }
        self._running = False
        self._ws_process = None
        self._known_rounds: set = set()
    
    def on(self, event_type: str, callback: Callable[[RoundEvent], None]):
        """Register callback for event type"""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
    
    def on_new_round(self, callback: Callable[[RoundEvent], None]):
        """Decorator for new round events"""
        self.on("new_round", callback)
        return callback
    
    def on_any(self, callback: Callable[[RoundEvent], None]):
        """Decorator for any event"""
        self.on("any", callback)
        return callback
    
    def _emit(self, event: RoundEvent):
        """Emit event to callbacks"""
        # Call specific callbacks
        for cb in self._callbacks.get(event.event_type, []):
            try:
                cb(event)
            except Exception as e:
                print(f"❌ Callback error: {e}")
        
        # Call 'any' callbacks
        for cb in self._callbacks.get("any", []):
            try:
                cb(event)
            except Exception as e:
                print(f"❌ Callback error: {e}")
    
    def _beep(self):
        """Make alert sound"""
        if self.config.sound_alerts:
            print("\a", end="", flush=True)
    
    def start_websocket(self):
        """Start WebSocket listener via Node.js"""
        script = f'''
const WebSocket = require("ws");

const ws = new WebSocket("{self.config.ws_url}");

ws.on("open", () => {{
    // Subscribe to program logs
    ws.send(JSON.stringify({{
        jsonrpc: "2.0",
        id: 1,
        method: "logsSubscribe",
        params: [
            {{ mentions: ["{self.config.program_id}"] }},
            {{ commitment: "confirmed" }}
        ]
    }}));
    console.log(JSON.stringify({{ type: "connected" }}));
}});

ws.on("message", (data) => {{
    try {{
        const msg = JSON.parse(data);
        if (msg.method === "logsNotification") {{
            const logs = msg.params.result.value.logs || [];
            const signature = msg.params.result.value.signature;
            
            // Parse logs for events
            for (const log of logs) {{
                if (log.includes("Program log: ")) {{
                    const logData = log.replace("Program log: ", "");
                    console.log(JSON.stringify({{
                        type: "log",
                        signature,
                        data: logData
                    }}));
                }}
            }}
        }}
    }} catch (e) {{}}
}});

ws.on("error", (err) => {{
    console.log(JSON.stringify({{ type: "error", message: err.message }}));
}});

ws.on("close", () => {{
    console.log(JSON.stringify({{ type: "disconnected" }}));
    process.exit(0);
}});

// Keep alive
setInterval(() => {{}}, 1000);
'''
        
        self._ws_process = subprocess.Popen(
            ["node", "-e", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        return self._ws_process
    
    def _parse_log(self, log_data: str) -> Optional[RoundEvent]:
        """
        Parse program log into event.
        
        Program msg! formats from lib.rs:
        - "Round {id} created. Model: {cid}, Dataset: {dataset:?}, Reward: {lamports} lamports"
        - "Round {id} claimed by validator {pubkey}. Join deadline: {ts}"
        - "Training started for round {id}. Baseline: {score}, Submit deadline: {ts}"
        - "Trainer {pubkey} submitted gradients for round {id}. CID: {cid}"
        - "Round {id} validation completed. Rewards ready to claim."
        - "Trainer {pubkey} joined round {id}. Total trainers: {n}"
        """
        import re
        now = datetime.now()
        
        # ═══════════════════════════════════════════════════════════════
        # Round created
        # Format: "Round 5 created. Model: Qm..., Dataset: Cifar10, Reward: 10000000 lamports"
        # ═══════════════════════════════════════════════════════════════
        match = re.search(r'Round (\d+) created\. Model: ([^,]+), Dataset: ([^,]+), Reward: (\d+)', log_data)
        if match:
            round_id = int(match.group(1))
            model_cid = match.group(2).strip()
            dataset = match.group(3).strip()
            reward_lamports = int(match.group(4))
            
            if round_id in self._known_rounds:
                return None
            self._known_rounds.add(round_id)
            
            return RoundEvent(
                event_type="new_round",
                round_id=round_id,
                timestamp=now,
                data={
                    "model_cid": model_cid,
                    "dataset": dataset,
                    "reward": reward_lamports,
                    "reward_sol": reward_lamports / 1e9,
                }
            )
        
        # ═══════════════════════════════════════════════════════════════
        # Round claimed by validator
        # Format: "Round 5 claimed by validator ABC123. Join deadline: 123456"
        # ═══════════════════════════════════════════════════════════════
        match = re.search(r'Round (\d+) claimed by validator ([^.]+)\. Join deadline: (\d+)', log_data)
        if match:
            return RoundEvent(
                event_type="round_claimed",
                round_id=int(match.group(1)),
                timestamp=now,
                data={
                    "validator": match.group(2).strip(),
                    "join_deadline": int(match.group(3)),
                }
            )
        
        # ═══════════════════════════════════════════════════════════════
        # Trainer joined
        # Format: "Trainer ABC joined round 5. Total trainers: 3"
        # ═══════════════════════════════════════════════════════════════
        match = re.search(r'Trainer ([^\s]+) joined round (\d+)\. Total trainers: (\d+)', log_data)
        if match:
            return RoundEvent(
                event_type="trainer_joined",
                round_id=int(match.group(2)),
                timestamp=now,
                data={
                    "trainer": match.group(1),
                    "trainers_count": int(match.group(3)),
                }
            )
        
        # ═══════════════════════════════════════════════════════════════
        # Training started
        # Format: "Training started for round 5. Baseline: 100, Submit deadline: 123456"
        # ═══════════════════════════════════════════════════════════════
        match = re.search(r'Training started for round (\d+)\. Baseline: (\d+), Submit deadline: (\d+)', log_data)
        if match:
            return RoundEvent(
                event_type="training_started",
                round_id=int(match.group(1)),
                timestamp=now,
                data={
                    "baseline_score": int(match.group(2)),
                    "submit_deadline": int(match.group(3)),
                }
            )
        
        # ═══════════════════════════════════════════════════════════════
        # Gradients submitted
        # Format: "Trainer ABC submitted gradients for round 5. CID: Qm..."
        # ═══════════════════════════════════════════════════════════════
        match = re.search(r'Trainer ([^\s]+) submitted gradients for round (\d+)\. CID: ([^\s]+)', log_data)
        if match:
            return RoundEvent(
                event_type="submission",
                round_id=int(match.group(2)),
                timestamp=now,
                data={
                    "trainer": match.group(1),
                    "gradient_cid": match.group(3),
                }
            )
        
        # ═══════════════════════════════════════════════════════════════
        # All trainers submitted - moving to validation
        # Format: "All 3 trainers submitted. Moving to validation."
        # ═══════════════════════════════════════════════════════════════
        match = re.search(r'All (\d+) trainers submitted\. Moving to validation', log_data)
        if match:
            return RoundEvent(
                event_type="validating",
                round_id=0,  # Not in log
                timestamp=now,
                data={"trainers_count": int(match.group(1))}
            )
        
        # ═══════════════════════════════════════════════════════════════
        # Validation completed
        # Format: "Round 5 validation completed. Rewards ready to claim."
        # ═══════════════════════════════════════════════════════════════
        match = re.search(r'Round (\d+) validation completed', log_data)
        if match:
            return RoundEvent(
                event_type="completed",
                round_id=int(match.group(1)),
                timestamp=now,
                data={}
            )
        
        # ═══════════════════════════════════════════════════════════════
        # Validator fee claimed
        # Format: "Validator ABC claimed 1000000 lamports fee from round 5"
        # ═══════════════════════════════════════════════════════════════
        match = re.search(r'Validator ([^\s]+) claimed (\d+) lamports fee from round (\d+)', log_data)
        if match:
            return RoundEvent(
                event_type="fee_claimed",
                round_id=int(match.group(3)),
                timestamp=now,
                data={
                    "validator": match.group(1),
                    "amount": int(match.group(2)),
                }
            )
        
        # ═══════════════════════════════════════════════════════════════
        # Trainer reward claimed
        # Format: "Trainer ABC claimed 900000 lamports from round 5"
        # ═══════════════════════════════════════════════════════════════
        match = re.search(r'Trainer ([^\s]+) claimed (\d+) lamports from round (\d+)', log_data)
        if match:
            return RoundEvent(
                event_type="reward_claimed",
                round_id=int(match.group(3)),
                timestamp=now,
                data={
                    "trainer": match.group(1),
                    "amount": int(match.group(2)),
                }
            )
        
        # ═══════════════════════════════════════════════════════════════
        # Round expired
        # Format: "Round 5 expired (no validator). Refunded 10000000 to creator"
        # ═══════════════════════════════════════════════════════════════
        match = re.search(r'Round (\d+) expired \(([^)]+)\)\. Refunded (\d+)', log_data)
        if match:
            return RoundEvent(
                event_type="expired",
                round_id=int(match.group(1)),
                timestamp=now,
                data={
                    "reason": match.group(2),
                    "refund": int(match.group(3)),
                }
            )
        
        # ═══════════════════════════════════════════════════════════════
        # Round cancelled
        # Format: "Round 5 cancelled. Refunded 10000000 lamports to creator"
        # ═══════════════════════════════════════════════════════════════
        match = re.search(r'Round (\d+) cancelled\. Refunded (\d+)', log_data)
        if match:
            return RoundEvent(
                event_type="cancelled",
                round_id=int(match.group(1)),
                timestamp=now,
                data={"refund": int(match.group(2))}
            )
        
        return None
    
    def start(self, callback: Optional[Callable[[RoundEvent], None]] = None):
        """
        Start monitoring (blocking).
        
        Args:
            callback: Optional callback for all events
        """
        if callback:
            self.on("any", callback)
        
        print(f"\n{'═'*60}")
        print("  🔭 DECLOUD Monitor")
        print(f"{'═'*60}")
        print(f"  Network:  {self.config.network}")
        print(f"  Program:  {self.config.program_id[:30]}...")
        print(f"  Mode:     {'WebSocket' if self.config.use_websocket else 'Polling'}")
        print(f"{'═'*60}")
        print(f"  Listening for events... (Ctrl+C to stop)\n")
        
        self._running = True
        
        if self.config.use_websocket:
            self._run_websocket()
        else:
            self._run_polling()
    
    def _run_websocket(self):
        """Run WebSocket-based monitoring"""
        try:
            process = self.start_websocket()
            
            while self._running:
                line = process.stdout.readline()
                if not line:
                    break
                
                try:
                    msg = json.loads(line.strip())
                    
                    if msg.get("type") == "connected":
                        print("✓ WebSocket connected")
                    
                    elif msg.get("type") == "log":
                        log_data = msg.get("data", "")
                        event = self._parse_log(log_data)
                        
                        if event:
                            self._print_event(event)
                            self._beep()
                            self._emit(event)
                    
                    elif msg.get("type") == "error":
                        print(f"❌ WS Error: {msg.get('message')}")
                    
                    elif msg.get("type") == "disconnected":
                        print("⚠️  WebSocket disconnected, reconnecting...")
                        time.sleep(2)
                        process = self.start_websocket()
                        
                except json.JSONDecodeError:
                    pass
                    
        except KeyboardInterrupt:
            print("\n\n👋 Monitor stopped")
        finally:
            if self._ws_process:
                self._ws_process.terminate()
    
    def _run_polling(self):
        """Run polling-based monitoring"""
        from .validator import Validator
        
        validator = Validator(self.config)
        last_rounds = set()
        
        try:
            while self._running:
                try:
                    rounds = validator.get_all_rounds()
                    current_ids = {r.id for r in rounds}
                    
                    # Check for new rounds
                    new_ids = current_ids - last_rounds
                    for round_id in new_ids:
                        round_data = next((r for r in rounds if r.id == round_id), None)
                        if round_data:
                            event = RoundEvent(
                                event_type="new_round",
                                round_id=round_id,
                                timestamp=datetime.now(),
                                data={
                                    "dataset": round_data.dataset,
                                    "reward": round_data.reward_amount,
                                    "creator": round_data.creator,
                                    "status": round_data.status,
                                }
                            )
                            self._print_event(event)
                            self._beep()
                            self._emit(event)
                    
                    last_rounds = current_ids
                    
                except Exception as e:
                    print(f"❌ Poll error: {e}")
                
                time.sleep(self.config.poll_interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitor stopped")
    
    def _print_event(self, event: RoundEvent):
        """Print event in nice format"""
        ts = event.timestamp.strftime("%H:%M:%S")
        
        icons = {
            "new_round": "🆕",
            "round_claimed": "✋",
            "trainer_joined": "👤",
            "training_started": "🚀",
            "submission": "📤",
            "validating": "🔍",
            "completed": "✅",
            "fee_claimed": "💰",
            "reward_claimed": "🎁",
            "expired": "⏰",
            "cancelled": "❌",
        }
        icon = icons.get(event.event_type, "📌")
        data = event.data
        
        if event.event_type == "new_round":
            reward_sol = data.get("reward_sol", data.get("reward", 0) / 1e9)
            dataset = data.get("dataset", "?")
            print(f"\n[{ts}] {icon} NEW ROUND #{event.round_id}")
            print(f"         Dataset: {dataset}")
            print(f"         Reward:  {reward_sol:.4f} SOL")
        
        elif event.event_type == "round_claimed":
            validator = data.get("validator", "?")[:20]
            print(f"[{ts}] {icon} Round #{event.round_id} claimed by {validator}...")
        
        elif event.event_type == "trainer_joined":
            count = data.get("trainers_count", "?")
            print(f"[{ts}] {icon} Trainer joined round #{event.round_id} (total: {count})")
        
        elif event.event_type == "training_started":
            baseline = data.get("baseline_score", "?")
            print(f"[{ts}] {icon} Training started for round #{event.round_id} (baseline: {baseline})")
        
        elif event.event_type == "submission":
            trainer = data.get("trainer", "?")[:20]
            print(f"[{ts}] {icon} Submission from {trainer}... for round #{event.round_id}")
        
        elif event.event_type == "validating":
            count = data.get("trainers_count", "?")
            print(f"[{ts}] {icon} All {count} trainers submitted. Validating...")
        
        elif event.event_type == "completed":
            print(f"[{ts}] {icon} Round #{event.round_id} COMPLETED! Rewards ready.")
        
        elif event.event_type == "fee_claimed":
            amount = data.get("amount", 0) / 1e9
            print(f"[{ts}] {icon} Validator fee claimed: {amount:.4f} SOL (round #{event.round_id})")
        
        elif event.event_type == "reward_claimed":
            amount = data.get("amount", 0) / 1e9
            print(f"[{ts}] {icon} Trainer reward claimed: {amount:.4f} SOL (round #{event.round_id})")
        
        elif event.event_type == "expired":
            reason = data.get("reason", "?")
            print(f"[{ts}] {icon} Round #{event.round_id} expired ({reason})")
        
        elif event.event_type == "cancelled":
            print(f"[{ts}] {icon} Round #{event.round_id} cancelled")
        
        else:
            print(f"[{ts}] {icon} {event.event_type.upper()} - Round #{event.round_id}")
    
    def stop(self):
        """Stop monitoring"""
        self._running = False
        if self._ws_process:
            self._ws_process.terminate()


class LiveValidator:
    """
    Validator with live WebSocket monitoring.
    Combines Monitor + Validator for real-time auto-claiming.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.monitor = Monitor(config)
        
        # Import here to avoid circular
        from .validator import Validator
        self.validator = Validator(config)
        
        self._active_rounds: set = set()
    
    def start(self, private_key: Optional[str] = None):
        """Start live validator"""
        if private_key:
            self.validator.login(private_key)
        elif self.config.private_key:
            self.validator.login(self.config.private_key)
        else:
            raise RuntimeError("No private key provided")
        
        print(f"\n{'═'*60}")
        print("  🌐 DECLOUD Live Validator")
        print(f"{'═'*60}")
        print(f"  Validator: {self.validator._public_key[:30]}...")
        print(f"  Network:   {self.config.network}")
        print(f"  Mode:      {'Dry Run (monitor only)' if self.config.dry_run else 'Live'}")
        print(f"\n  Filters:")
        print(f"    Min Reward: {self.config.min_reward} SOL")
        print(f"    Max Reward: {self.config.max_reward} SOL")
        print(f"    Datasets:   {self.config.allowed_datasets or 'all'}")
        print(f"{'═'*60}")
        print(f"  Listening... (Ctrl+C to stop)\n")
        
        # Register handlers
        @self.monitor.on_new_round
        def on_new(event: RoundEvent):
            self._handle_new_round(event)
        
        # Start monitoring
        self.monitor.start()
    
    def _handle_new_round(self, event: RoundEvent):
        """Handle new round event"""
        round_id = event.round_id
        data = event.data
        
        # Check filters
        matches, reason = self.config.matches_round({
            "reward_amount": data.get("reward", 0),
            "dataset": data.get("dataset", ""),
            "creator": data.get("creator", ""),
        })
        
        if not matches:
            print(f"   ⏭️  Skipping: {reason}")
            return
        
        # Check dataset downloaded
        if self.config.only_downloaded:
            dataset = data.get("dataset", "")
            if not self.validator.datasets.is_downloaded(dataset):
                print(f"   ⏭️  Skipping: dataset {dataset} not downloaded")
                return
        
        # Check concurrent limit
        if len(self._active_rounds) >= self.config.max_concurrent_rounds:
            print(f"   ⏭️  Skipping: max concurrent rounds ({self.config.max_concurrent_rounds})")
            return
        
        # Dry run mode
        if self.config.dry_run:
            print(f"   👁️  [DRY RUN] Would claim round #{round_id}")
            return
        
        # Claim!
        if self.config.auto_claim:
            try:
                time.sleep(self.config.claim_delay)  # Small delay
                tx = self.validator.claim_round(round_id)
                print(f"   ✅ Claimed round #{round_id}!")
                print(f"      TX: {tx[:40]}...")
                self._active_rounds.add(round_id)
            except Exception as e:
                print(f"   ❌ Failed to claim: {e}")


def print_live_stats(rounds: list, config: Config):
    """Print live dashboard"""
    from datetime import datetime
    
    now = datetime.now().strftime("%H:%M:%S")
    
    # Count by status
    by_status = {}
    for r in rounds:
        status = r.status
        by_status[status] = by_status.get(status, 0) + 1
    
    # Matching rounds
    matching = [r for r in rounds if config.matches_round({
        "reward_amount": r.reward_amount,
        "dataset": r.dataset,
        "creator": r.creator,
    })[0]]
    
    print(f"\r[{now}] Rounds: {len(rounds)} | "
          f"Available: {by_status.get('waitingValidator', 0)} | "
          f"Matching: {len(matching)}", end="", flush=True)