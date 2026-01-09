"""
WebSocket listener for Decloud events
Real-time subscription to program logs
"""
import asyncio
import json
import base64
import struct
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import websockets
from rich.console import Console

from config import PROGRAM_ID, config

console = Console()


# WebSocket endpoints
WS_ENDPOINTS = {
    "devnet": "wss://api.devnet.solana.com",
    "mainnet": "wss://api.mainnet-beta.solana.com",
    "testnet": "wss://api.testnet.solana.com",
}

# Alternative endpoints (more reliable)
WS_ENDPOINTS_ALT = {
    "devnet": "wss://devnet.helius-rpc.com/?api-key=1d8740dc-e5f4-421c-b823-e1bad1889eff",
}


class EventType(Enum):
    ROUND_CREATED = "RoundCreated"
    GRADIENT_SUBMITTED = "GradientSubmitted"
    PREVALIDATED = "Prevalidated"
    POSTVALIDATED = "Postvalidated"
    ROUND_FINALIZED = "RoundFinalized"
    ROUND_CANCELLED = "RoundCancelled"
    TRAINER_REWARD_CLAIMED = "TrainerRewardClaimed"
    VALIDATOR_REWARD_CLAIMED = "ValidatorRewardClaimed"


# Event discriminators (first 8 bytes of event data)
EVENT_DISCRIMINATORS = {
    bytes([16, 19, 68, 117, 87, 198, 7, 124]): EventType.ROUND_CREATED,
    bytes([165, 33, 17, 57, 235, 165, 150, 201]): EventType.GRADIENT_SUBMITTED,
    bytes([139, 133, 194, 202, 88, 229, 189, 30]): EventType.PREVALIDATED,
    bytes([189, 201, 251, 120, 36, 69, 198, 209]): EventType.POSTVALIDATED,
    bytes([43, 187, 17, 193, 36, 241, 48, 82]): EventType.ROUND_FINALIZED,
    bytes([238, 141, 105, 175, 182, 158, 15, 7]): EventType.ROUND_CANCELLED,
    bytes([21, 152, 206, 205, 60, 156, 85, 250]): EventType.TRAINER_REWARD_CLAIMED,
    bytes([178, 74, 55, 133, 68, 80, 223, 171]): EventType.VALIDATOR_REWARD_CLAIMED,
}


@dataclass
class RoundCreatedEvent:
    round_id: int
    creator: str
    dataset_id: int
    dataset: str
    reward_amount: int


@dataclass
class GradientSubmittedEvent:
    round_id: int
    trainer: str
    gradient_cid: str
    gradients_count: int


@dataclass 
class PrevalidatedEvent:
    round_id: int
    validator: str
    accuracy: int
    pre_count: int


@dataclass
class PostvalidatedEvent:
    round_id: int
    trainer: str
    validator: str
    accuracy: int
    post_count: int


@dataclass
class RoundFinalizedEvent:
    round_id: int


# Dataset mapping
from config import DATASET_ID_TO_NAME


class EventParser:
    """Parse events from program logs"""
    
    @staticmethod
    def parse_log_data(data_b64: str) -> Optional[Any]:
        """Parse base64 encoded event data"""
        try:
            data = base64.b64decode(data_b64)
            if len(data) < 8:
                return None
            
            discriminator = data[:8]
            event_type = EVENT_DISCRIMINATORS.get(bytes(discriminator))
            
            if not event_type:
                return None
            
            payload = data[8:]
            
            if event_type == EventType.ROUND_CREATED:
                return EventParser._parse_round_created(payload)
            elif event_type == EventType.GRADIENT_SUBMITTED:
                return EventParser._parse_gradient_submitted(payload)
            elif event_type == EventType.PREVALIDATED:
                return EventParser._parse_prevalidated(payload)
            elif event_type == EventType.POSTVALIDATED:
                return EventParser._parse_postvalidated(payload)
            elif event_type == EventType.ROUND_FINALIZED:
                return EventParser._parse_round_finalized(payload)
            
            return None
        except Exception as e:
            return None
    
    @staticmethod
    def _parse_round_created(data: bytes) -> Optional[RoundCreatedEvent]:
        try:
            offset = 0
            round_id = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            creator = base64.b64encode(data[offset:offset+32]).decode()
            offset += 32
            
            dataset_id = data[offset]
            offset += 1
            
            reward_amount = struct.unpack("<Q", data[offset:offset+8])[0]
            
            dataset = DATASET_ID_TO_NAME.get(dataset_id, f"Unknown({dataset_id})")
            
            return RoundCreatedEvent(
                round_id=round_id,
                creator=creator,
                dataset_id=dataset_id,
                dataset=dataset,
                reward_amount=reward_amount,
            )
        except:
            return None
    
    @staticmethod
    def _parse_gradient_submitted(data: bytes) -> Optional[GradientSubmittedEvent]:
        try:
            import base58
            offset = 0
            round_id = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            trainer = base58.b58encode(data[offset:offset+32]).decode()
            offset += 32
            
            # String: 4 bytes length + data
            cid_len = struct.unpack("<I", data[offset:offset+4])[0]
            offset += 4
            gradient_cid = data[offset:offset+cid_len].decode("utf-8")
            offset += cid_len
            
            gradients_count = data[offset]
            
            return GradientSubmittedEvent(
                round_id=round_id,
                trainer=trainer,
                gradient_cid=gradient_cid,
                gradients_count=gradients_count,
            )
        except:
            return None
    
    @staticmethod
    def _parse_prevalidated(data: bytes) -> Optional[PrevalidatedEvent]:
        try:
            offset = 0
            round_id = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            validator = base64.b64encode(data[offset:offset+32]).decode()
            offset += 32
            
            accuracy = struct.unpack("<I", data[offset:offset+4])[0]
            offset += 4
            
            pre_count = data[offset]
            
            return PrevalidatedEvent(
                round_id=round_id,
                validator=validator,
                accuracy=accuracy,
                pre_count=pre_count,
            )
        except:
            return None
    
    @staticmethod
    def _parse_postvalidated(data: bytes) -> Optional[PostvalidatedEvent]:
        try:
            offset = 0
            round_id = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            trainer = base64.b64encode(data[offset:offset+32]).decode()
            offset += 32
            
            validator = base64.b64encode(data[offset:offset+32]).decode()
            offset += 32
            
            accuracy = struct.unpack("<I", data[offset:offset+4])[0]
            offset += 4
            
            post_count = data[offset]
            
            return PostvalidatedEvent(
                round_id=round_id,
                trainer=trainer,
                validator=validator,
                accuracy=accuracy,
                post_count=post_count,
            )
        except:
            return None
    
    @staticmethod
    def _parse_round_finalized(data: bytes) -> Optional[RoundFinalizedEvent]:
        try:
            round_id = struct.unpack("<Q", data[:8])[0]
            return RoundFinalizedEvent(round_id=round_id)
        except:
            return None


class DecloudWebSocket:
    """
    WebSocket listener for Decloud program events
    """
    
    def __init__(self):
        self.ws_url = WS_ENDPOINTS.get(config.network, WS_ENDPOINTS["devnet"])
        self.program_id = PROGRAM_ID
        self.running = False
        self.subscription_id = None
        self.ws = None
        
        # Event handlers
        self.on_round_created: Optional[Callable] = None
        self.on_gradient_submitted: Optional[Callable] = None
        self.on_prevalidated: Optional[Callable] = None
        self.on_postvalidated: Optional[Callable] = None
        self.on_round_finalized: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
    
    async def connect(self) -> bool:
        """Connect to WebSocket"""
        try:
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=5,
            )
            console.print(f"[green]✓ WebSocket connected to {config.network}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]✗ WebSocket connection failed: {e}[/red]")
            # Try alternative endpoint
            alt_url = WS_ENDPOINTS_ALT.get(config.network)
            if alt_url:
                try:
                    self.ws = await websockets.connect(alt_url, ping_interval=30)
                    console.print(f"[green]✓ WebSocket connected (alt endpoint)[/green]")
                    return True
                except:
                    pass
            return False
    
    async def subscribe(self) -> bool:
        """Subscribe to program logs"""
        if not self.ws:
            return False
        
        subscribe_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "logsSubscribe",
            "params": [
                {"mentions": [self.program_id]},
                {"commitment": "confirmed"}
            ]
        }
        
        await self.ws.send(json.dumps(subscribe_msg))
        
        # Wait for subscription confirmation
        response = await self.ws.recv()
        data = json.loads(response)
        
        if "result" in data:
            self.subscription_id = data["result"]
            console.print(f"[green]✓ Subscribed to program logs (id: {self.subscription_id})[/green]")
            return True
        else:
            console.print(f"[red]✗ Subscription failed: {data}[/red]")
            return False
    
    async def unsubscribe(self):
        """Unsubscribe from logs"""
        if self.ws and self.subscription_id:
            unsubscribe_msg = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "logsUnsubscribe",
                "params": [self.subscription_id]
            }
            await self.ws.send(json.dumps(unsubscribe_msg))
            self.subscription_id = None
    
    async def disconnect(self):
        """Disconnect WebSocket"""
        self.running = False
        await self.unsubscribe()
        if self.ws:
            await self.ws.close()
            self.ws = None
    
    def _parse_notification(self, msg: Dict) -> Optional[Any]:
        """Parse notification message and extract event"""
        try:
            if msg.get("method") != "logsNotification":
                return None
            
            params = msg.get("params", {})
            result = params.get("result", {})
            value = result.get("value", {})
            logs = value.get("logs", [])
            
            # Look for Program data in logs
            for log in logs:
                if log.startswith("Program data: "):
                    data_b64 = log[14:]  # Remove "Program data: " prefix
                    event = EventParser.parse_log_data(data_b64)
                    if event:
                        return event
            
            return None
        except Exception as e:
            return None
    
    async def _handle_event(self, event: Any):
        """Route event to appropriate handler"""
        try:
            if isinstance(event, RoundCreatedEvent):
                if self.on_round_created:
                    await self._call_handler(self.on_round_created, event)
            
            elif isinstance(event, GradientSubmittedEvent):
                if self.on_gradient_submitted:
                    await self._call_handler(self.on_gradient_submitted, event)
            
            elif isinstance(event, PrevalidatedEvent):
                if self.on_prevalidated:
                    await self._call_handler(self.on_prevalidated, event)
            
            elif isinstance(event, PostvalidatedEvent):
                if self.on_postvalidated:
                    await self._call_handler(self.on_postvalidated, event)
            
            elif isinstance(event, RoundFinalizedEvent):
                if self.on_round_finalized:
                    await self._call_handler(self.on_round_finalized, event)
        
        except Exception as e:
            console.print(f"[red]Event handler error: {e}[/red]")
            if self.on_error:
                await self._call_handler(self.on_error, e)
    
    async def _call_handler(self, handler: Callable, *args):
        """Call handler (sync or async)"""
        if asyncio.iscoroutinefunction(handler):
            await handler(*args)
        else:
            handler(*args)
    
    async def listen(self):
        """Main listen loop"""
        self.running = True
        
        while self.running:
            try:
                if not self.ws:
                    if not await self.connect():
                        await asyncio.sleep(5)
                        continue
                    
                    if not await self.subscribe():
                        await asyncio.sleep(5)
                        continue
                
                # Receive message
                try:
                    msg_raw = await asyncio.wait_for(self.ws.recv(), timeout=60)
                    msg = json.loads(msg_raw)
                    
                    event = self._parse_notification(msg)
                    if event:
                        await self._handle_event(event)
                
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    continue
                
            except websockets.exceptions.ConnectionClosed:
                console.print("[yellow]WebSocket disconnected, reconnecting...[/yellow]")
                self.ws = None
                self.subscription_id = None
                await asyncio.sleep(2)
            
            except Exception as e:
                console.print(f"[red]WebSocket error: {e}[/red]")
                await asyncio.sleep(5)
    
    async def start(self):
        """Start listening in background"""
        await self.listen()
    
    def stop(self):
        """Stop listening"""
        self.running = False


# Convenience function
async def create_listener() -> DecloudWebSocket:
    """Create and connect a listener"""
    listener = DecloudWebSocket()
    await listener.connect()
    await listener.subscribe()
    return listener