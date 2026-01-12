"""
Solana client for interacting with Decloud contract
"""
import json
import struct
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import base58
from rich.console import Console

from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import ID as SYSTEM_PROGRAM_ID
from solders.instruction import Instruction, AccountMeta
from solders.transaction import Transaction
from solders.message import Message
from solders.hash import Hash
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts

from config import PROGRAM_ID, TREASURY, config, DATASETS, DATASET_ID_TO_NAME

console = Console()


@dataclass
class RoundInfo:
    """Round information from blockchain"""
    id: int
    creator: str
    model_cid: str
    dataset: str
    dataset_id: int
    reward_amount: int
    created_at: int
    status: str
    pre_count: int
    pre_accuracy_sum: int
    gradients_count: int
    total_validations: int
    total_improvement: int
    consensus_accuracy: int
    bump: int
    vault_bump: int


@dataclass
class GradientInfo:
    """Gradient information from blockchain"""
    round_id: int
    trainer: str
    cid: str
    post_count: int
    post_accuracy_sum: int
    improvement: int
    reward_claimed: bool


@dataclass
class PreValidationInfo:
    """PreValidation information from blockchain"""
    round_id: int
    validator: str
    accuracy: int
    reward_claimed: bool


@dataclass
class PostValidationInfo:
    """PostValidation information from blockchain"""
    round_id: int
    trainer: str
    validator: str
    accuracy: int
    reward_claimed: bool


class SolanaClient:
    """
    Client for interacting with Decloud Solana program
    """
    
    # Discriminators from IDL
    DISCRIMINATORS = {
        "prevalidate": bytes([22, 157, 136, 129, 168, 136, 183, 157]),
        "postvalidate": bytes([175, 191, 177, 236, 123, 174, 219, 124]),
        "claim_validator_pre": bytes([28, 54, 225, 41, 59, 120, 89, 201]),
        "claim_validator_post": bytes([225, 54, 234, 114, 42, 249, 104, 199]),
        "force_finalize": bytes([215, 116, 165, 101, 174, 80, 228, 5]),
    }
    
    # Account discriminators
    ACCOUNT_DISCRIMINATORS = {
        "Round": bytes([87, 127, 165, 51, 73, 78, 116, 174]),
        "RoundCounter": bytes([244, 2, 165, 240, 168, 10, 1, 238]),
        "Gradient": bytes([173, 254, 210, 185, 231, 180, 152, 152]),
        "PreValidation": bytes([145, 180, 236, 158, 228, 230, 229, 42]),
        "PostValidation": bytes([125, 230, 127, 163, 18, 214, 12, 100]),
    }
    
    def __init__(self, keypair: Optional[Keypair] = None):
        self.program_id = Pubkey.from_string(PROGRAM_ID)
        self.treasury = Pubkey.from_string(TREASURY)
        self.client = Client(config.rpc_url)
        self.keypair = keypair
    
    @classmethod
    def from_private_key(cls, private_key: str) -> "SolanaClient":
        """Create client from base58 private key"""
        secret = base58.b58decode(private_key)
        keypair = Keypair.from_bytes(secret)
        return cls(keypair)
    
    @property
    def pubkey(self) -> Optional[Pubkey]:
        """Get public key"""
        return self.keypair.pubkey() if self.keypair else None
    
    def get_balance(self) -> int:
        """Get SOL balance in lamports"""
        if not self.pubkey:
            return 0
        response = self.client.get_balance(self.pubkey, commitment=Confirmed)
        return response.value
    
    # ═══════════════════════════════════════════════════════════════
    # PDA Derivation
    # ═══════════════════════════════════════════════════════════════
    
    def get_round_counter_pda(self) -> Tuple[Pubkey, int]:
        """Get round counter PDA"""
        return Pubkey.find_program_address(
            [b"round_counter"],
            self.program_id
        )
    
    def get_round_pda(self, round_id: int) -> Tuple[Pubkey, int]:
        """Get round PDA"""
        return Pubkey.find_program_address(
            [b"round", round_id.to_bytes(8, "little")],
            self.program_id
        )
    
    def get_vault_pda(self, round_id: int) -> Tuple[Pubkey, int]:
        """Get vault PDA"""
        return Pubkey.find_program_address(
            [b"vault", round_id.to_bytes(8, "little")],
            self.program_id
        )
    
    def get_pre_validation_pda(self, round_id: int, validator: Pubkey) -> Tuple[Pubkey, int]:
        """Get pre-validation PDA"""
        return Pubkey.find_program_address(
            [b"pre", round_id.to_bytes(8, "little"), bytes(validator)],
            self.program_id
        )
    
    def get_gradient_pda(self, round_id: int, trainer: Pubkey) -> Tuple[Pubkey, int]:
        """Get gradient PDA"""
        return Pubkey.find_program_address(
            [b"gradient", round_id.to_bytes(8, "little"), bytes(trainer)],
            self.program_id
        )
    
    def get_post_validation_pda(self, round_id: int, trainer: Pubkey, validator: Pubkey) -> Tuple[Pubkey, int]:
        """Get post-validation PDA"""
        return Pubkey.find_program_address(
            [b"post", round_id.to_bytes(8, "little"), bytes(trainer), bytes(validator)],
            self.program_id
        )
    def get_validator_stake_pda(self, validator: Pubkey) -> Tuple[Pubkey, int]:
        """Get validator stake PDA"""
        return Pubkey.find_program_address(
            [b"validator_stake", bytes(validator)],
            self.program_id
        )
    # ═══════════════════════════════════════════════════════════════
    # Read Operations
    # ═══════════════════════════════════════════════════════════════
    
    def get_round_count(self) -> int:
        """Get current round count"""
        pda, _ = self.get_round_counter_pda()
        response = self.client.get_account_info(pda, commitment=Confirmed)
        
        if response.value is None:
            return 0
        
        data = response.value.data
        # Skip discriminator (8 bytes), read count (8 bytes)
        count = struct.unpack("<Q", data[8:16])[0]
        return count
    
    def get_round(self, round_id: int) -> Optional[RoundInfo]:
        """Get round information"""
        pda, _ = self.get_round_pda(round_id)
        response = self.client.get_account_info(pda, commitment=Confirmed)
        
        if response.value is None:
            return None
        
        data = bytes(response.value.data)
        return self._parse_round(data)
    
    def _parse_round(self, data: bytes) -> Optional[RoundInfo]:
        """Parse round account data. Returns None if parsing fails (old/incompatible format)."""
        try:
            offset = 8  # Skip discriminator
            
            # Verify discriminator matches Round account
            discriminator = data[:8]
            if discriminator != self.ACCOUNT_DISCRIMINATORS.get("Round"):
                return None
            
            # id: u64
            id = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            # creator: Pubkey (32 bytes)
            creator = base58.b58encode(data[offset:offset+32]).decode()
            offset += 32
            
            # model_cid: [u8; 64]
            model_cid_bytes = data[offset:offset+64]
            offset += 64
            
            # model_cid_len: u8
            model_cid_len = data[offset]
            offset += 1
            model_cid = model_cid_bytes[:model_cid_len].decode("utf-8", errors="ignore")
            
            # dataset: enum (1 byte)
            dataset_id = data[offset]
            offset += 1
            dataset = DATASET_ID_TO_NAME.get(dataset_id, f"Unknown({dataset_id})")
        
        # reward_amount: u64
        reward_amount = struct.unpack("<Q", data[offset:offset+8])[0]
        offset += 8
        
        # created_at: i64
        created_at = struct.unpack("<q", data[offset:offset+8])[0]
        offset += 8
        
        # status: enum (1 byte)
        status_id = data[offset]
        offset += 1
        status_map = {0: "Active", 1: "Finalized", 2: "Cancelled"}
        status = status_map.get(status_id, "Unknown")
        
        # pre_count: u8
        pre_count = data[offset]
        offset += 1
        
        # pre_accuracy_sum: u64
        pre_accuracy_sum = struct.unpack("<Q", data[offset:offset+8])[0]
        offset += 8
        
        # gradients_count: u8
        gradients_count = data[offset]
        offset += 1
        
        # total_validations: u16
        total_validations = struct.unpack("<H", data[offset:offset+2])[0]
        offset += 2
        
        # total_improvement: u64
        total_improvement = struct.unpack("<Q", data[offset:offset+8])[0]
        offset += 8

        consensus_accuracy = struct.unpack("<Q", data[offset:offset+8])[0]
        offset += 8
        
        # bump: u8
        bump = data[offset]
        offset += 1
        
        # vault_bump: u8
        vault_bump = data[offset]
        
        return RoundInfo(
            id=id,
            creator=creator,
            model_cid=model_cid,
            dataset=dataset,
            dataset_id=dataset_id,
            reward_amount=reward_amount,
            created_at=created_at,
            status=status,
            pre_count=pre_count,
            pre_accuracy_sum=pre_accuracy_sum,
            gradients_count=gradients_count,
            total_validations=total_validations,
            total_improvement=total_improvement,
            consensus_accuracy=consensus_accuracy,
            bump=bump,
            vault_bump=vault_bump,
        )
        except Exception:
            # Failed to parse round - likely old/incompatible format
            return None
    
    def get_gradient(self, round_id: int, trainer: Pubkey) -> Optional[GradientInfo]:
        """Get gradient information"""
        pda, _ = self.get_gradient_pda(round_id, trainer)
        response = self.client.get_account_info(pda, commitment=Confirmed)
        
        if response.value is None:
            return None
        
        data = bytes(response.value.data)
        return self._parse_gradient(data)
    
    def _parse_gradient(self, data: bytes) -> Optional[GradientInfo]:
        """Parse gradient account data. Returns None if parsing fails (old/incompatible format)."""
        try:
            offset = 8  # Skip discriminator
            
            # Verify discriminator matches Gradient account
            discriminator = data[:8]
            if discriminator != self.ACCOUNT_DISCRIMINATORS.get("Gradient"):
                return None
            
            round_id = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            trainer = base58.b58encode(data[offset:offset+32]).decode()
            offset += 32
            
            cid_bytes = data[offset:offset+64]
            offset += 64
            cid_len = data[offset]
            offset += 1
            cid = cid_bytes[:cid_len].decode("utf-8", errors="ignore")
            
            post_count = data[offset]
            offset += 1
            
            post_accuracy_sum = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            improvement = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            reward_claimed = bool(data[offset])
            
            return GradientInfo(
                round_id=round_id,
                trainer=trainer,
                cid=cid,
                post_count=post_count,
                post_accuracy_sum=post_accuracy_sum,
                improvement=improvement,
                reward_claimed=reward_claimed,
            )
        except Exception:
            # Failed to parse gradient - likely old/incompatible format
            return None
    
    def get_all_gradients_for_round(self, round_id: int) -> List[GradientInfo]:
        """Get all gradients submitted for a round using getProgramAccounts"""
        try:
            response = self.client.get_program_accounts(
                self.program_id,
                commitment=Confirmed,
            )
            
            gradients = []
            for account in response.value:
                try:
                    data = bytes(account.account.data)
                    
                    # Skip accounts that are too small or too large for gradient
                    if len(data) < 100 or len(data) > 200:
                        continue
                    
                    gradient = self._parse_gradient(data)
                    
                    # Skip if failed to parse (old format) or wrong round
                    if gradient is None:
                        continue
                    
                    if gradient.round_id == round_id:
                        gradients.append(gradient)
                except:
                    continue
            
            return gradients
        except Exception as e:
            return []
    
    def get_pre_validation(self, round_id: int, validator: Pubkey) -> Optional[PreValidationInfo]:
        """Get pre-validation info"""
        try:
            pda, _ = self.get_pre_validation_pda(round_id, validator)
            response = self.client.get_account_info(pda, commitment=Confirmed)
            
            if response.value is None:
                return None
            
            data = bytes(response.value.data)
            
            # Verify discriminator
            discriminator = data[:8]
            if discriminator != self.ACCOUNT_DISCRIMINATORS.get("PreValidation"):
                return None
            
            offset = 8
            
            round_id = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            validator_pk = base58.b58encode(data[offset:offset+32]).decode()
            offset += 32
            
            accuracy = struct.unpack("<I", data[offset:offset+4])[0]
            offset += 4
            
            reward_claimed = bool(data[offset])
            
            return PreValidationInfo(
                round_id=round_id,
                validator=validator_pk,
                accuracy=accuracy,
                reward_claimed=reward_claimed,
            )
        except Exception:
            return None
    
    def get_post_validation(self, round_id: int, trainer: Pubkey, validator: Pubkey) -> Optional[PostValidationInfo]:
        """Get post-validation info"""
        try:
            pda, _ = self.get_post_validation_pda(round_id, trainer, validator)
            response = self.client.get_account_info(pda, commitment=Confirmed)
            
            if response.value is None:
                return None
            
            data = bytes(response.value.data)
            
            # Verify discriminator
            discriminator = data[:8]
            if discriminator != self.ACCOUNT_DISCRIMINATORS.get("PostValidation"):
                return None
            
            offset = 8
            
            round_id = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            trainer_pk = base58.b58encode(data[offset:offset+32]).decode()
            offset += 32
            
            validator_pk = base58.b58encode(data[offset:offset+32]).decode()
            offset += 32
            
            accuracy = struct.unpack("<I", data[offset:offset+4])[0]
            offset += 4
            
            reward_claimed = bool(data[offset])
            
            return PostValidationInfo(
                round_id=round_id,
                trainer=trainer_pk,
                validator=validator_pk,
                accuracy=accuracy,
                reward_claimed=reward_claimed,
            )
        except Exception:
            return None
    
    def get_active_rounds(self) -> List[RoundInfo]:
        """Get all active rounds"""
        rounds = []
        count = self.get_round_count()
        
        for i in range(count):
            round_info = self.get_round(i)
            if round_info and round_info.status == "Active":
                rounds.append(round_info)
        
        return rounds
    
    def get_rounds_for_dataset(self, dataset: str) -> List[RoundInfo]:
        """Get active rounds for specific dataset"""
        return [r for r in self.get_active_rounds() if r.dataset == dataset]
    
    # ═══════════════════════════════════════════════════════════════
    # Write Operations
    # ═══════════════════════════════════════════════════════════════
    
    def _send_transaction(self, instruction: Instruction) -> str:
        """Send transaction with instruction"""
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        recent_blockhash = self.client.get_latest_blockhash(commitment=Confirmed).value.blockhash
        
        message = Message.new_with_blockhash(
            [instruction],
            self.keypair.pubkey(),
            recent_blockhash
        )
        
        tx = Transaction.new_unsigned(message)
        tx.sign([self.keypair], recent_blockhash)
        
        response = self.client.send_transaction(tx, opts=TxOpts(skip_preflight=False, preflight_commitment=Confirmed))
        return str(response.value)
    
    def prevalidate(self, round_id: int, accuracy: int) -> str:
        """
        Submit pre-validation for a round
        accuracy: percentage * 100 (e.g., 75% = 7500)
        """
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        round_pda, _ = self.get_round_pda(round_id)
        validator_stake_pda, _ = self.get_validator_stake_pda(self.keypair.pubkey())
        pre_pda, _ = self.get_pre_validation_pda(round_id, self.keypair.pubkey())
        
        # Build instruction data
        data = self.DISCRIMINATORS["prevalidate"]
        data += struct.pack("<Q", round_id)  # round_id: u64
        data += struct.pack("<I", accuracy)  # accuracy: u32
        
        accounts = [
            AccountMeta(round_pda, is_signer=False, is_writable=True),
            AccountMeta(validator_stake_pda, is_signer=False, is_writable=False),
            AccountMeta(pre_pda, is_signer=False, is_writable=True),
            AccountMeta(self.keypair.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
        ]
        
        instruction = Instruction(self.program_id, data, accounts)
        return self._send_transaction(instruction)
    
    def postvalidate(self, round_id: int, trainer: Pubkey, accuracy: int) -> str:
        """
        Submit post-validation for a gradient
        accuracy: percentage * 100 (e.g., 85% = 8500)
        """
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        round_pda, _ = self.get_round_pda(round_id)
        validator_stake_pda, _ = self.get_validator_stake_pda(self.keypair.pubkey())
        gradient_pda, _ = self.get_gradient_pda(round_id, trainer)
        post_pda, _ = self.get_post_validation_pda(round_id, trainer, self.keypair.pubkey())
    
        
        data = self.DISCRIMINATORS["postvalidate"]
        data += struct.pack("<Q", round_id)
        data += struct.pack("<I", accuracy)
        
        accounts = [
            AccountMeta(round_pda, is_signer=False, is_writable=True),
            AccountMeta(validator_stake_pda, is_signer=False, is_writable=False),
            AccountMeta(gradient_pda, is_signer=False, is_writable=True),
            AccountMeta(post_pda, is_signer=False, is_writable=True),
            AccountMeta(self.keypair.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
        ]
        
        instruction = Instruction(self.program_id, data, accounts)
        return self._send_transaction(instruction)
    
    def claim_validator_pre(self, round_id: int) -> str:
        """Claim pre-validation reward"""
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        round_pda, _ = self.get_round_pda(round_id)
        round_info = self.get_round(round_id)
        if not round_info:
            raise ValueError(f"Round {round_id} not found")
        
        vault_pda, _ = self.get_vault_pda(round_id)
        pre_pda, _ = self.get_pre_validation_pda(round_id, self.keypair.pubkey())
        
        data = self.DISCRIMINATORS["claim_validator_pre"]
        data += struct.pack("<Q", round_id)
        
        accounts = [
            AccountMeta(round_pda, is_signer=False, is_writable=False),
            AccountMeta(pre_pda, is_signer=False, is_writable=True),
            AccountMeta(vault_pda, is_signer=False, is_writable=True),
            AccountMeta(self.treasury, is_signer=False, is_writable=True),
            AccountMeta(self.keypair.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
        ]
        
        instruction = Instruction(self.program_id, data, accounts)
        return self._send_transaction(instruction)
    
    def claim_validator_post(self, round_id: int, trainer: Pubkey) -> str:
        """Claim post-validation reward"""
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        round_pda, _ = self.get_round_pda(round_id)
        vault_pda, _ = self.get_vault_pda(round_id)
        post_pda, _ = self.get_post_validation_pda(round_id, trainer, self.keypair.pubkey())
        
        data = self.DISCRIMINATORS["claim_validator_post"]
        data += struct.pack("<Q", round_id)
        
        accounts = [
            AccountMeta(round_pda, is_signer=False, is_writable=False),
            AccountMeta(post_pda, is_signer=False, is_writable=True),
            AccountMeta(vault_pda, is_signer=False, is_writable=True),
            AccountMeta(self.treasury, is_signer=False, is_writable=True),
            AccountMeta(self.keypair.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
        ]
        
        instruction = Instruction(self.program_id, data, accounts)
        return self._send_transaction(instruction)
    
    def force_finalize(self, round_id: int) -> str:
        """Force finalize a round after 12 hours"""
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        round_pda, _ = self.get_round_pda(round_id)
        
        data = self.DISCRIMINATORS["force_finalize"]
        data += struct.pack("<Q", round_id)
        
        accounts = [
            AccountMeta(round_pda, is_signer=False, is_writable=True),
            AccountMeta(self.keypair.pubkey(), is_signer=True, is_writable=False),
        ]
        
        instruction = Instruction(self.program_id, data, accounts)
        return self._send_transaction(instruction)
    
    def has_prevalidated(self, round_id: int) -> bool:
        """Check if current validator has prevalidated"""
        if not self.keypair:
            return False
        pre = self.get_pre_validation(round_id, self.keypair.pubkey())
        return pre is not None
    
    def has_postvalidated(self, round_id: int, trainer: Pubkey) -> bool:
        """Check if current validator has postvalidated a trainer"""
        if not self.keypair:
            return False
        post = self.get_post_validation(round_id, trainer, self.keypair.pubkey())
        return post is not None