"""
DECLOUD Validator
=================

Main validator class for participating in the DECLOUD network.
"""

import os
import json
import time
import subprocess
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

from .config import Config
from .dataset_manager import DatasetManager, DATASETS


@dataclass
class Round:
    """Represents a training round"""
    id: int
    creator: str
    model_cid: str
    dataset: str
    reward_amount: int
    status: str
    validator: str
    trainers_count: int
    submissions_count: int
    validator_deadline: int
    join_deadline: int
    submit_deadline: int


@dataclass
class Submission:
    """Represents a trainer submission"""
    trainer: str
    gradient_cid: str
    submitted: bool
    contribution_bps: int
    reward_claimed: bool


class Validator:
    """
    DECLOUD Validator Client
    
    Validates training submissions and earns rewards.
    
    Usage:
        validator = Validator()
        validator.login("your_base58_private_key")
        
        # Manual mode
        rounds = validator.get_available_rounds()
        validator.claim_round(round_id)
        validator.start_training(round_id)
        # ... wait for submissions ...
        validator.evaluate_and_complete(round_id)
        
        # Auto mode
        validator.start()  # Runs forever, auto-claims and validates
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize validator"""
        self.config = config or Config.load()
        self.datasets = DatasetManager(self.config.data_dir)
        self._private_key: Optional[str] = None
        self._public_key: Optional[str] = None
        self._balance: float = 0.0
        
        # Ensure Node.js dependencies exist
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        try:
            result = subprocess.run(
                ["node", "-e", "require('@solana/web3.js')"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print("⚠️  Warning: @solana/web3.js not found")
                print("   Run: npm install @solana/web3.js @coral-xyz/anchor bs58")
        except FileNotFoundError:
            print("⚠️  Warning: Node.js not found")
    
    def _get_idl_load_script(self) -> str:
        """Get JS code to load IDL"""
        idl_path = self.config.idl_path
        return f'const idl = JSON.parse(require("fs").readFileSync("{idl_path}"));'
    
    def _run_node(self, script: str) -> str:
        """Run a Node.js script and return stdout"""
        result = subprocess.run(
            ["node", "-e", script],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode != 0:
            raise RuntimeError(f"Node.js error: {result.stderr}")
        return result.stdout.strip()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AUTHENTICATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def login(self, private_key: str) -> bool:
        """
        Login with a base58-encoded private key
        
        Args:
            private_key: Base58-encoded private key
            
        Returns:
            True if successful
        """
        self._private_key = private_key
        
        try:
            script = f'''
const {{ Connection, Keypair }} = require("@solana/web3.js");
const bs58 = require("bs58");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const kp = Keypair.fromSecretKey(bs58.decode("{private_key}"));

(async () => {{
    const balance = await conn.getBalance(kp.publicKey);
    console.log(JSON.stringify({{
        pubkey: kp.publicKey.toString(),
        balance: balance
    }}));
}})();
'''
            result = json.loads(self._run_node(script))
            self._public_key = result["pubkey"]
            self._balance = result["balance"] / 1e9
            
            print(f"✓ Logged in as: {self._public_key[:20]}...")
            print(f"  Balance: {self._balance:.4f} SOL")
            
            return True
            
        except Exception as e:
            print(f"❌ Login failed: {e}")
            self._private_key = None
            return False
    
    def login_from_file(self, keypair_path: str) -> bool:
        """Login from a Solana keypair JSON file"""
        try:
            script = f'''
const fs = require("fs");
const bs58 = require("bs58");

const secret = JSON.parse(fs.readFileSync("{keypair_path}"));
console.log(bs58.encode(Buffer.from(secret)));
'''
            private_key = self._run_node(script)
            return self.login(private_key)
            
        except Exception as e:
            print(f"❌ Failed to read keypair: {e}")
            return False
    
    @property
    def is_logged_in(self) -> bool:
        """Check if logged in"""
        return self._private_key is not None
    
    @property
    def public_key(self) -> Optional[str]:
        """Get public key"""
        return self._public_key
    
    @property
    def balance(self) -> float:
        """Get SOL balance"""
        return self._balance
    
    def refresh_balance(self) -> float:
        """Refresh and return current balance"""
        if not self._public_key:
            return 0.0
        
        script = f'''
const {{ Connection, PublicKey }} = require("@solana/web3.js");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const pubkey = new PublicKey("{self._public_key}");

(async () => {{
    const balance = await conn.getBalance(pubkey);
    console.log(balance);
}})();
'''
        self._balance = int(self._run_node(script)) / 1e9
        return self._balance
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ROUND MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_all_rounds(self) -> List[Round]:
        """Get all rounds from the blockchain"""
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey }} = require("@solana/web3.js");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));

(async () => {{
    const wallet = {{ publicKey: PublicKey.default }};
    const provider = new anchor.AnchorProvider(conn, wallet, {{}});
    const program = new anchor.Program(idl, provider);
    
    const rounds = await program.account.round.all();
    const result = rounds.map(r => ({{
        id: r.account.id.toNumber(),
        creator: r.account.creator.toString(),
        modelCid: r.account.modelCid,
        dataset: Object.keys(r.account.dataset)[0],
        rewardAmount: r.account.rewardAmount.toNumber(),
        status: Object.keys(r.account.status)[0],
        validator: r.account.validator.toString(),
        trainersCount: r.account.trainersCount,
        submissionsCount: r.account.submissionsCount,
        validatorDeadline: r.account.validatorDeadline.toNumber(),
        joinDeadline: r.account.joinDeadline.toNumber(),
        submitDeadline: r.account.submitDeadline.toNumber(),
    }}));
    console.log(JSON.stringify(result));
}})();
'''
        data = json.loads(self._run_node(script))
        return [Round(
            id=r["id"],
            creator=r["creator"],
            model_cid=r["modelCid"],
            dataset=r["dataset"],
            reward_amount=r["rewardAmount"],
            status=r["status"],
            validator=r["validator"],
            trainers_count=r["trainersCount"],
            submissions_count=r["submissionsCount"],
            validator_deadline=r["validatorDeadline"],
            join_deadline=r["joinDeadline"],
            submit_deadline=r["submitDeadline"],
        ) for r in data]
    
    def get_available_rounds(self, check_datasets: bool = True) -> List[Round]:
        """
        Get rounds waiting for a validator.
        
        Args:
            check_datasets: If True, only return rounds where dataset is downloaded
        """
        rounds = self.get_all_rounds()
        now = int(time.time())
        
        available = []
        for r in rounds:
            # Basic checks
            if r.status != "waitingValidator":
                continue
            if r.validator_deadline <= now:
                continue
            if r.reward_amount / 1e9 < self.config.min_reward:
                continue
            
            # Check if dataset is downloaded
            if check_datasets:
                if not self.datasets.is_downloaded(r.dataset):
                    continue
            
            available.append(r)
        
        return available
    
    def get_missing_datasets(self) -> List[str]:
        """Get list of datasets needed for available rounds but not downloaded"""
        rounds = self.get_all_rounds()
        now = int(time.time())
        
        missing = set()
        for r in rounds:
            if r.status == "waitingValidator" and r.validator_deadline > now:
                if not self.datasets.is_downloaded(r.dataset):
                    missing.add(r.dataset)
        
        return list(missing)
    
    def get_my_rounds(self) -> List[Round]:
        """Get rounds where I am the validator"""
        if not self._public_key:
            return []
        
        rounds = self.get_all_rounds()
        return [r for r in rounds if r.validator == self._public_key]
    
    def claim_round(self, round_id: int, skip_dataset_check: bool = False) -> str:
        """
        Claim a round as validator
        
        Args:
            round_id: Round ID to claim
            skip_dataset_check: Skip checking if dataset is downloaded (not recommended)
        """
        if not self._private_key:
            raise RuntimeError("Not logged in")
        
        # Check if dataset is downloaded
        if not skip_dataset_check:
            rounds = self.get_all_rounds()
            round_data = next((r for r in rounds if r.id == round_id), None)
            
            if round_data:
                if not self.datasets.is_downloaded(round_data.dataset):
                    raise RuntimeError(
                        f"Dataset '{round_data.dataset}' not downloaded. "
                        f"Run: decloud datasets download --dataset {round_data.dataset}"
                    )
        
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey, Keypair }} = require("@solana/web3.js");
const bs58 = require("bs58");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));
const validator = Keypair.fromSecretKey(bs58.decode("{self._private_key}"));

(async () => {{
    const wallet = new anchor.Wallet(validator);
    const provider = new anchor.AnchorProvider(conn, wallet, {{}});
    const program = new anchor.Program(idl, provider);
    
    const roundId = new anchor.BN({round_id});
    const [roundPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    
    const tx = await program.methods
        .claimRound(roundId)
        .accounts({{ round: roundPda, validator: validator.publicKey }})
        .signers([validator])
        .rpc();
    console.log(tx);
}})();
'''
        return self._run_node(script)
    
    def start_training(self, round_id: int, baseline_score: int = 0) -> str:
        """Start training phase for a round"""
        if not self._private_key:
            raise RuntimeError("Not logged in")
        
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey, Keypair }} = require("@solana/web3.js");
const bs58 = require("bs58");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));
const validator = Keypair.fromSecretKey(bs58.decode("{self._private_key}"));

(async () => {{
    const wallet = new anchor.Wallet(validator);
    const provider = new anchor.AnchorProvider(conn, wallet, {{}});
    const program = new anchor.Program(idl, provider);
    
    const roundId = new anchor.BN({round_id});
    const [roundPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    
    const tx = await program.methods
        .startTraining(roundId, {baseline_score})
        .accounts({{ round: roundPda, validator: validator.publicKey }})
        .signers([validator])
        .rpc();
    console.log(tx);
}})();
'''
        return self._run_node(script)
    
    def start_training_with_baseline(self, round_id: int) -> str:
        """
        Start training with ML-computed baseline score.
        Downloads model from IPFS, evaluates on dataset, then starts training.
        """
        # Get round info
        rounds = self.get_all_rounds()
        round_data = next((r for r in rounds if r.id == round_id), None)
        
        if not round_data:
            raise RuntimeError(f"Round #{round_id} not found")
        
        baseline_score = 0
        
        try:
            from .ml_validator import MLValidator
            
            ml = MLValidator(self.config.data_dir)
            
            # Download model
            import os
            model_path = os.path.join(self.config.data_dir, f"model_{round_data.model_cid[:16]}.pt")
            
            if not os.path.exists(model_path):
                print(f"   📥 Downloading model from IPFS...")
                if not ml.ipfs.download(round_data.model_cid, model_path):
                    print(f"   ⚠️ Failed to download model, using baseline_score=0")
                else:
                    print(f"   ✓ Model downloaded")
            
            if os.path.exists(model_path):
                print(f"   🔬 Evaluating baseline accuracy...")
                model = ml.load_model(model_path)
                accuracy = ml.evaluate(model, round_data.dataset)
                baseline_score = int(accuracy * 100)  # Convert to basis points (e.g. 45.5% -> 4550)
                print(f"   ✓ Baseline accuracy: {accuracy:.2f}% (score: {baseline_score})")
                
        except ImportError:
            print("   ⚠️ ML validator not available, using baseline_score=0")
        except Exception as e:
            print(f"   ⚠️ Baseline evaluation failed: {e}, using score=0")
        
        return self.start_training(round_id, baseline_score)
    
    def get_submissions(self, round_id: int) -> List[Submission]:
        """Get all submissions for a round"""
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey }} = require("@solana/web3.js");
const bs58 = require("bs58");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));

(async () => {{
    const wallet = {{ publicKey: PublicKey.default }};
    const provider = new anchor.AnchorProvider(conn, wallet, {{}});
    const program = new anchor.Program(idl, provider);
    
    // Get all submissions and filter by round_id
    const allSubmissions = await program.account.submission.all();
    const submissions = allSubmissions.filter(s => s.account.roundId.toNumber() === {round_id});
    
    const result = submissions.map(s => ({{
        trainer: s.account.trainer.toString(),
        gradientCid: s.account.gradientCid,
        submitted: s.account.submitted,
        contributionBps: s.account.contributionBps,
        rewardClaimed: s.account.rewardClaimed,
    }}));
    console.log(JSON.stringify(result));
}})();
'''
        data = json.loads(self._run_node(script))
        return [Submission(
            trainer=s["trainer"],
            gradient_cid=s["gradientCid"],
            submitted=s["submitted"],
            contribution_bps=s["contributionBps"],
            reward_claimed=s["rewardClaimed"],
        ) for s in data]
    
    def set_contribution(self, round_id: int, trainer: str, contribution_bps: int) -> str:
        """Set contribution percentage for a trainer"""
        if not self._private_key:
            raise RuntimeError("Not logged in")
        
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey, Keypair }} = require("@solana/web3.js");
const bs58 = require("bs58");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));
const validator = Keypair.fromSecretKey(bs58.decode("{self._private_key}"));

(async () => {{
    const wallet = new anchor.Wallet(validator);
    const provider = new anchor.AnchorProvider(conn, wallet, {{}});
    const program = new anchor.Program(idl, provider);
    
    const roundId = new anchor.BN({round_id});
    const trainerPubkey = new PublicKey("{trainer}");
    
    const [roundPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    const [submissionPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("submission"), roundId.toArrayLike(Buffer, "le", 8), trainerPubkey.toBuffer()],
        programId
    );
    
    const tx = await program.methods
        .setContribution(roundId, {contribution_bps})
        .accounts({{
            round: roundPda,
            submission: submissionPda,
            validator: validator.publicKey,
        }})
        .signers([validator])
        .rpc();
    console.log(tx);
}})();
'''
        return self._run_node(script)
    
    def complete_validation(self, round_id: int) -> str:
        """Complete validation and receive fee"""
        if not self._private_key:
            raise RuntimeError("Not logged in")
        
        # Get referrer from config, default to TREASURY if not set
        referrer_js = "TREASURY"
        if self.config.referrer:
            referrer_js = f'new PublicKey("{self.config.referrer}")'
        
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey, Keypair }} = require("@solana/web3.js");
const bs58 = require("bs58");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));
const validator = Keypair.fromSecretKey(bs58.decode("{self._private_key}"));

const TREASURY = new PublicKey("FzuCxi65QyFXAGbHcXB28RXqyBZSZ5KXLQxeofx1P9K2");

(async () => {{
    const wallet = new anchor.Wallet(validator);
    const provider = new anchor.AnchorProvider(conn, wallet, {{}});
    const program = new anchor.Program(idl, provider);
    
    const roundId = new anchor.BN({round_id});
    const [roundPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    const [vaultPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("vault"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    
    const referrer = {referrer_js};
    
    const tx = await program.methods
        .completeValidation(roundId)
        .accounts({{
            round: roundPda,
            vault: vaultPda,
            treasury: TREASURY,
            referrer: referrer,
            validator: validator.publicKey,
            systemProgram: anchor.web3.SystemProgram.programId,
        }})
        .signers([validator])
        .rpc();
    console.log(tx);
}})();
'''
        return self._run_node(script)
    
    def abort_round(self, round_id: int, creator: str) -> str:
        """
        Abort round and refund all money to creator.
        Works on: WaitingTrainers, Training, Validating statuses.
        """
        if not self._private_key:
            raise RuntimeError("Not logged in")
        
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey, Keypair }} = require("@solana/web3.js");
const bs58 = require("bs58");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));
const validator = Keypair.fromSecretKey(bs58.decode("{self._private_key}"));

(async () => {{
    const wallet = new anchor.Wallet(validator);
    const provider = new anchor.AnchorProvider(conn, wallet, {{}});
    const program = new anchor.Program(idl, provider);
    
    const roundId = new anchor.BN({round_id});
    const [roundPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    const [vaultPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("vault"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    const creator = new PublicKey("{creator}");
    
    const tx = await program.methods
        .abortRound(roundId)
        .accounts({{
            round: roundPda,
            vault: vaultPda,
            creator: creator,
            validator: validator.publicKey,
            systemProgram: anchor.web3.SystemProgram.programId,
        }})
        .signers([validator])
        .rpc();
    console.log(tx);
}})();
'''
        return self._run_node(script)
    
    def check_round(self, round_id: int, creator: str) -> str:
        """
        Check round status and handle timeouts.
        This will transition expired rounds to proper status.
        """
        if not self._private_key:
            raise RuntimeError("Not logged in")
        
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey, Keypair }} = require("@solana/web3.js");
const bs58 = require("bs58");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));
const signer = Keypair.fromSecretKey(bs58.decode("{self._private_key}"));

(async () => {{
    const wallet = new anchor.Wallet(signer);
    const provider = new anchor.AnchorProvider(conn, wallet, {{}});
    const program = new anchor.Program(idl, provider);
    
    const roundId = new anchor.BN({round_id});
    const [roundPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    const [vaultPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("vault"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    const creator = new PublicKey("{creator}");
    
    const tx = await program.methods
        .checkRound(roundId)
        .accounts({{
            round: roundPda,
            vault: vaultPda,
            creator: creator,
            systemProgram: anchor.web3.SystemProgram.programId,
        }})
        .rpc();
    console.log(tx);
}})();
'''
        return self._run_node(script)
    
    def cleanup_stale_rounds(self):
        """Check and cleanup all stale rounds we're validator for"""
        print("   🧹 Cleaning up stale rounds...")
        my_rounds = self.get_my_rounds()
        
        for r in my_rounds:
            if r.status in ("completed", "cancelled", "expired"):
                continue
            
            try:
                print(f"   ⏳ Checking round #{r.id} ({r.status})...")
                tx = self.check_round(r.id, r.creator)
                print(f"   ✓ Round #{r.id} checked: {tx[:30]}...")
            except Exception as e:
                # Might fail if already in valid state
                if "already" not in str(e).lower():
                    print(f"   ⚠️ Round #{r.id}: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDATION HELPERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def evaluate_submissions(
        self,
        round_id: int,
        evaluator: Optional[Callable[[Submission], float]] = None,
        use_ml: bool = True
    ) -> Dict[str, int]:
        """
        Evaluate submissions and return contribution percentages.
        
        Args:
            round_id: Round ID
            evaluator: Optional custom evaluator function.
            use_ml: Use ML-based evaluation (gradient application + accuracy check)
        
        Returns:
            Dict mapping trainer pubkey to contribution_bps
        """
        submissions = self.get_submissions(round_id)
        submitted = [s for s in submissions if s.submitted]
        
        if not submitted:
            return {}
        
        # Try ML validation first
        if use_ml and not evaluator:
            try:
                from .ml_validator import MLValidator
                
                # Get round info
                rounds = self.get_all_rounds()
                round_data = next((r for r in rounds if r.id == round_id), None)
                
                if round_data:
                    ml = MLValidator(self.config.data_dir)
                    gradient_submissions = [(s.trainer, s.gradient_cid) for s in submitted]
                    contributions = ml.validate_round(
                        model_cid=round_data.model_cid,
                        gradient_submissions=gradient_submissions,
                        dataset_name=round_data.dataset,
                    )
                    return contributions
                    
            except ImportError:
                print("ML validator not available, using equal distribution")
            except Exception as e:
                print(f"ML validation failed: {e}, using equal distribution")
        
        if evaluator:
            scores = {s.trainer: evaluator(s) for s in submitted}
            total = sum(scores.values())
            if total == 0:
                bps = 10000 // len(submitted)
                return {s.trainer: bps for s in submitted}
            contributions = {trainer: int(score / total * 10000) for trainer, score in scores.items()}
        else:
            bps_each = 10000 // len(submitted)
            contributions = {s.trainer: bps_each for s in submitted}
        
        total = sum(contributions.values())
        if total != 10000:
            first = list(contributions.keys())[0]
            contributions[first] += 10000 - total
        
        return contributions
    
    def evaluate_and_complete(
        self,
        round_id: int,
        evaluator: Optional[Callable[[Submission], float]] = None
    ) -> Optional[str]:
        """
        Evaluate all submissions, set contributions, and complete validation.
        
        Returns:
            Transaction signature or None if skipped
        """
        print(f"\n📊 Evaluating round #{round_id}...")
        
        contributions = self.evaluate_submissions(round_id, evaluator)
        
        if not contributions:
            # Check if there were any submissions at all
            submissions = self.get_submissions(round_id)
            submitted = [s for s in submissions if s.submitted]
            
            if not submitted:
                print("   ⚠️ No submissions at all")
                print("   ⏭️  Waiting for timeout or submissions...")
                return None
            
            # There were submissions but all failed evaluation
            # DON'T complete round - wait for timeout, funds return to creator
            print("   💩 All submissions were garbage!")
            print("   ⏳ NOT completing round - waiting for timeout")
            print("   💰 Funds will return to creator when round expires")
            return None
        
        # Set contributions for each trainer
        for trainer, bps in contributions.items():
            print(f"   Setting {trainer[:20]}... = {bps/100:.1f}%")
            self.set_contribution(round_id, trainer, bps)
        
        # Complete validation
        print("   Completing validation...")
        tx = self.complete_validation(round_id)
        print(f"✓ Round #{round_id} completed! TX: {tx[:40]}...")
        
        return tx
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AUTO-VALIDATION LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════════════════
    # AUTO-VALIDATION LOOP (with WebSocket support)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def start(self, evaluator: Optional[Callable[[Submission], float]] = None):
        """
        Start the auto-validation loop with WebSocket support.
        
        This will:
        1. Listen for new rounds via WebSocket (or poll as fallback)
        2. Claim available rounds (only if dataset is downloaded)
        3. Wait for trainers to join
        4. Start training
        5. Wait for submissions
        6. Evaluate and complete
        
        Args:
            evaluator: Optional custom evaluator function
        """
        if not self.is_logged_in:
            raise RuntimeError("Not logged in. Call login() first.")
        
        print("\n" + "═" * 60)
        print("  🌐 DECLOUD Validator")
        print("═" * 60)
        print(f"  Validator:  {self._public_key[:30]}...")
        print(f"  Network:    {self.config.network}")
        print(f"  Mode:       {'WebSocket' if self.config.use_websocket else 'Polling'}")
        print(f"  Auto Claim: {self.config.auto_claim}")
        print(f"  Dry Run:    {self.config.dry_run}")
        print(f"  Min Reward: {self.config.min_reward} SOL")
        print(f"  Max Reward: {self.config.max_reward} SOL")
        
        # Show downloaded datasets
        downloaded = [d for d in self.datasets.list_datasets() if self.datasets.is_downloaded(d)]
        print(f"  Datasets:   {len(downloaded)} downloaded")
        print("═" * 60)
        print("  Listening for rounds... (Ctrl+C to stop)\n")
        
        if self.config.use_websocket:
            self._start_websocket_loop(evaluator)
        else:
            self._start_polling_loop(evaluator)
    
    def _start_websocket_loop(self, evaluator: Optional[Callable] = None):
        """WebSocket-based validation loop - real-time round monitoring"""
        from .monitor import Monitor
        import threading
        
        monitor = Monitor(self.config)
        active_rounds: set = set()
        
        # Cleanup stale rounds first
        self.cleanup_stale_rounds()
        
        # Check if we already have active rounds (after cleanup)
        existing = [r for r in self.get_my_rounds() 
                    if r.status not in ("completed", "cancelled", "expired")]
        if existing:
            for r in existing:
                active_rounds.add(r.id)
                print(f"   📌 Active round #{r.id} ({r.status})")
        else:
            print("   ✓ No active rounds, ready to claim new ones")
        
        def handle_new_round(event):
            """Handle new round event from WebSocket"""
            round_id = event.round_id
            data = event.data
            
            dataset = data.get("dataset", "")
            reward = data.get("reward", 0)
            
            # Already have active round?
            if active_rounds:
                print(f"   ⏭️  Skipped #{round_id}: already working on round #{list(active_rounds)[0]}")
                return
            
            # Check filters
            matches, reason = self.config.matches_round({
                "reward_amount": reward,
                "dataset": dataset,
                "creator": "",
            })
            
            if not matches:
                print(f"   ⏭️  Skipped: {reason}")
                return
            
            # Check dataset downloaded
            if self.config.only_downloaded and not self.datasets.is_downloaded(dataset):
                print(f"   ⏭️  Skipped: dataset '{dataset}' not downloaded")
                return
            
            # Dry run
            if self.config.dry_run:
                print(f"   👁️  [DRY RUN] Would claim round #{round_id}")
                return
            
            # Auto claim
            if self.config.auto_claim:
                try:
                    time.sleep(self.config.claim_delay)
                    tx = self.claim_round(round_id)
                    print(f"   ✅ Claimed! TX: {tx[:40]}...")
                    active_rounds.add(round_id)
                except Exception as e:
                    print(f"   ❌ Claim failed: {e}")
        
        def handle_trainer_joined(event):
            """React to trainers joining"""
            round_id = event.round_id
            trainers = event.data.get("trainers_count", 0)
            
            if round_id in active_rounds and self.config.auto_start:
                print(f"   👤 {trainers} trainer(s) in round #{round_id}")
        
        def handle_completed(event):
            """Handle round completion"""
            round_id = event.round_id
            active_rounds.discard(round_id)
        
        # Register WebSocket event handlers
        monitor.on("new_round", handle_new_round)
        monitor.on("trainer_joined", handle_trainer_joined)
        monitor.on("completed", handle_completed)
        
        # Background thread to check our rounds and auto-progress them
        def poll_my_rounds():
            print("   🔄 Background poller started")
            current_round_id = None
            last_check_time = 0
            
            while True:
                try:
                    time.sleep(5)  # Check every 5s
                    
                    # Get only MY active rounds (not completed, not cancelled)
                    my_rounds = [r for r in self.get_my_rounds() 
                                 if r.status not in ("completed", "cancelled", "expired")]
                    
                    if not my_rounds:
                        if current_round_id:
                            print(f"\n   ✓ Round #{current_round_id} finished!")
                            active_rounds.discard(current_round_id)
                            current_round_id = None
                        continue
                    
                    # Take only first active round (1 validator = 1 round)
                    r = my_rounds[0]
                    current_round_id = r.id
                    
                    # Show status
                    print(f"\r   ⏳ Round #{r.id}: {r.status}, trainers: {r.trainers_count}, subs: {r.submissions_count}   ", end="", flush=True)
                    
                    # Periodically call checkRound to handle timeouts (every 10s)
                    now = int(time.time())
                    if now - last_check_time >= 10:
                        try:
                            self.check_round(r.id, r.creator)
                            last_check_time = now
                        except Exception:
                            pass  # Ignore errors, status might not need update
                        
                        # Re-fetch round after check
                        my_rounds = [r for r in self.get_my_rounds() 
                                     if r.status not in ("completed", "cancelled", "expired")]
                        if not my_rounds:
                            continue
                        r = my_rounds[0]
                    
                    if r.status == "waitingTrainers":
                        if r.trainers_count > 0:
                            if self.config.auto_start and not self.config.dry_run:
                                print(f"\n\n🚀 Starting training for round #{r.id} ({r.trainers_count} trainers)")
                                try:
                                    self.start_training_with_baseline(r.id)
                                    print("   ✓ Training started!")
                                except Exception as e:
                                    print(f"   ❌ Failed to start training: {e}")
                    
                    elif r.status == "training":
                        # Show submissions progress
                        pass
                    
                    elif r.status == "validating":
                        if self.config.auto_validate and not self.config.dry_run:
                            print(f"\n\n📊 Validating round #{r.id}...")
                            try:
                                result = self.evaluate_and_complete(r.id, evaluator)
                                if result:
                                    active_rounds.discard(r.id)
                                    current_round_id = None
                                    print("   ✓ Validation complete! Ready for next round.")
                                else:
                                    print("   ⏳ Will retry validation later...")
                            except Exception as e:
                                print(f"   ❌ Validation error: {e}")
                                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"\n   ⚠️ Poll error: {e}")
        
        poll_thread = threading.Thread(target=poll_my_rounds, daemon=True)
        poll_thread.start()
        
        # Start WebSocket listener (blocking)
        monitor.start()
    
    def _start_polling_loop(self, evaluator: Optional[Callable] = None):
        """Polling-based validation loop (fallback when WebSocket disabled)"""
        active_round: Optional[int] = None
        last_missing_check = 0
        
        while True:
            try:
                # Check my active rounds
                my_rounds = self.get_my_rounds()
                
                for r in my_rounds:
                    print(f"\r   ⏳ Round #{r.id}: {r.status}, trainers: {r.trainers_count}, subs: {r.submissions_count}   ", end="", flush=True)
                    
                    if r.status == "waitingTrainers":
                        if r.trainers_count > 0 and self.config.auto_start:
                            if not self.config.dry_run:
                                print(f"\n\n🚀 Starting training for round #{r.id}")
                                try:
                                    self.start_training_with_baseline(r.id)
                                    active_round = r.id
                                    print("   ✓ Training started!")
                                except Exception as e:
                                    print(f"   ❌ Failed: {e}")
                            else:
                                print(f"\n👁️  [DRY RUN] Would start training #{r.id}")
                    
                    elif r.status == "validating":
                        if self.config.auto_validate:
                            if not self.config.dry_run:
                                print(f"\n📊 Validating round #{r.id}")
                                try:
                                    result = self.evaluate_and_complete(r.id, evaluator)
                                    if result:
                                        active_round = None
                                        print("   ✓ Done!")
                                    else:
                                        print("   ⏳ Will retry...")
                                except Exception as e:
                                    print(f"   ❌ Error: {e}")
                            else:
                                print(f"\n👁️  [DRY RUN] Would validate #{r.id}")
                
                # Look for new rounds
                if self.config.auto_claim and not active_round:
                    available = self.get_available_rounds(check_datasets=True)
                    
                    for r in available:
                        # Check filters
                        matches, reason = self.config.matches_round({
                            "reward_amount": r.reward_amount,
                            "dataset": r.dataset,
                            "creator": r.creator,
                        })
                        
                        if not matches:
                            continue
                        
                        print(f"\n📋 Round #{r.id}: {r.reward_amount/1e9:.4f} SOL, {r.dataset}")
                        
                        if self.config.dry_run:
                            print(f"   👁️  [DRY RUN] Would claim")
                            continue
                        
                        tx = self.claim_round(r.id)
                        print(f"   ✓ Claimed! TX: {tx[:40]}...")
                        active_round = r.id
                        break
                    
                    if not available:
                        now = int(time.time())
                        if now - last_missing_check > 60:
                            missing = self.get_missing_datasets()
                            if missing:
                                print(f"\n💡 {len(missing)} rounds need datasets:")
                                for ds in missing[:5]:
                                    print(f"   - {ds}")
                                print("   Run: decloud datasets download --dataset <n>")
                            last_missing_check = now
                
                time.sleep(self.config.poll_interval)
                
            except KeyboardInterrupt:
                print("\n\n👋 Validator stopped")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                time.sleep(self.config.poll_interval)