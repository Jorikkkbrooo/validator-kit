"""
IPFS client for fetching model packages
"""
import asyncio
import aiohttp
import json
from pathlib import Path
from typing import Dict, Optional, List
import hashlib

from config import IPFS_GATEWAYS, MODELS_CACHE


class IPFSClient:
    """
    IPFS client with gateway fallback
    """
    
    def __init__(self, gateways: List[str] = IPFS_GATEWAYS, timeout: int = 60):
        self.gateways = gateways
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.cache_dir = MODELS_CACHE
    
    async def fetch_file(self, cid: str, filename: str = "") -> Optional[bytes]:
        """
        Fetch a single file from IPFS
        Returns None if all gateways fail
        """
        path = f"{cid}/{filename}" if filename else cid
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            for gateway in self.gateways:
                url = f"{gateway}{path}"
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await response.read()
                except Exception as e:
                    continue
        
        return None
    
    async def fetch_json(self, cid: str, filename: str = "config.json") -> Optional[Dict]:
        """Fetch and parse JSON file"""
        data = await self.fetch_file(cid, filename)
        if data:
            return json.loads(data.decode("utf-8"))
        return None
    
    async def fetch_model_package(self, cid: str) -> Optional[Dict[str, bytes]]:
        """
        Fetch complete model package (config.json, head.safetensors, embeddings.safetensors)
        Returns dict of filename -> bytes
        """
        required_files = ["config.json", "head.safetensors"]
        optional_files = ["embeddings.safetensors"]
        
        result = {}
        
        # Fetch required files
        for filename in required_files:
            data = await self.fetch_file(cid, filename)
            if data is None:
                print(f"Failed to fetch required file: {filename}")
                return None
            result[filename] = data
        
        # Fetch optional files
        for filename in optional_files:
            data = await self.fetch_file(cid, filename)
            if data:
                result[filename] = data
        
        return result
    
    async def download_model_package(self, cid: str) -> Optional[Path]:
        """
        Download and cache model package
        Returns path to cached directory
        """
        cache_path = self.cache_dir / cid
        
        # Check cache
        if cache_path.exists() and (cache_path / "config.json").exists():
            return cache_path
        
        # Fetch from IPFS
        package = await self.fetch_model_package(cid)
        if package is None:
            return None
        
        # Save to cache
        cache_path.mkdir(parents=True, exist_ok=True)
        for filename, data in package.items():
            file_path = cache_path / filename
            with open(file_path, "wb") as f:
                f.write(data)
        
        return cache_path
    
    def download_model_package_sync(self, cid: str) -> Optional[Path]:
        """Synchronous wrapper for download_model_package"""
        return asyncio.run(self.download_model_package(cid))
    
    async def verify_cid(self, cid: str) -> bool:
        """Verify that CID exists and is accessible"""
        data = await self.fetch_file(cid, "config.json")
        return data is not None
    
    def verify_cid_sync(self, cid: str) -> bool:
        """Synchronous wrapper for verify_cid"""
        return asyncio.run(self.verify_cid(cid))
    
    def is_cached(self, cid: str) -> bool:
        """Check if model is cached locally"""
        cache_path = self.cache_dir / cid
        return cache_path.exists() and (cache_path / "config.json").exists()
    
    def get_cached_path(self, cid: str) -> Optional[Path]:
        """Get path to cached model if exists"""
        cache_path = self.cache_dir / cid
        if self.is_cached(cid):
            return cache_path
        return None
    
    def clear_cache(self, cid: Optional[str] = None):
        """Clear cache for specific CID or all"""
        import shutil
        
        if cid:
            cache_path = self.cache_dir / cid
            if cache_path.exists():
                shutil.rmtree(cache_path)
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)


# Global instance
ipfs_client = IPFSClient()
