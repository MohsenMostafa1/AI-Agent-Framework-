import torch
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class KVCache:
    keys: torch.Tensor
    values: torch.Tensor
    current_length: int = 0

class KVCacheManager:
    def __init__(self, 
                 max_seq_length: int = 4096,
                 chunk_size: int = 512,
                 offload_device: Optional[str] = "cpu"):
        self.max_seq_length = max_seq_length
        self.chunk_size = chunk_size
        self.offload_device = offload_device
        self.active_cache = None
        self.offloaded_chunks = []
        
    def init_cache(self, 
                  batch_size: int,
                  num_heads: int,
                  head_dim: int,
                  device: str = "cuda") -> KVCache:
        """Initialize empty KV cache"""
        shape = (batch_size, num_heads, self.max_seq_length, head_dim)
        self.active_cache = KVCache(
            keys=torch.zeros(shape, dtype=torch.float16, device=device),
            values=torch.zeros(shape, dtype=torch.float16, device=device)
        )
        return self.active_cache
    
    def update_cache(self, 
                    new_keys: torch.Tensor,
                    new_values: torch.Tensor,
                    positions: torch.Tensor) -> KVCache:
        """Update KV cache with new keys/values at specified positions"""
        if self.active_cache is None:
            raise ValueError("Cache not initialized")
            
        # Update the cache
        batch_idx = torch.arange(new_keys.size(0), device=new_keys.device)[:, None]
        head_idx = torch.arange(new_keys.size(1), device=new_keys.device)[None, :]
        
        self.active_cache.keys[batch_idx, head_idx, positions] = new_keys
        self.active_cache.values[batch_idx, head_idx, positions] = new_values
        self.active_cache.current_length = max(self.active_cache.current_length, positions.max() + 1)
        
        # Offload chunks if needed
        if (self.offload_device and 
            self.active_cache.current_length % self.chunk_size == 0):
            self._offload_chunk()
            
        return self.active_cache
    
    def _offload_chunk(self):
        """Offload a chunk of the cache to secondary device"""
        if self.active_cache.current_length <= self.chunk_size:
            return
            
        start_idx = len(self.offloaded_chunks) * self.chunk_size
        end_idx = start_idx + self.chunk_size
        
        chunk = KVCache(
            keys=self.active_cache.keys[..., start_idx:end_idx, :].to(self.offload_device),
            values=self.active_cache.values[..., start_idx:end_idx, :].to(self.offload_device),
            current_length=self.chunk_size
        )
        self.offloaded_chunks.append(chunk)
        
        # Free up space in active cache
        self.active_cache.keys[..., start_idx:end_idx, :] = 0
        self.active_cache.values[..., start_idx:end_idx, :] = 0
    
    def get_chunk(self, chunk_idx: int) -> KVCache:
        """Retrieve a specific chunk from offloaded memory"""
        if chunk_idx < len(self.offloaded_chunks):
            return self.offloaded_chunks[chunk_idx]
        return None
    
    def get_current_length(self) -> int:
        """Get total sequence length including offloaded chunks"""
        if not self.active_cache:
            return 0
        return len(self.offloaded_chunks) * self.chunk_size + self.active_cache.current_length
    
    def clear(self):
        """Clear all cache data"""
        self.active_cache = None
        self.offloaded_chunks = []
