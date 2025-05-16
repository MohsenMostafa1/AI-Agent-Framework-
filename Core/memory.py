from typing import List, Dict, Any
from collections import deque
import numpy as np
from pydantic import BaseModel

class MemoryItem(BaseModel):
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    importance: float = 0.5

class AgentMemory:
    def __init__(self, 
                 short_term_capacity: int = 20,
                 long_term_retriever: Any = None):
        # Episodic memory (context window)
        self.episodic = []
        
        # Short-term memory (recent interactions)
        self.short_term = deque(maxlen=short_term_capacity)
        
        # Long-term memory (vector store)
        self.long_term = long_term_retriever
        
    def add(self, 
            content: str, 
            memory_type: str = "short_term",
            **metadata):
        """Add item to specified memory store"""
        item = MemoryItem(content=content, metadata=metadata)
        
        if memory_type == "episodic":
            self.episodic.append(item)
        elif memory_type == "short_term":
            self.short_term.append(item)
        elif memory_type == "long_term" and self.long_term:
            self.long_term.add_documents([item.content], [item.metadata])
        else:
            raise ValueError(f"Invalid memory type: {memory_type}")
            
    def retrieve(self, query: str, n_results: int = 3) -> List[MemoryItem]:
        """Search across all memory stores"""
        results = []
        
        # Check episodic memory
        results.extend(self._search_episodic(query))
        
        # Check short-term memory
        results.extend(self._search_short_term(query))
        
        # Check long-term memory if available
        if self.long_term:
            long_term_results = self.long_term.search(query, k=n_results)
            results.extend([MemoryItem(content=r[0], metadata=r[1]) for r in long_term_results])
            
        # Sort by relevance (simplified)
        return sorted(results, key=lambda x: -x.importance)[:n_results]
    
    # Additional helper methods omitted...
