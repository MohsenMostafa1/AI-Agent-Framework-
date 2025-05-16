import torch
import numpy as np
from typing import List, Dict, Tuple
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig

class HybridRetriever:
    def __init__(self,
                 colbert_config: Dict[str, Any],
                 dense_retriever: Optional[Any] = None,
                 sparse_retriever: Optional[Any] = None):
        self.colbert_config = colbert_config
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self._init_colbert()
        
    def _init_colbert(self):
        """Initialize ColBERT components"""
        with Run().context(RunConfig(nranks=1, experiment="hybrid_retriever")):
            self.indexer = Indexer(
                checkpoint=self.colbert_config["checkpoint"],
                index_root=self.colbert_config["index_root"]
            )
            self.searcher = Searcher(
                index=self.colbert_config["index_name"],
                checkpoint=self.colbert_config["checkpoint"]
            )
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to all available retrievers"""
        metadata = metadata or [{}] * len(documents)
        
        # Add to ColBERT
        with Run().context(RunConfig(nranks=1)):
            self.indexer.index(
                name=self.colbert_config["index_name"],
                collection=documents,
                metadata=metadata
            )
        
        # Add to dense retriever if available
        if self.dense_retriever:
            self.dense_retriever.add_documents(documents, metadata)
            
        # Add to sparse retriever if available
        if self.sparse_retriever:
            self.sparse_retriever.add_documents(documents, metadata)
    
    def search(self, 
              query: str, 
              k: int = 5,
              hybrid_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)) -> List[Tuple[str, Dict]]:
        """Hybrid search combining ColBERT, dense, and sparse results"""
        results = []
        
        # Get ColBERT results
        colbert_results = self.searcher.search(query, k=k)
        colbert_scores = [r.score for r in colbert_results]
        colbert_max = max(colbert_scores) if colbert_scores else 1.0
        
        # Get dense results if available
        dense_results = []
        if self.dense_retriever:
            dense_results = self.dense_retriever.search(query, k=k)
            dense_scores = [r[1]["score"] for r in dense_results]
            dense_max = max(dense_scores) if dense_scores else 1.0
        
        # Get sparse results if available
        sparse_results = []
        if self.sparse_retriever:
            sparse_results = self.sparse_retriever.search(query, k=k)
            sparse_scores = [r[1]["score"] for r in sparse_results]
            sparse_max = max(sparse_scores) if sparse_scores else 1.0
        
        # Combine results
        combined = {}
        for i, r in enumerate(colbert_results):
            doc_id = r.docid
            normalized_score = r.score / colbert_max * hybrid_weights[0]
            combined[doc_id] = {
                "text": r.text,
                "score": normalized_score,
                "metadata": r.metadata,
                "components": {"colbert": r.score}
            }
        
        # Add dense results
        if self.dense_retriever:
            for text, meta in dense_results:
                doc_id = meta.get("doc_id", hash(text))
                normalized_score = meta["score"] / dense_max * hybrid_weights[1]
                if doc_id in combined:
                    combined[doc_id]["score"] += normalized_score
                    combined[doc_id]["components"]["dense"] = meta["score"]
                else:
                    combined[doc_id] = {
                        "text": text,
                        "score": normalized_score,
                        "metadata": meta,
                        "components": {"dense": meta["score"]}
                    }
        
        # Add sparse results
        if self.sparse_retriever:
            for text, meta in sparse_results:
                doc_id = meta.get("doc_id", hash(text))
                normalized_score = meta["score"] / sparse_max * hybrid_weights[2]
                if doc_id in combined:
                    combined[doc_id]["score"] += normalized_score
                    combined[doc_id]["components"]["sparse"] = meta["score"]
                else:
                    combined[doc_id] = {
                        "text": text,
                        "score": normalized_score,
                        "metadata": meta,
                        "components": {"sparse": meta["score"]}
                    }
        
        # Sort by combined score
        sorted_results = sorted(
            combined.items(),
            key=lambda x: -x[1]["score"]
        )[:k]
        
        return [(item[1]["text"], item[1]) for item in sorted_results]
