import pytest
from core.retriever import HybridRetriever
from core.memory import VectorMemory
from unittest.mock import MagicMock, patch

class TestRAGPipeline:
    @pytest.fixture
    def mock_embedding_model(self):
        model = MagicMock()
        model.encode.return_value = [[0.1, 0.2, 0.3]]  # Simple mock embedding
        return model

    @pytest.fixture
    def retriever(self, mock_embedding_model):
        return HybridRetriever(
            dense_model=mock_embedding_model,
            sparse_model="bm25",
            memory=VectorMemory(index_name="test_index")
        )

    def test_hybrid_retrieval(self, retriever):
        # Mock document store
        retriever.memory.add_documents([
            {"text": "Test document 1", "metadata": {"source": "test"}},
            {"text": "Test document 2", "metadata": {"source": "test"}}
        ])
        
        results = retriever.retrieve("test query", top_k=2)
        assert len(results) == 2, "Should retrieve requested number of documents"
        assert all('text' in doc for doc in results), "Results should contain text"
        assert all('score' in doc for doc in results), "Results should have scores"

    def test_reranking(self, retriever):
        mock_results = [
            {"text": "doc1", "score": 0.5, "embedding": [0.1, 0.1, 0.1]},
            {"text": "doc2", "score": 0.8, "embedding": [0.9, 0.9, 0.9]}
        ]
        
        reranked = retriever._rerank_results("test query", mock_results)
        assert len(reranked) == 2, "Should return same number of results"
        assert reranked[0]['score'] >= reranked[1]['score'], "Should be sorted by score"

    @patch('core.retriever.ColBERTModel')
    def test_colbert_retrieval(self, mock_colbert, retriever):
        mock_colbert.return_value.encode.return_value = [[0.1, 0.2, 0.3]]
        
        # Enable ColBERT
        retriever.enable_colbert = True
        results = retriever.retrieve("test query", top_k=1)
        
        mock_colbert.return_value.encode.assert_called_once()
        assert len(results) == 1, "Should return single result when top_k=1"
