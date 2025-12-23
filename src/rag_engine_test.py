"""
Unit tests for RAG engine.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock


@pytest.mark.unit
class TestRAGEngine:
    """Test RAG engine functionality."""
    
    def test_rag_engine_initialization(self, temp_cache_dir):
        """Test RAG engine initialization."""
        from .rag_engine import RAGEngine
        
        try:
            engine = RAGEngine(cache_dir=temp_cache_dir)
            assert engine is not None
            assert engine.cache_dir == temp_cache_dir
        except ImportError:
            pytest.skip("RAG dependencies not available")
    
    def test_rag_engine_retrieve_docs(self, temp_cache_dir, sample_command_docs):
        """Test document retrieval."""
        from .rag_engine import RAGEngine
        from .doc_collector import LinuxDocCollector
        
        try:
            # Setup: Save sample docs
            collector = LinuxDocCollector(temp_cache_dir)
            collector.save_docs(sample_command_docs)
            
            # Initialize RAG engine
            engine = RAGEngine(cache_dir=temp_cache_dir.parent / 'rag_cache')
            
            # Test retrieval
            results = engine.retrieve_relevant_docs("find files", top_k=2)
            assert isinstance(results, list)
            assert len(results) <= 2
        except ImportError:
            pytest.skip("RAG dependencies not available")
    
    def test_rag_engine_get_stats(self, temp_cache_dir):
        """Test getting RAG engine statistics."""
        from .rag_engine import RAGEngine
        
        try:
            engine = RAGEngine(cache_dir=temp_cache_dir)
            stats = engine.get_stats()
            assert isinstance(stats, dict)
            assert 'total_documents' in stats
        except ImportError:
            pytest.skip("RAG dependencies not available")
    
    def test_rag_engine_generate_context(self, temp_cache_dir, sample_command_docs):
        """Test context generation."""
        from .rag_engine import RAGEngine
        from .doc_collector import LinuxDocCollector
        
        try:
            # Setup: RAG engine looks for docs in cache_dir.parent / 'doc_cache'
            # So we need to save docs there
            doc_cache_dir = temp_cache_dir.parent / 'doc_cache'
            doc_cache_dir.mkdir(parents=True, exist_ok=True)
            collector = LinuxDocCollector(doc_cache_dir)
            collector.save_docs(sample_command_docs)
            
            # Initialize RAG engine - it will look in cache_dir.parent / 'doc_cache'
            engine = RAGEngine(cache_dir=temp_cache_dir)
            
            # Test context generation
            context = engine.get_context_for_query("list files", top_k=2)
            assert isinstance(context, str)
            # Context might be empty if no matching docs, so just check it's a string
            assert isinstance(context, str)
        except ImportError:
            pytest.skip("RAG dependencies not available")
    
    def test_rag_engine_missing_dependencies(self, temp_cache_dir):
        """Test RAG engine behavior when dependencies are missing."""
        from .rag_engine import RAGEngine
        
        with patch('src.rag_engine.SentenceTransformer', None):
            with patch('src.rag_engine.faiss', None):
                # Should still initialize but not work
                engine = RAGEngine(cache_dir=temp_cache_dir)
                assert engine is not None
                # Retrieval should fail gracefully
                results = engine.retrieve_relevant_docs("test", top_k=1)
                assert results == []

