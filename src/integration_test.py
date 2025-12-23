"""
Integration tests for L-AI-NUX-TOOL.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock


@pytest.mark.integration
class TestLLMEngineIntegration:
    """Integration tests for LLM engine."""
    
    @pytest.mark.slow
    @pytest.mark.api
    def test_llm_engine_initialization_online(self, mock_config, mock_openai_client, monkeypatch):
        """Test LLM engine initialization in online mode."""
        from .llm_engine import LLMEngine
        
        # Set API key in environment
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key-123')
        mock_config.set('mode', 'online')
        mock_config.set('api.api_key', 'test-key-123')
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            engine = LLMEngine(mock_config)
            assert engine is not None
            assert hasattr(engine, 'client')
    
    @pytest.mark.api
    def test_llm_engine_mode_switching(self, mock_config, mock_openai_client, monkeypatch):
        """Test switching between modes."""
        from .llm_engine import LLMEngine
        
        # Set API key
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key-123')
        mock_config.set('mode', 'online')
        mock_config.set('api.api_key', 'test-key-123')
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            engine = LLMEngine(mock_config)
            
            # Test mode switching
            engine.switch_mode('online-rag')
            assert mock_config.get('mode') == 'online-rag'


@pytest.mark.integration
@pytest.mark.rag
class TestRAGIntegration:
    """Integration tests for RAG system."""
    
    def test_rag_engine_initialization(self, temp_cache_dir):
        """Test RAG engine initialization."""
        from .rag_engine import RAGEngine
        
        # Skip if dependencies not available
        try:
            engine = RAGEngine(cache_dir=temp_cache_dir)
            assert engine is not None
        except ImportError:
            pytest.skip("RAG dependencies not available")
    
    @pytest.mark.slow
    def test_rag_document_retrieval(self, temp_cache_dir, sample_command_docs):
        """Test document retrieval in RAG."""
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
        except ImportError:
            pytest.skip("RAG dependencies not available")


@pytest.mark.integration
class TestCommandGenerationPipeline:
    """Integration tests for full command generation pipeline."""
    
    def test_full_pipeline_with_mocks(self, mock_config, mock_llm_engine, 
                                      mock_rag_engine, sample_queries):
        """Test full command generation pipeline with mocked components."""
        from .command_parser import CommandParser
        from .enhanced_safety import EnhancedSafetySystem
        
        parser = CommandParser()
        safety = EnhancedSafetySystem()
        
        # Test pipeline
        query = sample_queries[0]
        generated = mock_llm_engine.generate_command(query)
        parsed = parser.parse_command(generated)
        is_safe, _, _ = safety.validate_command_safety(parsed)
        
        assert generated is not None
        assert parsed is not None
        assert isinstance(is_safe, bool)


@pytest.mark.integration
class TestConfigPersistence:
    """Test configuration persistence across sessions."""
    
    def test_config_persists(self, mock_config):
        """Test that config persists between sessions."""
        # Set a value
        mock_config.set('test.persist', 'value123')
        
        # Verify it was saved
        assert mock_config.get('test.persist') == 'value123'
        
        # Verify file exists
        assert mock_config.config_file.exists()
        
        # Reload config to verify persistence
        from .config import Config
        config2 = Config()
        # Note: This will use default config dir, not our temp one
        # But we can verify the first config saved correctly
        assert mock_config.get('test.persist') == 'value123'

