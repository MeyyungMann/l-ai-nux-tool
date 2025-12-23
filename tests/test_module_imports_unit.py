"""
Unit tests for module imports.
"""

import pytest


@pytest.mark.unit
class TestModuleImports:
    """Test that all core modules can be imported."""
    
    def test_config_import(self):
        """Test Config module import."""
        from src.config import Config
        assert Config is not None
    
    def test_command_parser_import(self):
        """Test CommandParser module import."""
        from src.command_parser import CommandParser
        assert CommandParser is not None
    
    def test_llm_engine_import(self):
        """Test LLMEngine module import."""
        from src.llm_engine import LLMEngine
        assert LLMEngine is not None
    
    def test_rag_engine_import(self):
        """Test RAGEngine module import."""
        try:
            from src.rag_engine import RAGEngine
            assert RAGEngine is not None
        except ImportError:
            pytest.skip("RAG dependencies not available")
    
    def test_doc_collector_import(self):
        """Test DocCollector module import."""
        from src.doc_collector import LinuxDocCollector, CommandDoc
        assert LinuxDocCollector is not None
        assert CommandDoc is not None
    
    def test_enhanced_safety_import(self):
        """Test EnhancedSafetySystem module import."""
        from src.enhanced_safety import EnhancedSafetySystem
        assert EnhancedSafetySystem is not None
    

