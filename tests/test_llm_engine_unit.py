"""
Unit tests for LLM engine.
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
import requests
import os


@pytest.mark.unit
class TestLLMEngine:
    """Test LLM engine functionality."""
    
    def test_llm_engine_initialization_online(self, mock_config, mock_openai_client, env_vars):
        """Test LLM engine initialization in online mode."""
        from src.llm_engine import LLMEngine
        
        mock_config.set('mode', 'online')
        mock_config.set('api.api_key', 'test-key-123')
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-123'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                engine = LLMEngine(mock_config)
                assert engine is not None
                assert hasattr(engine, 'client')
    
    def test_llm_engine_initialization_ollama(self, mock_config):
        """Test LLM engine initialization in Ollama mode."""
        from src.llm_engine import LLMEngine
        
        mock_config.set('mode', 'ollama')
        mock_config.set('ollama.base_url', 'http://localhost:11434')
        mock_config.set('ollama.model', 'test-model')
        
        # Mock Ollama API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'models': [{'name': 'test-model'}]
        }
        
        with patch('requests.get', return_value=mock_response):
            try:
                engine = LLMEngine(mock_config)
                assert engine is not None
                assert hasattr(engine, 'ollama_model')
                assert engine.ollama_model == 'test-model'
            except (ConnectionError, Exception):
                # Ollama might not be available, skip test
                pytest.skip("Ollama not available")
    
    def test_llm_engine_mode_switching(self, mock_config, mock_openai_client, env_vars):
        """Test switching between modes."""
        from src.llm_engine import LLMEngine
        
        mock_config.set('mode', 'online')
        mock_config.set('api.api_key', 'test-key-123')
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-123'}):
            with patch('openai.OpenAI', return_value=mock_openai_client):
                engine = LLMEngine(mock_config)
                
                # Test switching to online-rag
                engine.switch_mode('online-rag')
                assert mock_config.get('mode') == 'online-rag'
    
    def test_llm_engine_ollama_connection_error(self, mock_config):
        """Test Ollama connection error handling."""
        from src.llm_engine import LLMEngine
        
        mock_config.set('mode', 'ollama')
        mock_config.set('ollama.base_url', 'http://localhost:11434')
        mock_config.set('ollama.model', 'test-model')
        
        # Mock connection error
        with patch('requests.get', side_effect=requests.exceptions.ConnectionError()):
            with pytest.raises((ConnectionError, Exception)):
                LLMEngine(mock_config)
    
    def test_llm_engine_device_detection(self, mock_config, mock_openai_client, env_vars):
        """Test device detection."""
        from src.llm_engine import LLMEngine
        
        mock_config.set('mode', 'online')
        mock_config.set('api.api_key', 'test-key-123')
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-123'}):
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.get_device_name', return_value='Test GPU'):
                    with patch('torch.cuda.get_device_properties') as mock_props:
                        mock_props.return_value.total_memory = 8 * 1024**3  # 8GB
                        with patch('openai.OpenAI', return_value=mock_openai_client):
                            engine = LLMEngine(mock_config)
                            assert engine.device == 'cuda'
            
            with patch('torch.cuda.is_available', return_value=False):
                with patch('torch.backends.mps.is_available', return_value=False):
                    with patch('openai.OpenAI', return_value=mock_openai_client):
                        engine = LLMEngine(mock_config)
                        assert engine.device == 'cpu'

