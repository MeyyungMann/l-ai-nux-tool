"""
Unit tests for configuration management.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open
from .config import Config


@pytest.mark.unit
class TestConfig:
    """Test configuration management."""
    
    def test_config_initialization(self, mock_config):
        """Test config initialization."""
        assert mock_config is not None
        assert mock_config.config_dir is not None
        assert mock_config.config_file is not None
    
    def test_config_get_set(self, mock_config):
        """Test getting and setting config values."""
        mock_config.set('test.key', 'value')
        assert mock_config.get('test.key') == 'value'
    
    def test_config_get_default(self, mock_config):
        """Test getting default values."""
        value = mock_config.get('nonexistent.key', 'default')
        assert value == 'default'
    
    def test_config_nested_keys(self, mock_config):
        """Test nested key access."""
        mock_config.set('model.base_model', 'test-model')
        assert mock_config.get('model.base_model') == 'test-model'
    
    def test_config_properties(self, mock_config):
        """Test config property accessors."""
        model_config = mock_config.model_config
        assert isinstance(model_config, dict)
        
        api_config = mock_config.api_config
        assert isinstance(api_config, dict)
    
    def test_config_save_load(self, mock_config):
        """Test saving and loading config."""
        # Set a value
        mock_config.set('test.value', 'saved')
        
        # Verify it was saved
        assert mock_config.get('test.value') == 'saved'
        
        # Verify file exists
        assert mock_config.config_file.exists()
    
    def test_config_cuda_detection(self):
        """Test CUDA availability detection."""
        with patch('torch.cuda.is_available', return_value=True):
            config = Config()
            assert config._has_cuda() is True
        
        with patch('torch.cuda.is_available', return_value=False):
            config = Config()
            assert config._has_cuda() is False

