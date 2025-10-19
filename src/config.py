"""
Configuration management for the Linux AI Command Generator.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Configuration manager for the application."""
    
    def __init__(self):
        self.config_dir = Path.home() / '.lai-nux-tool'
        self.config_file = self.config_dir / 'config.yaml'
        self.models_dir = self.config_dir / 'models'
        self.cache_dir = self.config_dir / 'cache'
        
        # Create directories if they don't exist
        self.config_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        default_config = {
            'mode': 'online',  # Default mode: 'offline' or 'online'
            'model': {
                'base_model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                'lora_model': 'lai-nux-tool/lora-command-generator',
                'device': 'cuda' if self._has_cuda() else 'cpu',
                'max_length': 256,  # Reduced for faster generation
                'temperature': 0.1,   # Very low for fastest, most deterministic output
                'top_p': 0.8,        # Reduced for faster generation
                'use_4bit': True,  # Note: Code uses 4-bit quantization on CUDA (see llm_engine.py)
                'use_flash_attention': True
            },
            'api': {
                'base_url': 'https://api.openai.com/v1',
                'model': 'gpt-5-mini',
                'max_tokens': 512,
                'temperature': 0.1
            },
            'man': {
                'cache_man_pages': True,
                'cache_duration': 3600,  # 1 hour
                'max_depth': 7
            },
            'ui': {
                'theme': 'default',
                'show_confidence': True,
                'auto_execute': False
            },
        'ollama': {
            'base_url': 'http://host.docker.internal:11434',
            'model': 'gpt-oss:20b',
            'timeout': 120
        },
            'auto_save_quantized': True  # Auto-save quantized cache after first run
        }
        
        self.save_config(default_config)
        return default_config
    
    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config(self.config)
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get('model', {})
    
    @property
    def api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.get('api', {})
    
    @property
    def man_config(self) -> Dict[str, Any]:
        """Get man page configuration."""
        return self.get('man', {})
    
    @property
    def ui_config(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return self.get('ui', {})
    
    @property
    def ollama_config(self) -> Dict[str, Any]:
        """Get Ollama configuration."""
        return self.get('ollama', {})


