"""
Model management for downloading and caching models.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm
import torch

class ModelManager:
    """Manages model downloads and caching."""
    
    def __init__(self, config):
        self.config = config
        self.models_dir = config.models_dir
        self.cache_dir = config.cache_dir
        
        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
    
    def download_base_model(self, model_name: str) -> Path:
        """Download base model from Hugging Face."""
        model_path = self.models_dir / model_name.replace('/', '_')
        
        if model_path.exists():
            print(f"Model {model_name} already exists at {model_path}")
            return model_path
        
        print(f"Downloading base model: {model_name}")
        
        try:
            # Download model files
            downloaded_path = snapshot_download(
                repo_id=model_name,
                cache_dir=self.cache_dir,
                local_dir=model_path,
                resume_download=True
            )
            
            print(f"Model downloaded successfully to {downloaded_path}")
            return Path(downloaded_path)
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    
    def download_lora_weights(self, lora_name: str) -> Path:
        """Download LoRA weights."""
        lora_path = self.models_dir / f"lora_{lora_name.replace('/', '_')}"
        
        if lora_path.exists():
            print(f"LoRA weights already exist at {lora_path}")
            return lora_path
        
        print(f"Downloading LoRA weights: {lora_name}")
        
        try:
            downloaded_path = snapshot_download(
                repo_id=lora_name,
                cache_dir=self.cache_dir,
                local_dir=lora_path,
                resume_download=True
            )
            
            print(f"LoRA weights downloaded successfully to {downloaded_path}")
            return Path(downloaded_path)
            
        except Exception as e:
            print(f"Error downloading LoRA weights: {e}")
            raise
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to downloaded model."""
        model_path = self.models_dir / model_name.replace('/', '_')
        return model_path if model_path.exists() else None
    
    def get_lora_path(self, lora_name: str) -> Optional[Path]:
        """Get path to downloaded LoRA weights."""
        lora_path = self.models_dir / f"lora_{lora_name.replace('/', '_')}"
        return lora_path if lora_path.exists() else None
    
    def cleanup_cache(self):
        """Clean up cache directory."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            print("Cache cleaned up")
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        model_path = self.get_model_path(model_name)
        
        if not model_path:
            return {"exists": False}
        
        # Get model size
        total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
        
        return {
            "exists": True,
            "path": str(model_path),
            "size_mb": total_size / (1024 * 1024),
            "files": len(list(model_path.rglob('*')))
        }
    
    def check_disk_space(self) -> Dict[str, float]:
        """Check available disk space."""
        stat = shutil.disk_usage(self.models_dir)
        
        return {
            "total_gb": stat.total / (1024**3),
            "used_gb": (stat.total - stat.free) / (1024**3),
            "free_gb": stat.free / (1024**3)
        }
    
    def estimate_download_size(self, model_name: str) -> float:
        """Estimate download size for a model."""
        # Rough estimates for common models
        size_estimates = {
            "microsoft/DialoGPT-medium": 350,  # MB
            "microsoft/DialoGPT-large": 800,   # MB
            "codellama/CodeLlama-7b-Instruct": 7000,  # MB
            "microsoft/CodeBERT-base": 500,     # MB
        }
        
        return size_estimates.get(model_name, 1000)  # Default 1GB
    
    def download_if_needed(self, model_name: str, lora_name: Optional[str] = None):
        """Download model and LoRA if not already present."""
        # Check if model exists
        if not self.get_model_path(model_name):
            self.download_base_model(model_name)
        
        # Check if LoRA exists
        if lora_name and not self.get_lora_path(lora_name):
            self.download_lora_weights(lora_name)
        
        print("All required models are available")
