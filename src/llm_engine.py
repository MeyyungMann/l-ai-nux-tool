"""
LLM Engine for handling both offline and online inference.
"""

import os
import time
import torch
import pickle
import hashlib
from typing import Optional, Dict, Any, List
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
# from peft import PeftModel  # Not using LoRA in this project
import openai
from huggingface_hub import hf_hub_download
from pathlib import Path

from .config import Config

class LLMEngine:
    """LLM Engine for command generation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = self._get_device()
        self.rag_engine = None
        # Pending RAG seed from online fallback (description, repaired_command)
        self._pending_rag_seed = None
        
        # Setup model cache directory
        self.cache_dir = Path.home() / '.lai-nux-tool' / 'model_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize based on mode
        mode = self.config.get('mode', 'offline')
        if mode == 'offline':
            self._load_offline_model()
        elif mode == 'rag':
            self._load_offline_model()
            self._setup_rag_engine()
        elif mode == 'online-rag':
            self._setup_online_client()
            self._setup_rag_engine()
        else:
            self._setup_online_client()
    
    def _get_device(self) -> str:
        """Get the appropriate device for inference."""
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 Using GPU: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            print(f"🔧 Device: {device.upper()}")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print(f"🍎 Using Apple Silicon GPU (MPS)")
            print(f"🔧 Device: {device.upper()}")
        else:
            device = 'cpu'
            print(f"💻 Using CPU")
            print(f"🔧 Device: {device.upper()}")
        
        return device
    
    # LoRA methods removed - not using LoRA in this project
    
    def _get_model_cache_key(self) -> str:
        """Generate a cache key for the current model configuration."""
        model_config = self.config.model_config
        base_model = model_config.get('base_model', 'mistralai/Mixtral-8x7B-Instruct-v0.1')
        # No LoRA path needed - using base model only
        
        # Create hash of model configuration
        config_str = f"{base_model}_{self.device}"
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _validate_hf_cache(self) -> bool:
        """Validate HuggingFace cache integrity for the model."""
        try:
            from huggingface_hub import snapshot_download
            import os
            
            model_config = self.config.model_config
            base_model = model_config.get('base_model', 'mistralai/Mixtral-8x7B-Instruct-v0.1')
            
            # Check if model directory exists in cache
            cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
            # Convert model name to cache directory format
            model_cache_name = base_model.replace('/', '--')
            model_dir = cache_dir / f'models--{model_cache_name}'
            
            if not model_dir.exists():
                print("❌ HuggingFace model cache directory not found")
                return False
            
            # Check for essential model files
            essential_files = [
                'config.json',
                'tokenizer.json',
                'tokenizer_config.json'
            ]
            
            for file_name in essential_files:
                file_path = model_dir / 'snapshots' / '*' / file_name
                import glob
                if not glob.glob(str(file_path)):
                    print(f"❌ Missing essential file: {file_name}")
                    return False
            
            # Check for model shards
            shard_pattern = model_dir / 'snapshots' / '*' / '*.safetensors'
            shard_files = glob.glob(str(shard_pattern))
            if not shard_files:
                print("❌ No model shard files found")
                return False
            
            print(f"✅ HuggingFace cache validation passed ({len(shard_files)} shards found)")
            return True
            
        except Exception as e:
            print(f"❌ HuggingFace cache validation failed: {e}")
            return False
    
    def _clear_corrupted_hf_cache(self):
        """Clear corrupted HuggingFace cache."""
        try:
            import shutil
            
            cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
            # Convert model name to cache directory format (e.g., mistralai/Mixtral-8x7B-Instruct-v0.1 -> models--mistralai--Mixtral-8x7B-Instruct-v0.1)
            base_model = self.config.model_config.get('base_model', 'mistralai/Mixtral-8x7B-Instruct-v0.1')
            model_cache_name = base_model.replace('/', '--')
            model_dir = cache_dir / f'models--{model_cache_name}'
            
            if model_dir.exists():
                shutil.rmtree(model_dir)
                print("🗑️  Cleared corrupted HuggingFace cache")
            
            # Also clear our custom cache
            cache_key = self._get_model_cache_key()
            cache_file = self.cache_dir / f"model_{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
                print("🗑️  Cleared corrupted custom cache")
                
        except Exception as e:
            print(f"⚠️  Failed to clear corrupted cache: {e}")
    
    def _auto_save_quantized_cache(self, model_name: str):
        """Automatically save quantized cache after first successful load."""
        try:
            # Check if we're on CUDA (quantization only works on GPU)
            if self.device != 'cuda':
                return
            
            # Check if quantized cache already exists
            quantized_cache_dir = Path("./quantized_model_cache")
            if (quantized_cache_dir / "config.json").exists():
                # Cache already exists, nothing to do
                return
            
            # Check if we've already attempted auto-save (marker file)
            marker_file = self.cache_dir / '.quantized_auto_save_attempted'
            if marker_file.exists():
                # Already tried, don't spam user
                return
            
            # Create marker file to indicate we're attempting this
            marker_file.touch()
            
            # Check if auto-save is enabled (default: yes)
            auto_save = self.config.get('auto_save_quantized', True)
            if not auto_save:
                return
            
            print()
            print("="*70)
            print("🚀 AUTO-SAVING QUANTIZED CACHE")
            print("="*70)
            print()
            print("📌 First successful model load detected!")
            print("💡 Saving quantized cache for faster future startups...")
            print("⏱️  This takes 3-5 minutes but only happens once.")
            print()
            print("💾 Benefits: Faster loading, consistent 4-bit quantization")
            print("🎯 To disable: Set 'auto_save_quantized: false' in config")
            print()
            
            # Import and run the save function
            from save_quantized_model import save_quantized_model_auto
            
            # Save in foreground (user sees progress)
            success = save_quantized_model_auto(model_name, quantized_cache_dir)
            
            if success:
                print()
                print("="*70)
                print("✅ QUANTIZED CACHE SAVED SUCCESSFULLY!")
                print("="*70)
                print("🚀 Next startup will be faster!")
                print()
            else:
                print()
                print("⚠️  Quantized cache save failed - will retry next time")
                print("💡 You can manually run: python save_quantized_model.py")
                print()
                # Remove marker so we try again next time
                if marker_file.exists():
                    marker_file.unlink()
                    
        except Exception as e:
            print(f"⚠️  Auto-save quantized cache failed: {e}")
            print("💡 You can manually run: python save_quantized_model.py")
            # Don't crash, just continue
    
    def _save_model_to_cache(self):
        """Save the loaded model to cache for faster loading."""
        try:
            cache_key = self._get_model_cache_key()
            cache_file = self.cache_dir / f"model_{cache_key}.pkl"
            
            if self.model is not None and self.tokenizer is not None:
                print(f"💾 Saving model configuration to cache: {cache_file}")
                
                # Save model configuration and metadata (safer approach)
                cache_data = {
                    'model_state': self.model.state_dict(),
                    'tokenizer': self.tokenizer,
                    'model_config': self.config.model_config,
                    'device': self.device,
                    'cache_key': cache_key,
                    'model_loaded': True,
                    'pipeline_ready': self.pipeline is not None,
                    'cache_version': '2.1',  # Updated version
                    'model_type': 'codellama',
                    'quantization': '4bit' if self.device == 'cuda' else 'none',
                    'lora_loaded': False,  # Not using LoRA
                    'cache_timestamp': time.time()
                }
                
                # Use safer pickle protocol
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                print(f"✅ Model configuration cached successfully!")
                return True
        except Exception as e:
            print(f"⚠️  Failed to cache model: {e}")
            # Fallback to legacy method
            return self._save_model_to_cache_legacy()
    
    def _save_model_to_cache_legacy(self):
        """Legacy cache saving method (fallback)."""
        try:
            cache_key = self._get_model_cache_key()
            cache_file = self.cache_dir / f"model_{cache_key}_legacy.pkl"
            
            if self.model is not None and self.tokenizer is not None:
                print(f"💾 Saving model state to cache (legacy): {cache_file}")
                
                cache_data = {
                    'model_state': self.model.state_dict(),
                    'tokenizer': self.tokenizer,
                    'model_config': self.config.model_config,
                    'device': self.device,
                    'cache_key': cache_key,
                    'model_loaded': True,
                    'pipeline_ready': self.pipeline is not None,
                    'cache_version': '1.0'
                }
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                print(f"✅ Model state cached successfully!")
                return True
        except Exception as e:
            print(f"⚠️  Failed to cache model (legacy): {e}")
            return False
    
    def _load_model_from_cache(self) -> bool:
        """Load model from cache if available."""
        try:
            cache_key = self._get_model_cache_key()
            cache_file = self.cache_dir / f"model_{cache_key}.pkl"
            
            if cache_file.exists():
                print(f"🔄 Loading complete model from cache: {cache_file}")
                
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Verify cache is compatible
                if (cache_data.get('cache_key') == cache_key and 
                    cache_data.get('device') == self.device):
                    
                    cache_version = cache_data.get('cache_version', '1.0')
                    
                    if cache_version in ['2.0', '2.1']:
                        # Check if HuggingFace cache is valid before attempting smart loading
                        print("🔍 Validating HuggingFace cache integrity...")
                        if self._validate_hf_cache():
                            print("🚀 Smart cache loading disabled - using legacy method...")
                            print("🔄 Using legacy cache method for compatibility...")
                            return self._load_model_from_cache_legacy(cache_data)
                        else:
                            print("⚠️  HuggingFace cache is corrupted or incomplete")
                            print("🗑️  Clearing corrupted cache...")
                            self._clear_corrupted_hf_cache()
                            print("🔄 Falling back to fresh model download...")
                            return False
                    
                    else:
                        # Legacy cache format - fallback to old method
                        print("🔄 Loading legacy cache format...")
                        return self._load_model_from_cache_legacy(cache_data)
                        
                else:
                    print(f"⚠️  Cache incompatible, will reload model")
                    return False
            else:
                print(f"ℹ️  No cache found, will load model normally")
                return False
                
        except Exception as e:
            print(f"⚠️  Failed to load from cache: {e}")
            return False
    
    def _load_model_from_cache_legacy(self, cache_data) -> bool:
        """Load model from legacy cache format.
        
        NOTE: Legacy cache only stores state_dict, not full model.
        We need to load base model architecture first, then apply cached weights.
        Uses HuggingFace cache (local_files_only=True) to avoid re-downloading.
        """
        try:
            print("📦 Loading model architecture from HuggingFace cache...")
            
            # Load tokenizer from our cache
            self.tokenizer = cache_data['tokenizer']
            
            # Load base model architecture (uses HF cache, doesn't download)
            model_config = self.config.model_config
            base_model_name = model_config.get('base_model', 'mistralai/Mixtral-8x7B-Instruct-v0.1')
            
            if self.device == 'cuda':
                # Use 4-bit quantization for RTX 5090 (faster loading, reliable saving)
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True  # Enable for Docker compatibility
                )
                
                # Load from HuggingFace cache only (faster)
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    local_files_only=True  # Use HF cache, don't download
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    dtype=torch.float32,
                    local_files_only=True  # Use HF cache, don't download
                )
            
            # Restore trained weights from our cache
            print("🔄 Restoring trained weights from cache...")
            # Suppress verbose output during state dict loading
            import logging
            logging.getLogger().setLevel(logging.ERROR)
            self.model.load_state_dict(cache_data['model_state'], strict=False)
            logging.getLogger().setLevel(logging.INFO)
            
            # Using base model only (no LoRA weights)
            
            # Create pipeline
            pipeline_kwargs = {
                "task": "text-generation",
                "model": self.model,
                "tokenizer": self.tokenizer,
                "max_length": model_config.get('max_length', 512),
                "temperature": model_config.get('temperature', 0.1),
                "top_p": model_config.get('top_p', 0.9),
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            if self.device != 'cuda' or not model_config.get('use_4bit', False):
                pipeline_kwargs['device'] = self.device
            
            self.pipeline = pipeline(**pipeline_kwargs)
            
            print(f"✅ Model loaded from legacy cache successfully!")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load from legacy cache: {e}")
            return False
    
    # Single-file cache merging functions removed - using 5-shard cache directly
    
    # Single-file cache creation disabled - using 5-shard cache directly
    
    def _load_model_from_cache_smart(self, cache_data) -> bool:
        """Smart cache loading that uses HuggingFace cache + applies state."""
        try:
            print("📦 Loading base model from HuggingFace cache...")
            
            # Load tokenizer from cache
            self.tokenizer = cache_data['tokenizer']
            
            # Load model from HuggingFace cache (much faster than download)
            model_config = self.config.model_config
            base_model_name = model_config.get('base_model', 'mistralai/Mixtral-8x7B-Instruct-v0.1')
            
            # Use HuggingFace cache directory for faster loading
            cache_dir = Path.home() / '.cache' / 'huggingface'
            
            if self.device == 'cuda':
                # Use 4-bit quantization for RTX 5090 (faster loading, reliable saving)
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True  # Enable for Docker compatibility
                )
                
                # Load from HuggingFace cache (much faster)
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    local_files_only=True  # Only use cached files
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    dtype=torch.float32,
                    local_files_only=True
                )
            
            # Apply cached model state
            print("🔄 Applying cached model state...")
            self.model.load_state_dict(cache_data['model_state'])
            
            # Using base model only (no LoRA weights)
            
            # Create pipeline
            pipeline_kwargs = {
                "task": "text-generation",
                "model": self.model,
                "tokenizer": self.tokenizer,
                "max_length": model_config.get('max_length', 512),
                "temperature": model_config.get('temperature', 0.1),
                "top_p": model_config.get('top_p', 0.9),
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            if self.device != 'cuda' or not model_config.get('use_4bit', False):
                pipeline_kwargs['device'] = self.device
            
            self.pipeline = pipeline(**pipeline_kwargs)
            
            print(f"✅ Model loaded from smart cache successfully!")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load from smart cache: {e}")
            print("🔄 Falling back to normal loading...")
            return False
    
    def _load_offline_model(self):
        """Load the offline model with caching."""
        try:
            model_config = self.config.model_config
            base_model_name = model_config.get('base_model', 'mistralai/Mixtral-8x7B-Instruct-v0.1')
            
            # Check for pre-quantized model cache first (FASTEST option!)
            quantized_cache_dir = Path("./quantized_model_cache")
            single_file_cache_dir = Path("./quantized_model_cache_single")
            
            # Skip single-file cache and go directly to 5-shard cache
            print("🚀 Loading directly from 5-shard quantized model cache...")
            
            # Load from 5-shard cache (current working method)
            if quantized_cache_dir.exists() and (quantized_cache_dir / "config.json").exists():
                print("🚀 Found pre-quantized model cache!")
                print(f"📍 Loading from: {quantized_cache_dir.absolute()}")
                
                try:
                    # Load tokenizer from quantized cache
                    self.tokenizer = AutoTokenizer.from_pretrained(quantized_cache_dir)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # Load pre-quantized model with proper quantization config
                    from transformers import BitsAndBytesConfig
                    
                    # Configure 4-bit quantization for loading
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        quantized_cache_dir,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                    
                    print("✅ Pre-quantized model loaded successfully!")
                    print("🎯 Ready for command generation!")
                    
                    # Create pipeline
                    pipeline_kwargs = {
                        "task": "text-generation",
                        "model": self.model,
                        "tokenizer": self.tokenizer,
                        "max_length": model_config.get('max_length', 512),
                        "temperature": model_config.get('temperature', 0.1),
                        "top_p": model_config.get('top_p', 0.9),
                        "do_sample": True,
                        "pad_token_id": self.tokenizer.eos_token_id
                    }
                    
                    self.pipeline = pipeline(**pipeline_kwargs)
                    print(f"💡 Model: Pre-quantized {base_model_name}")
                    
                    # Single-file cache creation disabled - using 5-shard cache directly
                    
                    return  # Early return - we're done!
                    
                except Exception as e:
                    print(f"⚠️  Failed to load pre-quantized cache: {e}")
                    print("🔄 Falling back to normal loading...")
            else:
                print("ℹ️  No pre-quantized model found")
                print(f"💡 TIP: Run 'python save_quantized_model.py' once to save a pre-quantized model")
                print(f"📂 This will create cache in: {quantized_cache_dir.absolute()}")
                print("⚡ Next time it will load 4x faster!")
                print()
            
            # Normal loading path (if no quantized cache or it failed)
            # Using base model only - no LoRA weights
            
            print(f"Loading model: {base_model_name}")
            print("Using 4-bit quantization for faster loading and reliable caching")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            if self.device == 'cuda':
                print("🔧 Configuring 4-bit quantization for GPU...")
                # Use 4-bit quantization for RTX 5090 (faster loading, reliable saving)
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True  # Enable for Docker compatibility
                )
                
                print("📦 Loading model with quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                print("✅ Model loaded with 4-bit quantization on GPU")
            else:
                print("📦 Loading model on CPU (no quantization)...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    dtype=torch.float32
                )
                print("✅ Model loaded on CPU")
            
            # Using base model only (no LoRA weights)
            
            # Create pipeline with conditional device argument
            pipeline_kwargs = {
                "task": "text-generation",
                "model": self.model,
                "tokenizer": self.tokenizer,
                "max_length": model_config.get('max_length', 512),
                "temperature": model_config.get('temperature', 0.1),
                "top_p": model_config.get('top_p', 0.9),
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # Only add device if not using device_map="auto" (which conflicts with explicit device)
            if self.device != 'cuda' or not model_config.get('use_4bit', False):
                pipeline_kwargs['device'] = self.device
            
            self.pipeline = pipeline(**pipeline_kwargs)
            
            print(f"Offline model loaded successfully on {self.device}")
            print(f"🎯 Ready for command generation!")
            print(f"💡 Model: {base_model_name}")
            print(f"🔧 Quantization: {'4-bit (GPU)' if self.device == 'cuda' else 'None (CPU)'}")
            
            # Save model to cache for faster loading next time
            self._save_model_to_cache()
            
            # Auto-save quantized cache after first successful load
            self._auto_save_quantized_cache(base_model_name)
            
        except Exception as e:
            print(f"Error loading offline model: {e}")
            print("Falling back to online mode...")
            self._setup_online_client()
    
    def _setup_online_client(self):
        """Setup OpenAI-compatible client for online inference."""
        api_config = self.config.api_config
        base_url = api_config.get('base_url', 'https://api.openai.com/v1')
        api_key = os.getenv('OPENAI_API_KEY') or api_config.get('api_key')
        
        if not api_key:
            raise ValueError("API key required for online mode")
        
        try:
            # Try newer OpenAI library format (v1.0+)
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        except TypeError:
            # Fallback for older versions
            openai.api_key = api_key
            openai.api_base = base_url
            self.client = openai
        
        print("Online client setup successfully")
    
    def _setup_rag_engine(self):
        """Setup RAG engine for retrieval-augmented generation."""
        try:
            from .rag_engine import get_rag_engine
            print("🔍 Initializing RAG system...")
            self.rag_engine = get_rag_engine()
            print("✅ RAG system initialized")
            
            # Print stats
            stats = self.rag_engine.get_stats()
            print(f"📚 Indexed {stats['total_documents']} Linux command documents")
            
        except ImportError as e:
            print(f"⚠️  RAG dependencies not available: {e}")
            print("Install with: pip install sentence-transformers faiss-cpu")
            self.rag_engine = None
        except Exception as e:
            print(f"⚠️  Failed to initialize RAG system: {e}")
            self.rag_engine = None
    
    def generate_command(self, description: str) -> str:
        """Generate a Linux command from natural language description."""
        mode = self.config.get('mode', 'offline')
        
        if mode == 'rag' and self.rag_engine:
            return self._generate_with_rag(description)
        elif mode == 'online-rag' and self.rag_engine:
            return self._generate_with_online_rag(description)
        elif self.pipeline:
            return self._generate_offline(description)
        else:
            return self._generate_online(description)
    
    def _generate_offline(self, description: str) -> str:
        """Generate command using offline model."""
        prompt = self._create_prompt(description)
        
        # Generate response (optimized for speed)
        response = self.pipeline(
            prompt,
            max_new_tokens=100,  # Reduced from 200 for faster generation
            num_return_sequences=1,
            temperature=0.1,     # Very low temperature for fastest, most deterministic output
            do_sample=False,     # Greedy decoding is faster than sampling
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = response[0]['generated_text']
        command = self._extract_command(generated_text, prompt)
        
        return command
    
    def _clean_rag_templates(self, context: str) -> str:
        """Clean template placeholders from RAG context to avoid confusion.
        
        Converts template examples like:
          head {{[-n|--lines]}} {{count}} {{path/to/file}}
        Into concrete examples like:
          head -n 10 file.txt
        
        PRESERVES known-good commands from cache.
        """
        import re
        
        # Don't clean known-good commands from cache
        if "🎯 Known-good command for this task:" in context:
            return context
        
        # Common template replacements
        replacements = {
            # File paths
            r'\{\{path/to/file\}\}': 'file.txt',
            r'\{\{filename\}\}': 'file.txt',
            r'\{\{file\}\}': 'file.txt',
            r'\{\{path/to/directory\}\}': 'mydir',
            r'\{\{directory\}\}': 'mydir',
            
            # Patterns
            r'\{\{\*\.ext\}\}': '*.txt',
            r'\{\{pattern\}\}': '*.txt',
            r'\{\{\*pattern\*\}\}': '*file*',
            
            # Numbers and counts
            r'\{\{count\}\}': '10',
            r'\{\{number\}\}': '10',
            r'\{\{n\}\}': '10',
            
            # Options (make concrete or remove)
            r'\{\{\[-n\|--lines\]\}\}': '-n',
            r'\{\{\[-c\|--bytes\]\}\}': '-c',
            r'\{\{\[-F\|--fixed-strings\]\}\}': '-F',
            r'\{\{\[options\]\}\}': '',
            
            # Search patterns
            r'\{\{search_pattern\}\}': 'text',
            r'\{\{exact_string\}\}': 'text',
            r'\{\{foo\}\}': 'pattern',
            
            # Paths
            r'\{\{root_path\}\}': '.',
            r'\{\{\*/path/\*/\*\.ext\}\}': '*.txt',
            r'\{\{\*/filename\}\}': '*/file.txt',
            
            # Generic catch-all for any remaining templates
            r'\{\{[^}]+\}\}': 'value',
        }
        
        cleaned = context
        for pattern, replacement in replacements.items():
            cleaned = re.sub(pattern, replacement, cleaned)
        
        return cleaned
    
    def _generate_with_rag(self, description: str) -> str:
        """Generate command using offline model augmented with RAG."""
        # Get relevant context from RAG
        context = self.rag_engine.get_context_for_query(description)
        
        # Clean templates from context to avoid confusion
        cleaned_context = self._clean_rag_templates(context)
        
        # Create augmented prompt with cleaned context
        prompt = self._create_rag_prompt(description, cleaned_context)
        
        # Generate response (optimized for speed)
        response = self.pipeline(
            prompt,
            max_new_tokens=100,  # Reduced from 200 for faster generation
            num_return_sequences=1,
            temperature=0.1,     # Very low temperature for fastest, most deterministic output
            do_sample=False,     # Greedy decoding is faster than sampling
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = response[0]['generated_text']
        command = self._extract_command(generated_text, prompt)
        
        return command
    
    def _generate_with_online_rag(self, description: str) -> str:
        """Generate command using online API augmented with RAG."""
        # Safety check: Ensure online client is set up
        if not hasattr(self, 'client'):
            print("⚠️  Online client not set up, initializing now...")
            self._setup_online_client()
        
        # Safety check: Ensure RAG engine is set up
        if self.rag_engine is None:
            print("⚠️  RAG engine not set up, initializing now...")
            self._setup_rag_engine()
        
        # Get relevant context from RAG
        context = self.rag_engine.get_context_for_query(description)
        
        # Clean templates from context to avoid confusion
        cleaned_context = self._clean_rag_templates(context)
        
        # Create augmented prompt with cleaned context for online API
        prompt = self._create_online_rag_prompt(description, cleaned_context)
        
        try:
            model = self.config.api_config.get('model', 'gpt-5-mini')
            
            # GPT-5 models only support temperature=1 (default)
            api_params = {
                'model': model,
                'messages': [
                    {"role": "system", "content": self._get_system_prompt()},  # Use same system prompt as online mode
                    {"role": "user", "content": prompt}
                ],
                'max_completion_tokens': self.config.api_config.get('max_tokens', 512)  # Use same token limit as online mode
            }
            
            # Only add temperature for non-GPT-5 models
            if not model.startswith('gpt-5'):
                api_params['temperature'] = self.config.api_config.get('temperature', 0.1)  # Use same temperature as online mode
            
            response = self.client.chat.completions.create(**api_params)
            
            generated_text = response.choices[0].message.content
            command = self._extract_command(generated_text, prompt)
            
            return command
            
        except Exception as e:
            print(f"Error with online RAG generation: {e}")
            # Fallback to regular online generation
            return self._generate_online(description)
    
    def _generate_online(self, description: str) -> str:
        """Generate command using online API."""
        prompt = self._create_prompt(description)
        
        try:
            model = self.config.api_config.get('model', 'gpt-5-mini')
            
            # GPT-5 models only support temperature=1 (default)
            # Older models support custom temperature values
            api_params = {
                'model': model,
                'messages': [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                'max_completion_tokens': self.config.api_config.get('max_tokens', 512)
            }
            
            # Only add temperature for non-GPT-5 models
            if not model.startswith('gpt-5'):
                api_params['temperature'] = self.config.api_config.get('temperature', 0.1)
            
            response = self.client.chat.completions.create(**api_params)
            
            generated_text = response.choices[0].message.content
            command = self._extract_command(generated_text, prompt)
            
            return command
            
        except Exception as e:
            raise Exception(f"API request failed: {e}")
    
    def _create_prompt(self, description: str) -> str:
        """Create prompt for command generation - optimized for speed."""
        return f"""Generate Linux command for: {description}

Rules:
- Linux/Unix only (no Windows commands)
- Complete, executable command
- No placeholders or templates
- Simple and direct

Examples:
- "list files" → ls -la
- "list files with date" → ls -lt
- "find txt files" → find . -name "*.txt"
- "files today" → find . -type f -newermt "$(date +%Y-%m-%d)"

Command:"""
    
    def _create_rag_prompt(self, description: str, context: str) -> str:
        """Create prompt with RAG context - optimized for speed."""
        # Check if context contains a known-good command
        if "Known-good command" in context:
            # Prioritize the cached successful command
            return f"""Generate Linux command for: {description}

IMPORTANT: Use this successful command as reference:
{context}

Rules:
- Linux/Unix only
- Complete, executable command
- Prefer simple commands like ls, find, grep
- No complex find -exec combinations unless necessary

Command:"""
        else:
            # Standard RAG prompt
            return f"""Generate Linux command for: {description}

Context:
{context}

Rules:
- Linux/Unix only
- Complete, executable command
- No placeholders (except {{}} in find -exec)
- Use $(date) for dates, not literal strings
- Simple and direct

Examples:
- "list files" → ls -la
- "list files with date" → ls -lt
- "files today" → find . -type f -newermt "$(date +%Y-%m-%d)"
- "find and grep" → find . -name "*.log" -exec grep 'error' {{}} \;

Command:"""
    
    def _create_online_rag_prompt(self, description: str, context: str) -> str:
        """Create prompt with RAG context for online API - optimized for speed."""
        return f"""Generate Linux command for: {description}

Documentation:
{context}

Rules:
- Linux/Unix only
- Complete, executable command
- No placeholders
- Simple and direct

Command:"""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for online API."""
        return """You are a Linux/Unix command generator.

    Rules:
    - Output ONLY one valid Linux/Unix command (bash/sh/zsh compatible).
    - DO NOT include explanations, markdown, quotes, or extra text.
    - Output must be a single line containing the command only.
    - Use only Linux commands (ls, cp, mv, rm, cat, find, etc.).
    - NEVER use Windows commands (dir, copy, del, etc.).
    - Use safe, correct syntax and realistic paths or filenames.
    - Do not use template placeholders like {{...}} or [[...]].
    - Use '{}' for find -exec placeholders.

    Output example:
    ls -la
    ls -lt
    find . -type f -newermt "$(date +%Y-%m-%d)"
    """
    
    def _extract_command(self, generated_text: str, prompt: str) -> str:
        """Extract the command from generated text with improved validation."""
        print(f"🔍 DEBUG: Raw generated text: '{generated_text}'")
        print(f"🔍 DEBUG: Prompt: '{prompt}'")
        
        # Remove the prompt from the generated text
        if prompt in generated_text:
            command_part = generated_text.replace(prompt, '').strip()
        else:
            command_part = generated_text.strip()
        
        print(f"🔍 DEBUG: Command part after prompt removal: '{command_part}'")
        
        # Extract the first line that looks like a command
        lines = command_part.split('\n')
        print(f"🔍 DEBUG: Lines found: {lines}")
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # Clean up the command - remove markdown code block backticks only
                # Do NOT strip quotes as they might be part of the command syntax
                command = line.strip('`')
                
                # Remove leading markdown list markers (-, *, •, etc.)
                if command.startswith(('-', '*', '•', '·', '→')):
                    command = command[1:].strip()
                
                print(f"🔍 DEBUG: Processing line: '{line}' -> command: '{command}'")
                
                # Fix common syntax errors before validation
                command = self._fix_common_syntax_errors(command)
                
                # Fix date command issues specifically
                command = self._fix_date_commands(command)
                
                print(f"🔍 DEBUG: After fixes: '{command}'")
                
                # Basic syntax validation
                if self._is_valid_command(command):
                    print(f"🔍 DEBUG: Valid command found: '{command}'")
                    return command
                else:
                    print(f"🔍 DEBUG: Command failed validation: '{command}'")
        
        # If no valid command found, return the first non-empty line
        for line in lines:
            line = line.strip()
            if line:
                command = line.strip('`')
                
                # Remove leading markdown list markers
                if command.startswith(('-', '*', '•', '·', '→')):
                    command = command[1:].strip()
                
                command = self._fix_common_syntax_errors(command)
                print(f"🔍 DEBUG: Fallback command: '{command}'")
                return command
        
        final_command = self._fix_common_syntax_errors(command_part)
        print(f"🔍 DEBUG: Final fallback command: '{final_command}'")
        return final_command
    
    def _fix_common_syntax_errors(self, command: str) -> str:
        """Fix common syntax errors in generated commands."""
        import re
        
        # CRITICAL: Fix template placeholders that LLM sometimes generates
        # But be careful not to treat {} in find -exec as templates
        if '{{' in command or ('{' in command and not self._is_find_exec_command(command)):
            print(f"🔧 Detected template placeholder in command: {command}")
            command = self._fix_template_placeholders(command)
        
        # Fix missing spaces after commands (e.g., "find." -> "find .")
        command = re.sub(r'\b(find|grep|ls|cat|head|tail|awk|sed|cut|sort|uniq|wc|du|df|ps|top|kill|chmod|chown|cp|mv|rm|mkdir|rmdir|tar|zip|unzip)\.', r'\1 .', command)
        
        # Fix extra dots in flags (e.g., "grep -l." -> "grep -l")
        command = re.sub(r'(-\w+)\.', r'\1', command)
        
        # Fix missing spaces around operators
        command = re.sub(r'([a-zA-Z0-9])\{\}([a-zA-Z0-9])', r'\1 {} \2', command)
        command = re.sub(r'([a-zA-Z0-9])\+([a-zA-Z0-9])', r'\1 + \2', command)
        
        # CRITICAL: Validate -exec commands have proper {} placeholder
        if re.search(r'-exec\s+.*[^{}]\s*\\;', command):
            print(f"🔧 Warning: -exec command missing {{}} placeholder: {command}")
            # Try to fix by adding {} where it should be
            if re.search(r'-exec\s+(\w+)\s+', command):
                command = re.sub(r'-exec\s+(\w+)\s+', r'-exec \1 {} ', command)
                print(f"🔧 Fixed -exec command: {command}")
        
        # Fix missing spaces in common patterns
        command = re.sub(r'ls-', 'ls -', command)
        command = re.sub(r'find-', 'find -', command)
        command = re.sub(r'grep-', 'grep -', command)
        
        # Fix common awk quote issues
        # If awk command ends with unclosed single quote, add closing quote
        awk_pattern = r"(awk\s+'[^']*)$"
        if re.search(awk_pattern, command):
            command = re.sub(awk_pattern, r"\1'", command)
            print(f"🔧 Fixed unclosed awk quotes")
        
        # Fix common quote mismatches
        single_quotes = command.count("'")
        if single_quotes % 2 != 0:
            # Add missing closing quote at the end
            command += "'"
            print(f"🔧 Fixed unmatched single quotes")
        
        # Fix inefficient find patterns
        if re.search(r'find\s+.*-exec\s+ls\s+', command):
            print(f"🔧 Warning: Inefficient find with ls detected")
            # Could suggest better alternatives here
        
        # Fix incomplete grep commands
        if re.search(r'grep\s+-[a-zA-Z]+\s*$', command):
            print(f"🔧 Warning: Incomplete grep command detected")
        
        return command.strip()
    
    def _is_find_exec_command(self, command: str) -> bool:
        """Check if command contains find -exec with {} placeholder."""
        import re
        # Check if it's a find command with -exec and {} placeholder
        return bool(re.search(r'find\s+.*-exec\s+.*\{\}.*\\;', command))
    
    def _fix_date_commands(self, command: str) -> str:
        """Fix common date command issues."""
        import re
        
        # Fix the specific issue we saw: extra ! in date commands
        # Pattern: $(date --date="today" +"%Y-%m-%d")! -newermt $(date --date="tomorrow" +"%Y-%m-%d")
        if '-newermt' in command and '!' in command:
            print(f"🔧 Fixing date command with extra exclamation mark")
            # Remove the ! from date commands
            command = re.sub(r'\+\"[^"]*\"\!', lambda m: m.group(0)[:-1], command)
            
        # Fix complex date ranges that are overly complicated
        # Replace complex today/tomorrow logic with simple today logic
        if '-newermt' in command and 'tomorrow' in command:
            print(f"🔧 Simplifying overly complex date command")
            # Replace with simple today command
            command = re.sub(
                r'-newermt\s+\$\(date[^)]*\)\!\s+-newermt\s+\$\(date[^)]*tomorrow[^)]*\)', 
                r'-newermt "$(date +%Y-%m-%d)"', 
                command
            )
        
        # Fix date format issues
        if 'date --date="today"' in command:
            print(f"🔧 Simplifying date command format")
            command = command.replace('date --date="today"', 'date')
            
        return command
    
    # Single-file cache creation function removed - using 5-shard cache directly
    
    def _fix_template_placeholders(self, command: str) -> str:
        """Use RAG-enhanced LLM to fix template placeholders intelligently."""
        try:
            # Extract the original task description from context if available
            task_description = "fix template placeholders in command"
            
            # Use RAG if available for better context
            if self.rag_engine:
                context = self.rag_engine.get_context_for_query("find exec commands template placeholders")
                fix_prompt = f"""You are a Linux command expert. Fix this command with template placeholders:

PROBLEM: {command}

RULES:
- Replace ALL template placeholders ({{...}}, {{{...}}}, etc.) with actual values
- CRITICAL: In find -exec commands, keep {{}} as {{}} (don't replace it!)
- Generate a REAL, EXECUTABLE Linux command
- Use the RAG context examples as reference

{context}

Generate ONLY the corrected Linux command:"""
            else:
                # Fallback to simple prompt
                fix_prompt = f"""Fix this Linux command with template placeholders:

PROBLEM: {command}

RULES:
- Replace ALL template placeholders with actual values
- CRITICAL: In find -exec commands, keep {{}} as {{}} (don't replace it!)
- Generate a REAL, EXECUTABLE Linux command

Generate ONLY the corrected Linux command:"""

            # Use the same pipeline to fix the command
            if self.pipeline:
                response = self.pipeline(
                    fix_prompt,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    temperature=0.1,  # Very low temperature for fastest, most deterministic output
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                fixed_command = response[0]['generated_text']
                # Extract the command from the response
                fixed_command = fixed_command.replace(fix_prompt, '').strip()
                
                # Clean up the command
                lines = fixed_command.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('//'):
                        # Remove only markdown backticks, not command quotes
                        cleaned_command = line.strip('`')
                        if cleaned_command and self._is_valid_command(cleaned_command):
                            print(f"RAG-enhanced LLM fixed template: {command} -> {cleaned_command}")
                            return cleaned_command
                
                print(f"WARNING: RAG-enhanced LLM fix failed, using fallback")
            else:
                print(f"WARNING: No pipeline available for RAG-enhanced fix")
                
        except Exception as e:
            print(f"WARNING: RAG-enhanced template fix failed: {e}")
        
        # Fallback: simple template replacement based on context
        return self._fallback_template_fix(command)

    # --- New public API: LLM-based repair for malformed commands ---
    def repair_command(self, description: str, candidate_command: str) -> str:
        """Repair a malformed command using the LLM only (no hardcoded rules).

        Uses the currently active mode (offline/online, with RAG context if available)
        to ask the model to output a single valid Linux command for the task.
        """
        try:
            # Prefer RAG context when available
            context = ""
            if self.rag_engine is not None:
                try:
                    context = self.rag_engine.get_context_for_query(description)
                except Exception:
                    context = ""

            prompt = self._create_llm_repair_prompt(description, candidate_command or "", context)

            # Route based on availability
            if self.pipeline is not None and self.config.get('mode', 'offline') in ['offline', 'rag']:
                response = self.pipeline(
                    prompt,
                    max_new_tokens=120,
                    num_return_sequences=1,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                generated_text = response[0]['generated_text']
                return self._extract_command(generated_text, prompt)

            # Online path
            if hasattr(self, 'client'):
                model = self.config.api_config.get('model', 'gpt-5-mini')
                api_params = {
                    'model': model,
                    'messages': [
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    'max_completion_tokens': 180
                }
                if not model.startswith('gpt-5'):
                    api_params['temperature'] = 0.1
                response = self.client.chat.completions.create(**api_params)
                generated_text = response.choices[0].message.content
                repaired = self._extract_command(generated_text, prompt)

                # If extraction failed (empty), try a simpler online generation fallback
                if not repaired.strip():
                    try:
                        alt_api_params = {
                            'model': model,
                            'messages': [
                                {"role": "system", "content": "Return ONLY a single valid Linux command for the user task. No explanations."},
                                {"role": "user", "content": f"Task: {description}\nLinux command:"}
                            ],
                            'max_completion_tokens': 120
                        }
                        if not model.startswith('gpt-5'):
                            alt_api_params['temperature'] = 0.1
                        alt_resp = self.client.chat.completions.create(**alt_api_params)
                        alt_text = alt_resp.choices[0].message.content
                        repaired = self._extract_command(alt_text, "")
                    except Exception:
                        pass

                # Defer RAG cache seeding to user approval in interactive flow
                if repaired:
                    self._pending_rag_seed = (description, repaired)
                return repaired

            # If neither pipeline nor client is available, fallback to a fresh generation
            return self.generate_command(description)
        except Exception:
            # Last resort: return original candidate
            return (candidate_command or "").strip()

    def _create_llm_repair_prompt(self, description: str, candidate: str, context: str) -> str:
        """Build a strict repair prompt asking for a single valid Linux command only."""
        parts = [
            "You are a Linux/Unix command generation expert.",
            "\nRules:",
            "- Output ONLY the command, no explanations.",
            "- Do NOT use placeholders like {{...}} or [[...]].",
            "- Prefer the simplest correct command for the task.",
            "- Generate a single, valid, executable Linux bash command.",
        ]
        if context:
            parts += ["\nLinux Documentation Context:", context]
        if candidate:
            parts += ["\nThe previous attempt was malformed:", candidate]
        parts += [
            "\nTask: " + description,
            "\nLinux command:",
        ]
        return "\n".join(parts)

    # --- Pending RAG seed helpers ---
    def has_pending_rag_seed(self) -> bool:
        return self._pending_rag_seed is not None

    def get_and_clear_pending_rag_seed(self):
        seed = self._pending_rag_seed
        self._pending_rag_seed = None
        return seed

    def _looks_like_shell_command(self, text: str) -> bool:
        """Heuristic check that the output resembles a shell command.

        Accepts forms like: 'ls -l', 'find . -type f', 'tar -cvf file.tar ...'
        Rejects empty text, markdown blocks, or prose sentences.
        """
        if not text:
            return False
        s = text.strip()
        # Strip common markdown fences
        if s.startswith("```"):
            return False
        # First token should be an alphanumeric/command-y token (no leading punctuation)
        first = s.split()[0]
        if not first:
            return False
        if first[0].isalnum():
            return True
        return False
    
    def _fallback_template_fix(self, command: str) -> str:
        """Fallback method to fix template placeholders without LLM."""
        import re
        
        original_command = command
        
        # More intelligent template replacement based on context
        if '{{command}}' in command or '{{{command}}}' in command:
            # Pure template placeholder - replace with appropriate command
            if 'list' in original_command.lower() or 'ls' in original_command.lower():
                command = re.sub(r'\{\{\{command\}\}\}|\{\{command\}\}', 'ls -la', command)
            elif 'find' in original_command.lower():
                command = re.sub(r'\{\{\{command\}\}\}|\{\{command\}\}', 'find . -name "*.txt"', command)
            else:
                command = re.sub(r'\{\{\{command\}\}\}|\{\{command\}\}', 'ls -la', command)
        
        elif '{{options}}' in command:
            # Replace options with appropriate flags
            if 'ls' in command:
                command = re.sub(r'\{\{options\}\}', '-la', command)
            elif 'find' in command:
                command = re.sub(r'\{\{options\}\}', '-type f', command)
            else:
                command = re.sub(r'\{\{options\}\}', '-la', command)
        
        elif '{{pattern}}' in command:
            # Replace pattern with appropriate pattern
            if 'find' in command:
                command = re.sub(r'\{\{pattern\}\}', '"*.txt"', command)
            elif 'grep' in command:
                command = re.sub(r'\{\{pattern\}\}', '"text"', command)
            else:
                command = re.sub(r'\{\{pattern\}\}', '"*.txt"', command)
        
        elif '{{find}}' in command:
            # Replace find template
            command = re.sub(r'\{\{find\}\}', 'find', command)
        
        # Handle file path placeholders ({{path/to/file}}, {{file}}, {{filename}}, etc.)
        elif re.search(r'\{\{.*?(?:path|file|filename|name).*?\}\}', command, re.IGNORECASE):
            # Replace path/file placeholders with example filename
            command = re.sub(r'\{\{.*?(?:path|file|filename|name).*?\}\}', 'file.txt', command, flags=re.IGNORECASE)
        
        else:
            # Generic fallback - distinguish between file paths and commands
            # If the template is in a position where a file would go, use a filename
            # Otherwise use a command
            
            # CRITICAL FIX: Handle -exec {} properly
            if '-exec' in command and '{}' in command:
                # This is a find -exec command with {} placeholder - keep {} as is
                print(f"🔧 Preserving {{}} placeholder in -exec command")
                # Only fix other template placeholders, not {}
                command = re.sub(r'\{\{\{.*?\}\}\}', 'ls -la', command)  # Triple braces
                command = re.sub(r'\{\{.*?\}\}', 'ls -la', command)      # Double braces
                # DON'T replace single {} in -exec commands!
            elif re.search(r'(cat|head|tail|less|more|grep|awk|sed)\s+.*\{\{', command):
                # File-oriented command - replace with filename
                command = re.sub(r'\{\{\{.*?\}\}\}', 'file.txt', command)  # Triple braces
                command = re.sub(r'\{\{.*?\}\}', 'file.txt', command)      # Double braces
                command = re.sub(r'\{.*?\}', 'file.txt', command)          # Single braces
            else:
                # Command position - replace with command
                template_fixes = {
                    r'\{\{\{.*?\}\}\}': 'ls -la',  # Triple braces
                    r'\{\{.*?\}\}': 'ls -la',      # Double braces
                    r'\{.*?\}': 'ls -la',          # Single braces
                }
                
                for pattern, replacement in template_fixes.items():
                    if re.search(pattern, command):
                        command = re.sub(pattern, replacement, command)
                        break
        
        if command != original_command:
            print(f"Fallback fix: {original_command} -> {command}")
        
        return command
    
    def _is_valid_command(self, command: str) -> bool:
        """Enhanced validation for command syntax."""
        if not command:
            return False
        
        # Clean the command first
        command = command.strip()
        
        # Check for basic Linux command patterns
        linux_commands = [
            'find', 'grep', 'ls', 'cat', 'head', 'tail', 'awk', 'sed', 'cut', 
            'sort', 'uniq', 'wc', 'du', 'df', 'ps', 'top', 'htop', 'kill', 
            'chmod', 'chown', 'cp', 'mv', 'rm', 'mkdir', 'rmdir', 'tar', 
            'zip', 'unzip', 'pwd', 'cd', 'echo', 'touch', 'ln', 'less', 
            'more', 'which', 'whereis', 'locate', 'file', 'stat', 'whoami',
            'hostname', 'uptime', 'free', 'uname', 'date', 'cal', 'man',
            'apropos', 'whatis', 'history', 'clear', 'exit', 'logout'
        ]
        
        # Check if command starts with a valid Linux command
        first_word = command.split()[0] if command.split() else ""
        if first_word not in linux_commands:
            return False
        
        # Additional syntax validation
        return self._validate_command_syntax(command)
    
    def _validate_command_syntax(self, command: str) -> bool:
        """Validate command syntax more thoroughly."""
        try:
            import re
            
            # Check for specific problematic patterns
            problematic_patterns = [
                # Extra exclamation marks in date commands (like the bug we saw)
                r'\+\"[^"]*\!\"',
                # Commands with missing spaces (e.g., "find." instead of "find .")
                r'\b(find|grep|ls|cat|head|tail|awk|sed|cut|sort|uniq|wc|du|df|ps|top|kill|chmod|chown|cp|mv|rm|mkdir|rmdir|tar|zip|unzip)\.',
                # Commands with extra dots in flags (e.g., "grep -l." instead of "grep -l")
                r'-\w+\.',
                # Missing spaces around operators
                r'[a-zA-Z0-9]{}[a-zA-Z0-9]',
                r'[a-zA-Z0-9]\+[a-zA-Z0-9]',
                # Invalid flag combinations
                r'--[a-zA-Z]+\.',
            ]
            
            for pattern in problematic_patterns:
                if re.search(pattern, command):
                    print(f"❌ Syntax Error: Detected problematic pattern: {pattern}")
                    return False
            
            # Check for quote mismatches
            single_quotes = command.count("'")
            double_quotes = command.count('"')
            if single_quotes % 2 != 0:
                print(f"❌ Syntax Error: Unmatched single quotes in command")
                return False
            if double_quotes % 2 != 0:
                print(f"❌ Syntax Error: Unmatched double quotes in command")
                return False
            
            # Check for proper spacing around common operators
            if 'find' in command:
                # Ensure find has proper spacing
                if re.search(r'find[^.\s]', command):
                    return False
                # Check for proper path specification
                if re.search(r'find\s+[^-]', command) and not re.search(r'find\s+\.', command) and not re.search(r'find\s+/', command):
                    # Allow find without path if it has flags
                    if not re.search(r'find\s+-', command):
                        return False
            
            # Check for proper grep syntax
            if 'grep' in command:
                # Ensure grep flags are properly formatted
                if re.search(r'grep\s+-[a-zA-Z]+\.', command):
                    return False
            
            # Check for awk syntax issues
            if 'awk' in command:
                # Check for proper awk quote usage
                awk_pattern = r"awk\s+'[^']*$"
                if re.search(awk_pattern, command):
                    print(f"❌ Syntax Error: Unclosed awk quotes")
                    return False
            
            # Check for incomplete commands
            incomplete_patterns = [
                r'grep\s+-[a-zA-Z]+\s*$',  # grep with flag but no pattern
                r'find\s+.*-exec\s+ls\s+',  # inefficient find with ls
                r'awk\s+\{[^}]*$',  # incomplete awk
            ]
            
            for pattern in incomplete_patterns:
                if re.search(pattern, command):
                    print(f"❌ Syntax Error: Incomplete command detected")
                    return False
            
            return True
            
        except Exception:
            # If validation fails, assume it's valid to avoid false negatives
            return True
    
    def switch_mode(self, mode: str):
        """Switch between offline, online, and RAG modes."""
        current_mode = self.config.get('mode', 'offline')
        
        # If already in the requested mode, don't reload
        if current_mode == mode:
            print(f"ℹ️  Already in {mode} mode")
            return
        
        self.config.set('mode', mode)
        
        if mode == 'offline':
            # Only load offline model if not already loaded
            if self.pipeline is None:
                print("🔄 Loading offline model...")
                self._load_offline_model()
            else:
                print("✅ Switched to offline mode (model already loaded)")
        elif mode == 'rag':
            # Load offline model if needed
            if self.pipeline is None:
                print("🔄 Loading offline model for RAG...")
                self._load_offline_model()
            # Setup RAG engine if needed
            if self.rag_engine is None:
                print("🔄 Setting up RAG engine...")
                self._setup_rag_engine()
            else:
                print("✅ Switched to RAG mode (components already loaded)")
        elif mode == 'online-rag':
            # Setup online client if needed
            if not hasattr(self, 'client'):
                print("🔄 Setting up online client...")
                self._setup_online_client()
            # Setup RAG engine if needed
            if self.rag_engine is None:
                print("🔄 Setting up RAG engine...")
                self._setup_rag_engine()
            else:
                print("✅ Switched to online-rag mode (components already loaded)")
        else:  # online mode
            # Only setup online client if not already setup
            if not hasattr(self, 'client'):
                print("🔄 Setting up online client...")
                self._setup_online_client()
            else:
                print("✅ Switched to online mode (client already setup)")
        
        print(f"✅ Switched to {mode} mode successfully")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.pipeline:
            return {
                'mode': 'offline',
                'model': self.config.model_config.get('base_model'),
                'device': self.device,
                'lora_loaded': False,  # Not using LoRA
            }
        else:
            return {
                'mode': 'online',
                'api_url': self.config.api_config.get('base_url'),
                'model': self.config.api_config.get('model')
            }
    
    def is_mode_configured(self, mode: str) -> bool:
        """Check if a mode is properly configured."""
        if mode == 'offline':
            return self.pipeline is not None
        elif mode == 'online':
            api_key = self.config.get('api.api_key') or os.getenv('OPENAI_API_KEY')
            return api_key is not None and hasattr(self, 'client')
        return False
    
    def get_current_mode(self) -> str:
        """Get the current mode."""
        return self.config.get('mode', 'offline')
    
    def preload_both_modes(self):
        """Preload both offline and online modes for instant switching."""
        print("🔄 Preloading both modes for instant switching...")
        
        # Load offline model
        if self.pipeline is None:
            print("📦 Loading offline model...")
            self._load_offline_model()
        
        # Setup online client
        if not hasattr(self, 'client'):
            print("🌐 Setting up online client...")
            self._setup_online_client()
        
        print("✅ Both modes preloaded - instant switching available!")
    
    def is_offline_ready(self) -> bool:
        """Check if offline mode is ready."""
        return self.pipeline is not None
    
    def is_online_ready(self) -> bool:
        """Check if online mode is ready."""
        return hasattr(self, 'client')
    
    def clear_model_cache(self):
        """Clear the model cache."""
        try:
            cache_files = list(self.cache_dir.glob("model_*.pkl"))
            if cache_files:
                for cache_file in cache_files:
                    cache_file.unlink()
                print(f"🗑️  Cleared {len(cache_files)} cached model(s)")
            else:
                print("ℹ️  No cached models found")
        except Exception as e:
            print(f"⚠️  Failed to clear cache: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the model cache."""
        try:
            cache_files = list(self.cache_dir.glob("model_*.pkl"))
            cache_info = {
                'cache_dir': str(self.cache_dir),
                'cached_models': len(cache_files),
                'cache_files': [str(f) for f in cache_files],
                'total_size': sum(f.stat().st_size for f in cache_files) if cache_files else 0
            }
            return cache_info
        except Exception as e:
            return {'error': str(e)}


