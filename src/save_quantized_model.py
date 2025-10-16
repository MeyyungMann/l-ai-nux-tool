#!/usr/bin/env python3
"""
Save Quantized Model
Saves the 4-bit quantized model to disk for faster loading next time.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path
import sys
import os

def save_quantized_model():
    """Save the quantized model to disk."""
    
    print("="*70)
    print("SAVING QUANTIZED MODEL")
    print("="*70)
    print()
    
    # Model configuration
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    save_dir = Path("./quantized_model_cache")
    
    print(f"📦 Model: {model_name}")
    print(f"💾 Save to: {save_dir}")
    print()
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already saved
    if (save_dir / "config.json").exists():
        print("⚠️  Quantized model already exists!")
        print(f"   Location: {save_dir}")
        print()
        response = input("Overwrite? [y/N]: ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
    
    # Configure 4-bit quantization
    print("🔧 Configuring 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load model with quantization
    print("📥 Loading model with quantization...")
    print("   (This will take 2-3 minutes)")
    print()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16,
        )
        
        print("✅ Model loaded with quantization")
        print()
        
        # Load tokenizer
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        print("✅ Tokenizer loaded")
        print()
        
        # Save quantized model
        print("💾 Saving quantized model to disk...")
        print("   (This will take 1-2 minutes)")
        print()
        
        model.save_pretrained(
            save_dir,
            safe_serialization=True  # Use safetensors format
        )
        
        print("✅ Model saved!")
        print()
        
        # Save tokenizer
        print("💾 Saving tokenizer...")
        tokenizer.save_pretrained(save_dir)
        
        print("✅ Tokenizer saved!")
        print()
        
        # Show size
        total_size = sum(f.stat().st_size for f in save_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        
        print("="*70)
        print("✅ QUANTIZED MODEL SAVED SUCCESSFULLY!")
        print("="*70)
        print(f"📍 Location: {save_dir.absolute()}")
        print(f"💾 Size: {size_gb:.2f} GB")
        print()
        print("🚀 Next time you load, it will be much faster!")
        print()
        print("To use the saved model, update your config to point to:")
        print(f"   {save_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def save_quantized_model_auto(model_name: str, save_dir: Path) -> bool:
    """
    Auto-save quantized model (non-interactive version).
    
    Args:
        model_name: HuggingFace model name to save
        save_dir: Directory to save the quantized model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"📦 Model: {model_name}")
        print(f"💾 Save to: {save_dir.absolute()}")
        print()
        
        # Create save directory
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already saved
        if (save_dir / "config.json").exists():
            print("✅ Quantized model already exists!")
            print(f"   Location: {save_dir}")
            return True
        
        # Configure 4-bit quantization
        print("🔧 Configuring 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Suppress HuggingFace warnings during auto-save
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        
        # Load model with quantization
        print("📥 Loading model with quantization...")
        print("   ⏱️  Progress: [          ] 0%", end='\r')
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16,
        )
        
        print("   ⏱️  Progress: [█████     ] 50%", end='\r')
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        print("   ⏱️  Progress: [███████   ] 70%")
        
        # Save quantized model
        print("💾 Saving quantized model to disk...")
        
        model.save_pretrained(
            save_dir,
            safe_serialization=True  # Use safetensors format
        )
        
        print("   ⏱️  Progress: [█████████ ] 90%")
        
        # Save tokenizer
        tokenizer.save_pretrained(save_dir)
        
        print("   ⏱️  Progress: [██████████] 100%")
        print()
        
        # Show size
        total_size = sum(f.stat().st_size for f in save_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        
        print(f"✅ Saved successfully!")
        print(f"📍 Location: {save_dir.absolute()}")
        print(f"💾 Size: {size_gb:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during auto-save: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    try:
        sys.exit(save_quantized_model())
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(1)


