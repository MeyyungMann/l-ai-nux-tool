#!/usr/bin/env python3
"""
Load Quantized Model
Loads a previously saved quantized model for faster startup.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import time

def load_saved_quantized_model():
    """Load the saved quantized model."""
    
    print("="*70)
    print("LOADING SAVED QUANTIZED MODEL")
    print("="*70)
    print()
    
    model_dir = Path("./quantized_model_cache")
    
    # Check if model exists
    if not model_dir.exists():
        print(f"❌ Quantized model not found at: {model_dir}")
        print()
        print("Run this first to save the model:")
        print("   python save_quantized_model.py")
        return 1
    
    print(f"📍 Loading from: {model_dir.absolute()}")
    print()
    
    # Time the loading
    start_time = time.time()
    
    try:
        # Load tokenizer
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print("✅ Tokenizer loaded")
        print()
        
        # Load quantized model
        print("📥 Loading quantized model...")
        print("   (Should be faster than re-quantizing!)")
        print()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            dtype=torch.float16,
        )
        
        elapsed = time.time() - start_time
        
        print(f"✅ Model loaded in {elapsed:.1f} seconds!")
        print()
        
        # Test generation
        print("🧪 Testing model...")
        prompt = "Generate a Linux command to list all files:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test output: {response[:100]}...")
        print()
        
        print("="*70)
        print("✅ LOADED SUCCESSFULLY!")
        print("="*70)
        print(f"⏱️  Loading time: {elapsed:.1f} seconds")
        print(f"💾 Model size on disk: {sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024**3):.2f} GB")
        print()
        print("🚀 Model is ready to use!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    try:
        sys.exit(load_saved_quantized_model())
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(1)


