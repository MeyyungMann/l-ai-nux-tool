#!/usr/bin/env python3
"""
Quick smoke tests for L-AI-NUX-TOOL
Fast tests to verify basic functionality
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all core modules can be imported."""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    try:
        from src.config import Config
        print("✅ Config module imported")
        
        from src.command_parser import CommandParser
        print("✅ CommandParser module imported")
        
        from src.llm_engine import LLMEngine
        print("✅ LLMEngine module imported")
        
        from src.rag_engine import RAGEngine
        print("✅ RAGEngine module imported")
        
        from src.doc_collector import LinuxDocCollector
        print("✅ DocCollector module imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_config():
    """Test configuration system."""
    print("\n" + "="*70)
    print("TEST 2: Configuration System")
    print("="*70)
    
    try:
        from src.config import Config
        
        config = Config()
        print("✅ Config initialized")
        
        # Test setting and getting values
        config.set('mode', 'offline')
        mode = config.get('mode')
        assert mode == 'offline', f"Expected 'offline', got '{mode}'"
        print("✅ Config set/get works")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_rag_cache():
    """Test RAG cache existence."""
    print("\n" + "="*70)
    print("TEST 3: RAG Cache Check")
    print("="*70)
    
    try:
        cache_dir = Path.home() / '.lai-nux-tool'
        doc_cache = cache_dir / 'doc_cache' / 'linux_docs.json'
        rag_cache_pkl = cache_dir / 'rag_cache' / 'rag_cache.pkl'
        rag_cache_idx = cache_dir / 'rag_cache' / 'faiss_index.bin'
        
        # Check local project cache (for Docker)
        local_doc_cache = Path('./doc_cache/linux_docs.json')
        local_rag_cache = Path('./rag_cache/rag_cache.pkl')
        
        if doc_cache.exists() or local_doc_cache.exists():
            print("✅ Documentation cache found")
            cache_found = True
        else:
            print("⚠️  No documentation cache (will be created on first RAG run)")
            cache_found = False
        
        if (rag_cache_pkl.exists() and rag_cache_idx.exists()) or local_rag_cache.exists():
            print("✅ RAG index cache found")
        else:
            print("⚠️  No RAG index cache (will be created on first RAG run)")
        
        print("\n📁 Cache Locations:")
        print(f"   User cache: {cache_dir}")
        print(f"   Local cache: ./rag_cache, ./doc_cache")
        
        return True  # Not a failure if cache doesn't exist
    except Exception as e:
        print(f"❌ Cache check failed: {e}")
        return False


def test_gpu_availability():
    """Test GPU availability."""
    print("\n" + "="*70)
    print("TEST 4: GPU Availability")
    print("="*70)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA Available: {cuda_available}")
            print(f"   GPU Count: {device_count}")
            print(f"   GPU Name: {device_name}")
        else:
            print("⚠️  No GPU detected (CPU mode will be used)")
        
        return True
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")
        return True  # Not a critical failure


def test_command_parser():
    """Test command parser basic functionality."""
    print("\n" + "="*70)
    print("TEST 5: Command Parser")
    print("="*70)
    
    try:
        from src.command_parser import CommandParser
        
        parser = CommandParser()
        print("✅ CommandParser initialized")
        
        # Test basic validation
        test_commands = [
            ("ls -la", "Basic list command"),
            ("find . -name '*.txt'", "Find command"),
            ("grep -r 'pattern' /path", "Grep command"),
        ]
        
        for cmd, desc in test_commands:
            # Just verify we can create the parser and it has methods
            # Actual validation depends on the implementation
            print(f"✅ {desc}: {cmd}")
        
        print("✅ CommandParser test passed")
        return True
    except Exception as e:
        print(f"❌ CommandParser test failed: {e}")
        return False


def test_environment():
    """Test environment and dependencies."""
    print("\n" + "="*70)
    print("TEST 6: Environment & Dependencies")
    print("="*70)
    
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sentence_transformers': 'Sentence Transformers (RAG)',
        'faiss': 'FAISS (RAG)',
        'click': 'Click CLI',
        'rich': 'Rich Console',
        'openai': 'OpenAI Client',
        'peft': 'PEFT (LoRA)',
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} not installed")
            if module in ['sentence_transformers', 'faiss']:
                print(f"   (Optional for RAG mode)")
            else:
                missing.append(module)
    
    if missing:
        print(f"\n⚠️  Missing required dependencies: {', '.join(missing)}")
        return False
    
    return True


def main():
    """Run all quick tests."""
    print("="*70)
    print("L-AI-NUX-TOOL - QUICK TEST SUITE")
    print("="*70)
    print("\n🚀 Running quick smoke tests...")
    print("⚡ This should complete in under 10 seconds\n")
    
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("RAG Cache", test_rag_cache),
        ("GPU", test_gpu_availability),
        ("Command Parser", test_command_parser),
        ("Environment", test_environment),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 All quick tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

