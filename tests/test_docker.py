#!/usr/bin/env python3
"""
Docker-specific tests
Tests for Docker environment, mounts, and configuration
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_docker_detection():
    """Test if running in Docker."""
    print("\n" + "="*70)
    print("TEST: Docker Detection")
    print("="*70)
    
    is_docker = os.path.exists('/.dockerenv')
    
    if is_docker:
        print("✅ Running in Docker container")
        
        # Check Docker-specific env vars
        env_vars = [
            'NVIDIA_VISIBLE_DEVICES',
            'NVIDIA_DRIVER_CAPABILITIES',
            'HF_HUB_DISABLE_SYMLINKS_WARNING',
            'TORCH_CUDA_ARCH_LIST',
        ]
        
        print("\n🔍 Environment variables:")
        for var in env_vars:
            value = os.getenv(var, 'NOT SET')
            print(f"   {var}: {value}")
        
    else:
        print("ℹ️  Running outside Docker (local environment)")
    
    return True


def test_docker_mounts():
    """Test Docker volume mounts."""
    print("\n" + "="*70)
    print("TEST: Docker Volume Mounts")
    print("="*70)
    
    is_docker = os.path.exists('/.dockerenv')
    
    if not is_docker:
        print("ℹ️  Skipping (not in Docker)")
        return True
    
    # Expected mounts
    mounts = {
        '/app': 'Application directory',
        '/root/.cache/huggingface': 'HuggingFace model cache',
        '/root/.lai-nux-tool/model_cache': 'Custom model cache',
        '/root/.lai-nux-tool/rag_cache': 'RAG index cache',
        '/root/.lai-nux-tool/doc_cache': 'Documentation cache',
    }
    
    print("\n📁 Checking mounts:")
    all_mounted = True
    
    for mount_path, description in mounts.items():
        exists = os.path.exists(mount_path)
        is_mounted = os.path.ismount(mount_path) or exists
        
        if exists:
            # Check if writable
            test_file = Path(mount_path) / '.write_test'
            try:
                test_file.touch()
                test_file.unlink()
                writable = True
            except:
                writable = False
            
            status = "✅" if writable else "⚠️ "
            print(f"   {status} {description}: {mount_path}")
            if not writable:
                print(f"      (exists but not writable)")
        else:
            print(f"   ❌ {description}: {mount_path} (NOT FOUND)")
            all_mounted = False
    
    return all_mounted


def test_docker_gpu():
    """Test GPU access in Docker."""
    print("\n" + "="*70)
    print("TEST: Docker GPU Access")
    print("="*70)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA available in Docker")
            print(f"   GPU count: {device_count}")
            
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {name} ({memory:.1f} GB)")
            
            return True
        else:
            print("⚠️  CUDA not available")
            print("   Make sure docker-compose uses 'runtime: nvidia'")
            return False
            
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False


def test_docker_cache_persistence():
    """Test that caches persist across Docker runs."""
    print("\n" + "="*70)
    print("TEST: Cache Persistence")
    print("="*70)
    
    is_docker = os.path.exists('/.dockerenv')
    
    if not is_docker:
        print("ℹ️  Skipping (not in Docker)")
        return True
    
    # Check cache locations
    caches = {
        '/root/.lai-nux-tool/doc_cache/linux_docs.json': 'Documentation',
        '/root/.lai-nux-tool/rag_cache/rag_cache.pkl': 'RAG pickle',
        '/root/.lai-nux-tool/rag_cache/faiss_index.bin': 'FAISS index',
    }
    
    print("\n💾 Checking cache files:")
    for cache_path, description in caches.items():
        path = Path(cache_path)
        
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"   ✅ {description}: {size:.1f} MB")
        else:
            print(f"   ⚠️  {description}: not found")
            print(f"      (will be created on first RAG run)")
    
    # Check model cache
    model_cache = Path('/root/.cache/huggingface')
    if model_cache.exists():
        cached_files = list(model_cache.rglob('*'))
        total_size = sum(f.stat().st_size for f in cached_files if f.is_file())
        total_size_gb = total_size / (1024**3)
        print(f"   ✅ HuggingFace cache: {total_size_gb:.2f} GB")
    else:
        print(f"   ⚠️  HuggingFace cache: empty")
    
    return True


def test_docker_entrypoint():
    """Test Docker entrypoint configuration."""
    print("\n" + "="*70)
    print("TEST: Docker Entrypoint")
    print("="*70)
    
    entrypoint_path = Path('/entrypoint.sh')
    
    if entrypoint_path.exists():
        print("✅ Entrypoint script exists")
        
        # Check if executable
        is_executable = os.access(entrypoint_path, os.X_OK)
        if is_executable:
            print("✅ Entrypoint is executable")
        else:
            print("❌ Entrypoint is NOT executable")
            return False
        
        # Check content
        content = entrypoint_path.read_text()
        
        checks = [
            ('CUDA Available', 'torch.cuda.is_available' in content),
            ('GPU Count', 'cuda.device_count' in content),
            ('Test environment', 'test_env' in content),
            ('Interactive mode', 'interactive' in content),
        ]
        
        print("\n📝 Entrypoint features:")
        for feature, exists in checks:
            status = "✅" if exists else "⚠️ "
            print(f"   {status} {feature}")
        
        return True
    else:
        print("⚠️  Entrypoint script not found (not critical)")
        return True


def main():
    """Run all Docker tests."""
    print("="*70)
    print("L-AI-NUX-TOOL - DOCKER TESTS")
    print("="*70)
    
    tests = [
        ("Docker Detection", test_docker_detection),
        ("Volume Mounts", test_docker_mounts),
        ("GPU Access", test_docker_gpu),
        ("Cache Persistence", test_docker_cache_persistence),
        ("Entrypoint", test_docker_entrypoint),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("DOCKER TEST SUMMARY")
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
        print("\n🎉 All Docker tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

