# L-AI-NUX-TOOL Test Suite

Comprehensive test suite for the Linux AI Command Generator.

## Test Files

### Comparison Test
**`test_comparison.py`** - Compare all modes (~5 minutes)
- Offline (No RAG) vs Offline + RAG
- Online (No RAG) vs Online + RAG
- Shows RAG impact
- Provides improvement suggestions

```bash
# Run comparison
python tests/test_comparison.py

# Using helper script
compare_modes.bat      # Windows
./compare_modes.sh     # Linux/Mac
```

### Quick Tests
**`quick_test.py`** - Fast smoke tests (~10 seconds)
- Module imports
- Configuration system
- RAG cache check
- GPU availability
- Command parser
- Environment dependencies

```bash
# Run quick tests
python tests/quick_test.py

# In Docker
docker-compose run --rm lai-nux-tool-test
```

### Comprehensive Tests
**`test_suite.py`** - Full test suite (~60 seconds)
- Documentation collection
- RAG engine initialization
- RAG search functionality
- Offline model configuration
- Command generation
- Docker environment
- Integration tests

```bash
# Run full suite
python tests/test_suite.py

# In Docker
docker-compose run --rm lai-nux-tool bash -c "python tests/test_suite.py"
```

### RAG Tests
**`test_rag.py`** - RAG-specific tests
- RAG dependencies
- Documentation collector
- RAG engine
- Context generation

```bash
# Run RAG tests
python tests/test_rag.py

# In Docker with RAG mode
docker-compose run --rm lai-nux-tool-rag bash -c "python tests/test_rag.py"
```

### Docker Tests
**`test_docker.py`** - Docker environment tests
- Docker detection
- Volume mounts
- GPU access
- Cache persistence
- Entrypoint configuration

```bash
# Run Docker tests (must run in Docker)
docker-compose run --rm lai-nux-tool-test bash -c "python tests/test_docker.py"
```

## Running Tests

### Local (Windows/Linux)
```bash
# Quick smoke tests
python tests/quick_test.py

# Full test suite
python tests/test_suite.py

# Specific test
python tests/test_rag.py
python tests/test_docker.py
```

### Docker
```bash
# Using test service
docker-compose up lai-nux-tool-test

# Run specific test
docker-compose run --rm lai-nux-tool-test bash -c "python tests/quick_test.py"

# Interactive testing
docker-compose run --rm lai-nux-tool-shell
python tests/test_suite.py
```

### Helper Scripts
```bash
# Windows
quick_docker_test_enhanced.bat

# Linux/Mac
./verify_docker_model.sh
```

## Test Categories

### 1. Unit Tests
- Individual component testing
- Module imports
- Configuration
- Parsers

### 2. Integration Tests
- RAG + Offline mode
- Command generation pipeline
- Cache systems

### 3. System Tests
- Docker environment
- GPU access
- Volume mounts
- Full workflow

### 4. Performance Tests
- Model loading time
- RAG search speed
- Cache efficiency

## Expected Results

### ✅ All Tests Pass
All functionality working correctly.

### ⚠️ Warnings
- No RAG cache (first run)
- No LoRA weights (will use base model)
- No GPU (CPU mode)

These are not failures, just informational.

### ❌ Failures
Check:
1. Dependencies: `pip install -r requirements.txt`
2. Docker setup: `docker-compose config`
3. GPU access: `nvidia-smi`
4. Permissions: Check file/directory access

## Test Data

### Test Environment
Located in `test_env/` directory:
- Sample files and directories
- Scripts and logs
- Hidden files
- Symbolic links
- Files with special characters

Created automatically by Docker entrypoint.

### Training Data
- `quality_train_enhanced.jsonl` - 100+ examples
- `quality_train_enhanced_v2.jsonl` - Extended dataset

### LoRA Weights
- `lora_linux_commands_enhanced/` - Latest weights
- `lora_linux_commands_improved_v2/` - Alternative weights

## Continuous Testing

### Before Commits
```bash
python tests/quick_test.py
```

### Before Releases
```bash
python tests/test_suite.py
python tests/test_rag.py
python tests/test_docker.py
```

### Performance Benchmarking
```bash
# Time full workflow
time python gen_cmd.py --mode offline "list files"
time python gen_cmd.py --mode rag "list files"
```

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
pip install -r requirements-docker.txt  # For Docker
```

### RAG Tests Fail
```bash
# Initialize RAG system
python test_rag_system.py
```

### Docker Tests Fail
```bash
# Check Docker setup
docker-compose config

# Rebuild image
docker-compose build lai-nux-tool

# Check mounts
docker-compose run --rm lai-nux-tool-shell ls -la /root/.lai-nux-tool
```

### GPU Not Detected
```bash
# Check NVIDIA runtime
nvidia-smi
docker run --rm --runtime=nvidia nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check docker-compose
docker-compose config | grep runtime
```

## Adding New Tests

### 1. Create test file
```python
#!/usr/bin/env python3
"""Test description."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_my_feature():
    """Test my feature."""
    print("\n" + "="*70)
    print("TEST: My Feature")
    print("="*70)
    
    # Test code here
    
    return True

if __name__ == '__main__':
    sys.exit(0 if test_my_feature() else 1)
```

### 2. Add to test suite
Edit `test_suite.py` and add your test function to the `tests` list.

### 3. Document
Update this README with test description and usage.

## Coverage

Current test coverage:
- ✅ Configuration system
- ✅ RAG engine
- ✅ Documentation collector
- ✅ Command parser
- ✅ Docker environment
- ✅ Cache systems
- ⚠️  Full model loading (slow, skipped in quick tests)
- ⚠️  Online API mode (requires API key)

## CI/CD Integration

### GitHub Actions (example)
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run quick tests
        run: python tests/quick_test.py
      - name: Run full tests
        run: python tests/test_suite.py
```

## Support

For issues with tests:
1. Check this README
2. Review test output carefully
3. Check main project README
4. Check docs/TEST_ENVIRONMENT_GUIDE.md

