# Shell Scripts

This folder contains all shell scripts (.sh) for Linux/Mac/WSL.

## Available Scripts:

### Docker Management
- `docker_rag.sh` - **Main entry point** - Start in RAG mode
- `docker_setup.sh` - Initial Docker setup
- `docker_test.sh` - Test Docker installation
- `docker_test_rag.sh` - Test RAG system
- `cleanup_docker.sh` - Clean up Docker containers

### Comparison & Demo
- `compare_modes.sh` - Compare all modes (offline, online, RAG, online+RAG)
- `demo_rag.sh` - RAG demonstration
- `show_rag_examples.sh` - Show RAG examples

### Utilities
- `docker_save_quantized.sh` - Save quantized model cache
- `run_tests.sh` - Run test suite
- `entrypoint.sh` - Docker entrypoint (system file)

## How to Use:

Make scripts executable and run:
```bash
chmod +x sh_files/docker_rag.sh
./sh_files/docker_rag.sh
```

Or run directly:
```bash
bash sh_files/docker_rag.sh
```

## Main Workflow:

1. **First Time:** `./sh_files/docker_setup.sh`
2. **Test:** `./sh_files/docker_test.sh`
3. **Run:** `./sh_files/docker_rag.sh`
4. **Compare:** `./sh_files/compare_modes.sh`

