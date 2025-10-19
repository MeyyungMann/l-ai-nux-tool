# L-AI-NUX-TOOL

Linux AI Command Generator with RAG (Retrieval-Augmented Generation) support.

## Features

- AI-powered Linux command generation
- RAG system for enhanced accuracy
- Docker support for easy deployment
- Support for both offline and online modes
- Optimized for RTX 5090 GPU with LoRA Fine-Tuning
- Multiple AI model options (HuggingFace, Ollama, OpenAI)

## 🚀 Getting Started with Docker

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)
- At least 8GB RAM available
- RTX 5090 GPU recommended (but not required)

### Quick Start (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MeyyungMann/L-AI-NUX-TOOL-.git
   cd L-AI-NUX-TOOL-
   ```

2. **Run with Online + RAG mode (recommended):**
   ```bash
   docker-compose run --rm lai-nux-tool-rag --interactive
   ```

That's it! The first run will take 10-15 minutes to download models and build the RAG index, but subsequent runs will be instant.

### Available Docker Modes

#### 🌐 Online Mode
```bash
docker-compose run --rm lai-nux-tool-rag --mode online --interactive
```
- Uses OpenAI-compatible API
- Requires API key
- Fast and accurate

#### 🌐📚 Online + RAG Mode (Recommended)
```bash
docker-compose run --rm lai-nux-tool-rag --interactive
```
- Uses API with Linux documentation retrieval
- Best accuracy for Linux commands
- Requires API key + RAG cache
- First run: downloads models + builds RAG index (~10-15 min)

#### 🤖 Ollama Mode (Open Source)
```bash
docker-compose run --rm lai-nux-tool-ollama
```
- Uses Ollama with GPT-OSS models
- No external API keys required
- Good balance of speed and accuracy

#### 🤖📚 GPT-OSS + RAG Mode
```bash
docker-compose run --rm lai-nux-tool-ollama-rag
```
- Combines GPT-OSS with RAG for enhanced accuracy
- Best of both worlds: open-source + documentation retrieval

#### 🧪 Compare Mode
```bash
docker-compose run --rm lai-nux-tool-compare
```
- Runs all available modes and compares results
- Useful for testing and evaluation

#### 🔧 Interactive Mode (Basic)
```bash
docker-compose run --rm lai-nux-tool
```
- Basic interactive mode without RAG
- Fastest startup time

#### 🧪 Testing Services
```bash
# Lightweight testing service
docker-compose run --rm lai-nux-tool-test

# Comprehensive 50-query testing
docker-compose run --rm lai-nux-tool-test-50

# Offline testing (no API required)
docker-compose run --rm lai-nux-tool-test-50-offline
```

#### 🔧 Development Services
```bash
# Shell access for debugging
docker-compose run --rm lai-nux-tool-shell

# Training service (for model fine-tuning)
docker-compose run --rm lai-nux-tool-train
```

### Advanced Setup

#### First-Time Setup Script
```bash
./sh_files/docker_setup.sh
```
This script will:
- Build the Docker image
- Set up necessary directories
- Download initial models
- Configure the environment

#### Testing
```bash
./sh_files/docker_test.sh
```
Run comprehensive tests to verify everything works correctly.

#### Shell Scripts (Convenience Tools)
The `sh_files/` directory contains helpful scripts:

```bash
# Main workflow scripts
./sh_files/docker_setup.sh      # Initial Docker setup
./sh_files/docker_test.sh       # Test Docker installation
./sh_files/compare_modes.sh     # Compare all modes

# Ollama setup (for open-source mode)
./sh_files/setup_ollama.sh      # Setup Ollama with GPT-OSS model

# Make scripts executable
chmod +x sh_files/*.sh
```

**Main Workflow:**
1. **First Time:** `./sh_files/docker_setup.sh`
2. **Test:** `./sh_files/docker_test.sh`
3. **Run:** `docker-compose run --rm lai-nux-tool-rag --interactive`
4. **Compare:** `./sh_files/compare_modes.sh`

#### Shell Access for Debugging
```bash
docker-compose run --rm lai-nux-tool-shell
```
Get a bash shell inside the container for debugging.

### Configuration

#### Environment Variables
Create a `.env` file for API keys (optional):

```bash
# Copy the example file
cp env.example .env

# Edit with your API key
nano .env
```

**Supported API Providers:**

1. **OpenAI (Default):**
   ```bash
   OPENAI_API_KEY=sk-your-api-key-here
   OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_MODEL=gpt-3.5-turbo
   ```

2. **Local Ollama (Free, Open Source):**
   ```bash
   OPENAI_BASE_URL=http://localhost:11434/v1
   OPENAI_MODEL=gpt-oss
   OPENAI_API_KEY=ollama
   ```

3. **Azure OpenAI:**
   ```bash
   OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
   OPENAI_MODEL=gpt-35-turbo
   OPENAI_API_KEY=your-azure-api-key
   ```

4. **LiteLLM Proxy:**
   ```bash
   OPENAI_BASE_URL=http://localhost:4000
   OPENAI_API_KEY=your-litellm-key
   ```

**Setup Ollama (Open Source Option):**
```bash
# Install Ollama first
curl -fsSL https://ollama.ai/install.sh | sh

# Setup GPT-OSS model
./sh_files/setup_ollama.sh
```

#### GPU Support
The Docker setup automatically detects and uses NVIDIA GPUs. For RTX 5090:
- CUDA architecture is optimized (`TORCH_CUDA_ARCH_LIST: "8.9"`)
- Memory allocation is tuned (`PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512"`)

#### Memory Requirements
- **Minimum**: 8GB RAM
- **Recommended**: 16GB+ RAM
- **GPU**: any NVIDIA GPU with 8GB+ VRAM

### Troubleshooting

#### Common Issues

1. **Out of Memory Error:**
   ```bash
   # Use lighter memory service
   docker-compose run --rm lai-nux-tool-test
   
   # Or use Ollama mode (lighter)
   docker-compose run --rm lai-nux-tool-ollama
   
   # Reduce memory limits in docker-compose.yml if needed
   ```

2. **CUDA Not Available:**
   ```bash
   # Install NVIDIA Docker runtime
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   
   # Or run without GPU (slower but works)
   docker-compose run --rm lai-nux-tool-rag
   ```

3. **First Run Taking Too Long:**
   - This is normal! Models are large (several GB)
   - Subsequent runs will be much faster
   - Consider using Ollama mode for faster startup

4. **Docker Build Failures:**
   ```bash
   # Clean Docker cache
   docker system prune -a
   
   # Rebuild from scratch
   docker-compose build --no-cache
   ```

5. **Permission Issues:**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   
   # Make scripts executable
   chmod +x sh_files/*.sh
   ```

6. **API Key Issues:**
   ```bash
   # Check .env file exists and has correct format
   cat .env
   
   # Test API key
   python gen_cmd.py --api-key YOUR_KEY "test"
   ```

7. **RAG Cache Issues:**
   ```bash
   # Rebuild RAG cache
   python src/utils/fetch_man_pages.py --auto
   
   # Clear corrupted cache
   rm -rf rag_cache/* doc_cache/*
   python src/utils/fetch_man_pages.py --auto
   ```

#### Performance Tips

- **Faster Startup**: Use Ollama mode instead of RAG mode
- **Better Accuracy**: Use RAG mode with full Linux documentation
- **Memory Issues**: Use `lai-nux-tool-test` service (lighter memory footprint)
- **Development**: Use `lai-nux-tool-shell` for debugging

#### Debug Mode
```bash
# Enable debug logging
export DEBUG=1
python gen_cmd.py --interactive

# Or run tests to verify setup
./sh_files/docker_test.sh
```

### Manual Installation (Alternative)

If you prefer not to use Docker:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup environment:**
   ```bash
   # Copy environment file
   cp env.example .env
   
   # Edit with your API key (optional)
   nano .env
   ```

3. **Setup RAG system (optional but recommended):**
   ```bash
   # Build RAG cache with Linux documentation
   python src/utils/fetch_man_pages.py --auto
   ```

4. **Run the tool:**
   ```bash
   # Interactive mode with mode selection
   python gen_cmd.py --interactive
   
   # Direct command generation
   python gen_cmd.py "find all text files"
   ```

5. **Available modes:**
   ```bash
   python gen_cmd.py --mode online --interactive
   python gen_cmd.py --mode online-rag --interactive
   python gen_cmd.py --mode ollama --interactive
   python gen_cmd.py --mode ollama-rag --interactive
   python gen_cmd.py --mode compare --interactive
   ```

6. **Command line options:**
   ```bash
   python gen_cmd.py --help                    # Show help
   python gen_cmd.py --mode online-rag "list files"  # Generate command
   python gen_cmd.py --api-key YOUR_KEY "find .txt"  # With custom API key
   python gen_cmd.py --show-rag-impact "find files"  # Show RAG comparison
   ```

## 🚀 Quick Commands Reference

### Docker Commands
```bash
# Quick start (recommended)
docker-compose run --rm lai-nux-tool-rag --interactive

# Different modes
docker-compose run --rm lai-nux-tool-test                      # Run tests

# Development
docker-compose run --rm lai-nux-tool-shell                     # Shell access
```

### Python Commands
```bash
# Interactive mode
python gen_cmd.py --interactive

# Direct generation
python gen_cmd.py --mode online-rag "find all .txt files"
python gen_cmd.py --mode ollama "list files with permissions"
python gen_cmd.py --mode compare "search for text in files"

# With custom API key
python gen_cmd.py --api-key YOUR_KEY --mode online "backup directory"
```

### Common Use Cases
```bash
# File operations
python gen_cmd.py "find all PDF files in Documents folder"
python gen_cmd.py "delete all .tmp files recursively"
python gen_cmd.py "copy all .jpg files to backup folder"

# System operations
python gen_cmd.py "check disk usage of all directories"
python gen_cmd.py "find processes using most memory"
python gen_cmd.py "compress all log files older than 7 days"

# Text processing
python gen_cmd.py "search for 'error' in all log files"
python gen_cmd.py "count lines in all Python files"
python gen_cmd.py "replace text in all .txt files"
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Copyright

Copyright 2025 Meyyung Mann

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
