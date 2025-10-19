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
   git clone <repository-url>
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

#### Shell Access for Debugging
```bash
docker-compose run --rm lai-nux-tool-shell
```
Get a bash shell inside the container for debugging.

### Configuration

#### Environment Variables
Create a `.env` file for API keys (optional):
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

#### GPU Support
The Docker setup automatically detects and uses NVIDIA GPUs. For RTX 5090:
- CUDA architecture is optimized (`TORCH_CUDA_ARCH_LIST: "8.9"`)
- Memory allocation is tuned (`PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512"`)

#### Memory Requirements
- **Minimum**: 8GB RAM
- **Recommended**: 16GB+ RAM
- **GPU**: RTX 5090 (or any NVIDIA GPU with 8GB+ VRAM)

### Troubleshooting

#### Common Issues

1. **Out of Memory Error:**
   ```bash
   # Reduce memory limits in docker-compose.yml
   # Or use a smaller model
   docker-compose run --rm lai-nux-tool-ollama
   ```

2. **CUDA Not Available:**
   ```bash
   # Install NVIDIA Docker runtime
   # Or run without GPU (slower but works)
   docker-compose run --rm lai-nux-tool-rag
   ```

3. **First Run Taking Too Long:**
   - This is normal! Models are large (several GB)
   - Subsequent runs will be much faster
   - Consider using Ollama mode for faster startup

#### Performance Tips

- **Faster Startup**: Use Ollama mode instead of RAG mode
- **Better Accuracy**: Use RAG mode with full Linux documentation
- **Memory Issues**: Use `lai-nux-tool-test` service (lighter memory footprint)

### Manual Installation (Alternative)

If you prefer not to use Docker:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the tool:**
   ```bash
   python gen_cmd.py
   ```

3. **Available modes:**
   ```bash
   python gen_cmd.py --mode rag --interactive
   python gen_cmd.py --mode ollama --interactive
   python gen_cmd.py --mode compare --interactive
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
