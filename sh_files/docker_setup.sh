#!/bin/bash
# Docker setup script for Linux AI Command Generator

set -e

echo "🐳 Linux AI Command Generator - Docker Setup"
echo "============================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if NVIDIA Docker runtime is available (for GPU support)
if ! docker info | grep -q nvidia; then
    echo "⚠️  NVIDIA Docker runtime not detected."
    echo "   GPU acceleration may not be available."
    echo "   Install nvidia-docker2 for RTX 5090 support:"
    echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Create cache directories
echo "📁 Creating cache directories..."
mkdir -p rag_cache doc_cache

# Cache directories created
echo "✅ Cache directories ready"

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t lai-nux-tool:latest .

echo "✅ Docker setup complete!"
echo ""
echo "🚀 Usage:"
echo "  Interactive mode:    docker-compose up lai-nux-tool"
echo "  Run tests:          docker-compose up lai-nux-tool-test"
echo "  Shell access:       docker-compose up lai-nux-tool-shell"
echo ""
echo "🔧 Alternative commands:"
echo "  docker run -it --gpus all lai-nux-tool:latest interactive"
echo "  docker run -it --gpus all lai-nux-tool:latest test"
echo "  docker run -it --gpus all lai-nux-tool:latest bash"
echo ""
echo "📁 Project Structure:"
echo "  docs/     - Documentation files"
echo "  tests/    - Test files"
echo "  src/      - Source code"
echo ""
echo "💡 Tips:"
echo "  • First run will initialize RAG system (downloads embedding models)"
echo "  • RAG cache is stored in ./rag_cache directory"
echo "  • Use Ctrl+C to stop interactive mode"
echo "  • GPU support requires nvidia-docker2"
