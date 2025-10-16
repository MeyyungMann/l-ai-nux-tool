#!/bin/bash
# Build RAG cache for Docker use (in project directory)

echo "=================================================================="
echo "🐳 Building RAG Cache for Docker"
echo "=================================================================="
echo ""

# Ensure we're building in the project directory (for Docker to use)
export RAG_CACHE_DIR="./rag_cache"
export DOC_CACHE_DIR="./doc_cache"

echo "📍 Cache will be built in: $RAG_CACHE_DIR"
echo "📚 Using docs from: $DOC_CACHE_DIR"
echo ""

# Clear any existing cache in project directory
if [ -d "$RAG_CACHE_DIR" ]; then
    echo "🗑️  Clearing existing project RAG cache..."
    rm -rf $RAG_CACHE_DIR/*
    echo "✅ Cleared"
else
    mkdir -p $RAG_CACHE_DIR
fi
echo ""

# Build RAG cache using Docker
echo "🔨 Building RAG cache in Docker..."
echo ""

docker-compose run --rm lai-nux-tool-test-50-offline \
    python -c "
import sys
sys.path.insert(0, '/app')
from pathlib import Path
from src.rag_engine import RAGEngine

print('Initializing RAG engine...')
# Force it to use project directory
cache_dir = Path('/root/.lai-nux-tool/rag_cache')
rag = RAGEngine(cache_dir=cache_dir)

print('RAG cache built!')
print(f'Cache location: {rag.cache_dir}')
print(f'Documents: {len(rag.documents)}')
"

echo ""
echo "=================================================================="
echo "✅ RAG Cache Built!"
echo "=================================================================="
echo ""

# Show what was created
echo "📁 Created files:"
ls -lh $RAG_CACHE_DIR/
echo ""

echo "🧪 Ready to test:"
echo "  ./docker_test_50_queries_offline.sh"
echo ""


