#!/bin/bash
# L-AI-NUX-TOOL - RAG Mode (Docker)
# Runs the tool with RAG (Retrieval-Augmented Generation) mode

echo "========================================"
echo "L-AI-NUX-TOOL - RAG Mode (Docker)"
echo "========================================"
echo ""
echo "This will run the tool with Linux documentation retrieval"
echo "for enhanced accuracy."
echo ""

# Create cache directories if they don't exist
mkdir -p ./rag_cache
mkdir -p ./doc_cache

# Check if running for first time
if [ ! -f "./doc_cache/linux_docs.json" ]; then
    echo ""
    echo "========================================"
    echo "FIRST TIME SETUP DETECTED"
    echo "========================================"
    echo ""
    echo "This is your first time running RAG mode in Docker."
    echo "The system will:"
    echo "  1. Download Linux documentation (5-10 min)"
    echo "  2. Build vector index (3-5 min)"
    echo "  3. Cache everything for future use"
    echo ""
    echo "Total time: ~10-15 minutes (one-time only)"
    echo "Future runs will be instant!"
    echo ""
    read -p "Press Enter to continue..."
fi

echo ""
echo "Starting Docker container with RAG mode..."
echo ""

# Use 'run' instead of 'up' for interactive mode
docker-compose run --rm lai-nux-tool-rag

echo ""
echo "Container stopped."

