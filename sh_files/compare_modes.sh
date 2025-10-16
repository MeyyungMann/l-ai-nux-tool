#!/bin/bash
# Compare all modes - Offline/Online with/without RAG

echo "========================================"
echo "L-AI-NUX-TOOL - Mode Comparison"
echo "========================================"
echo ""
echo "This will test and compare:"
echo "  1. Offline (No RAG) - Base model only"
echo "  2. Offline + RAG - With documentation"
echo "  3. Online (No RAG) - API without context"
echo "  4. Online + RAG - API with context"
echo ""
echo "This may take 5-10 minutes..."
echo ""
read -p "Press Enter to continue..."

python tests/test_comparison.py

