#!/bin/bash
# RAG Impact Demonstration - Show how RAG improves command generation

echo "========================================"
echo "L-AI-NUX-TOOL - RAG Impact Demo"
echo "========================================"
echo ""
echo "This will show side-by-side comparison:"
echo "  - Without RAG: Base model only"
echo "  - With RAG: Enhanced with documentation"
echo ""
echo "Perfect for demonstrating RAG's value!"
echo ""

if [ -z "$1" ]; then
    # Interactive mode if no argument
    echo "Starting interactive RAG demo..."
    echo ""
    python gen_cmd.py --show-rag-impact --interactive
else
    # Single command demo if argument provided
    echo "Demonstrating for: $*"
    echo ""
    python gen_cmd.py --show-rag-impact "$*"
fi

