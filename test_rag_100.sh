#!/bin/bash
# Run 100-command test suite

MODE=${1:-rag}

echo "🧪 100 Command Test Suite - Mode: $MODE"
echo "========================================="
echo ""
echo "This will test $MODE mode with 100 diverse queries:"
echo "  - File listing (10 queries)"
echo "  - File search (10 queries)"
echo "  - File operations (10 queries)"
echo "  - Text search (10 queries)"
echo "  - Text processing (10 queries)"
echo "  - Archives (10 queries)"
echo "  - Process management (10 queries)"
echo "  - System info (10 queries)"
echo "  - Network (10 queries)"
echo "  - Permissions (10 queries)"
echo ""
echo "Expected time: 2-5 minutes"
echo ""

# Run in Docker with shell service
docker-compose run --rm lai-nux-tool-shell python tests/test_rag_100_commands.py --mode $MODE

echo ""
echo "✅ Test complete!"
echo "📁 Results saved to: tests/rag_test_results.json"

