#!/bin/bash
# Run 50-query comprehensive test for OFFLINE MODE inside Docker

echo "=================================================================="
echo "🐳 Running 50-Query Test Suite in Docker - OFFLINE MODE"
echo "=================================================================="
echo ""

# Check if RAG cache exists
if [ ! -d "rag_cache" ]; then
    echo "❌ Error: rag_cache directory not found"
    echo "Please build RAG cache first:"
    echo "  python -m src.rag_engine --build-cache"
    exit 1
fi

echo "✓ Found rag_cache directory"

# Check if model cache exists
if [ ! -d "quantized_model_cache" ] && [ ! -d "model_cache" ]; then
    echo "⚠️  Warning: No model cache found"
    echo "First run will download models (this may take time)"
fi

echo "✓ Building Docker image..."
echo ""

# Build the image
docker-compose build lai-nux-tool-test-50-offline

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo ""
echo "✓ Image built successfully"
echo "✓ Running 50-query OFFLINE test..."
echo ""

# Run the test inside Docker in offline mode
docker-compose run --rm lai-nux-tool-test-50-offline

if [ $? -ne 0 ]; then
    echo "❌ Test execution failed"
    exit 1
fi

echo ""
echo "=================================================================="
echo "✅ Offline Test completed!"
echo "📝 Report saved to: OFFLINE_MODE_TEST_REPORT.md"
echo "=================================================================="
echo ""
echo "To view the report:"
echo "  cat OFFLINE_MODE_TEST_REPORT.md"
echo ""
echo "To view summary:"
echo "  head -80 OFFLINE_MODE_TEST_REPORT.md"
echo ""
echo "To compare with online mode:"
echo "  echo '=== ONLINE MODE ==='"
echo "  grep 'Success Rate' ONLINE_MODE_TEST_REPORT.md | head -1"
echo "  echo '=== OFFLINE MODE ==='"
echo "  grep 'Success Rate' OFFLINE_MODE_TEST_REPORT.md | head -1"
echo ""

