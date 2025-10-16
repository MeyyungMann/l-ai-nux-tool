#!/bin/bash
# Run 50-query comprehensive test inside Docker

echo "=================================================================="
echo "🐳 Running 50-Query Test Suite in Docker"
echo "=================================================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Please create .env file with your OPENAI_API_KEY"
    exit 1
fi

echo "✓ Found .env file"
echo "✓ Building Docker image..."
echo ""

# Build the image
docker-compose build lai-nux-tool-test-50

echo ""
echo "✓ Image built successfully"
echo "✓ Running 50-query test..."
echo ""

# Run the test inside Docker
docker-compose run --rm lai-nux-tool-test-50

echo ""
echo "=================================================================="
echo "✅ Test completed!"
echo "📝 Report saved to: ONLINE_MODE_TEST_REPORT.md"
echo "=================================================================="
echo ""
echo "To view the report:"
echo "  cat ONLINE_MODE_TEST_REPORT.md"
echo ""
echo "To view summary:"
echo "  head -50 ONLINE_MODE_TEST_REPORT.md"

