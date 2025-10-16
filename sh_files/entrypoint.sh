#!/bin/bash
echo "🐧 Linux AI Command Generator - Docker Environment"
echo "=================================================="
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "Current GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo ""

# Create test environment (only if it doesn't exist or is incomplete)
if [ ! -d "/app/test_env" ] || [ ! -f "/app/test_env/documents/reports/sample_report.txt" ]; then
    echo "🔧 Creating test environment..."
    /create_test_env.sh
else
    echo "📁 Test environment already exists, skipping creation..."
fi
echo ""

if [ "$1" = "interactive" ]; then
    echo "🚀 Starting interactive mode..."
    echo "📁 Test environment available at: /app/test_env/"
    echo "💡 Try commands like: \"List files in test environment\" or \"Find all .txt files\""
    echo ""
    python gen_cmd.py --interactive
elif [ "$1" = "test" ]; then
    echo "🧪 Running tests..."
    python tests/test_suite.py
elif [ "$1" = "quick-test" ]; then
    echo "⚡ Running quick tests..."
    python tests/quick_test.py
else
    echo "💡 Available commands:"
    echo "  docker run -it lai-nux-tool interactive  # Interactive mode"
    echo "  docker run -it lai-nux-tool test         # Run tests"
    echo "  docker run -it lai-nux-tool quick-test   # Quick tests"
    echo "  docker run -it lai-nux-tool bash         # Shell access"
    echo ""
    echo "🔧 Example usage:"
    echo "  docker run -it lai-nux-tool interactive"
    echo "  docker run -it lai-nux-tool bash"
    echo ""
    exec "$@"
fi
