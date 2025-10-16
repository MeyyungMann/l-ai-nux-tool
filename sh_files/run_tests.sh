#!/bin/bash
# Run tests for L-AI-NUX-TOOL

echo "========================================"
echo "L-AI-NUX-TOOL - Test Runner"
echo "========================================"
echo ""

case "$1" in
    quick)
        echo "Running quick tests..."
        python tests/quick_test.py
        ;;
    full)
        echo "Running full test suite..."
        python tests/test_suite.py
        ;;
    rag)
        echo "Running RAG tests..."
        python tests/test_rag.py
        ;;
    docker)
        echo "Running Docker tests..."
        echo "Note: Docker tests must be run inside Docker container"
        docker-compose run --rm lai-nux-tool-test python tests/test_docker.py
        ;;
    all)
        echo "Running ALL tests..."
        echo ""
        echo "=== Quick Tests ==="
        python tests/quick_test.py
        echo ""
        echo "=== Full Suite ==="
        python tests/test_suite.py
        echo ""
        echo "=== RAG Tests ==="
        python tests/test_rag.py
        ;;
    *)
        echo "Usage: ./run_tests.sh [quick|full|rag|docker|all]"
        echo ""
        echo "Options:"
        echo "  quick  - Fast smoke tests (10 seconds)"
        echo "  full   - Comprehensive test suite (60 seconds)"
        echo "  rag    - RAG-specific tests"
        echo "  docker - Docker environment tests (runs in Docker)"
        echo "  all    - Run all local tests"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh quick"
        echo "  ./run_tests.sh full"
        echo "  ./run_tests.sh all"
        exit 1
        ;;
esac

