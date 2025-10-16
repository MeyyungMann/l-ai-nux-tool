#!/bin/bash
# Run tests in Docker

echo "========================================"
echo "L-AI-NUX-TOOL - Docker Tests"
echo "========================================"
echo ""

case "$1" in
    quick)
        echo "Running quick tests in Docker..."
        docker-compose run --rm lai-nux-tool-test python tests/quick_test.py
        ;;
    full)
        echo "Running full test suite in Docker..."
        docker-compose run --rm lai-nux-tool-test python tests/test_suite.py
        ;;
    rag)
        echo "Running RAG tests in Docker..."
        docker-compose run --rm lai-nux-tool-rag bash -c "python tests/test_rag.py"
        ;;
    docker)
        echo "Running Docker-specific tests..."
        docker-compose run --rm lai-nux-tool-test python tests/test_docker.py
        ;;
    all)
        echo "Running ALL tests in Docker..."
        echo ""
        echo "=== Quick Tests ==="
        docker-compose run --rm lai-nux-tool-test python tests/quick_test.py
        echo ""
        echo "=== Full Suite ==="
        docker-compose run --rm lai-nux-tool-test python tests/test_suite.py
        echo ""
        echo "=== RAG Tests ==="
        docker-compose run --rm lai-nux-tool-rag bash -c "python tests/test_rag.py"
        echo ""
        echo "=== Docker Tests ==="
        docker-compose run --rm lai-nux-tool-test python tests/test_docker.py
        ;;
    *)
        echo "Usage: ./docker_test.sh [quick|full|rag|docker|all]"
        echo ""
        echo "Options:"
        echo "  quick  - Fast smoke tests in Docker"
        echo "  full   - Comprehensive test suite in Docker"
        echo "  rag    - RAG-specific tests in Docker"
        echo "  docker - Docker environment tests"
        echo "  all    - Run all tests in Docker"
        echo ""
        echo "Examples:"
        echo "  ./docker_test.sh quick"
        echo "  ./docker_test.sh full"
        echo "  ./docker_test.sh all"
        exit 1
        ;;
esac

