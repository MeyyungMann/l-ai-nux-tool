#!/bin/bash
# Test RAG in Docker

echo "========================================"
echo "Testing RAG in Docker"
echo "========================================"
echo ""

case "$1" in
    showcase)
        echo "Running 33 RAG examples showcase..."
        docker-compose run --rm lai-nux-tool-shell python tests/test_rag_showcase.py
        ;;
    demo)
        if [ -z "$2" ]; then
            echo "Running interactive RAG demo..."
            docker-compose run --rm lai-nux-tool-shell python gen_cmd.py --show-rag-impact --interactive
        else
            echo "Running RAG demo for: $*"
            shift
            docker-compose run --rm lai-nux-tool-shell python gen_cmd.py --show-rag-impact "$*"
        fi
        ;;
    compare)
        echo "Running 4-mode comparison..."
        docker-compose run --rm lai-nux-tool-shell python tests/test_comparison.py
        ;;
    quick)
        echo "Running quick tests..."
        docker-compose run --rm lai-nux-tool-test python tests/quick_test.py
        ;;
    full)
        echo "Running full test suite..."
        docker-compose run --rm lai-nux-tool-test python tests/test_suite.py
        ;;
    all)
        echo "Running ALL tests..."
        echo ""
        echo "=== 1. Quick Tests ==="
        docker-compose run --rm lai-nux-tool-test python tests/quick_test.py
        echo ""
        echo "=== 2. RAG Showcase ==="
        docker-compose run --rm lai-nux-tool-shell python tests/test_rag_showcase.py
        echo ""
        echo "=== 3. Live Demo ==="
        docker-compose run --rm lai-nux-tool-shell python gen_cmd.py --show-rag-impact "find all text files"
        ;;
    *)
        echo "Usage: ./docker_test_rag.sh [command] [args]"
        echo ""
        echo "Commands:"
        echo "  showcase         - Show 33 RAG examples"
        echo "  demo [query]     - Live RAG demo (interactive if no query)"
        echo "  compare          - 4-mode comparison test"
        echo "  quick            - Quick smoke tests"
        echo "  full             - Full test suite"
        echo "  all              - Run all tests"
        echo ""
        echo "Examples:"
        echo "  ./docker_test_rag.sh showcase"
        echo "  ./docker_test_rag.sh demo 'find all text files'"
        echo "  ./docker_test_rag.sh demo"
        echo "  ./docker_test_rag.sh all"
        exit 1
        ;;
esac


