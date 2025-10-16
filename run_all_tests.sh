#!/bin/bash
# Run both online and offline tests, then compare results

echo "=================================================================="
echo "🧪 Running Complete Test Suite - Online + Offline"
echo "=================================================================="
echo ""

# Check prerequisites
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found (needed for online test)"
    echo "Proceeding with offline test only..."
    SKIP_ONLINE=true
fi

if [ ! -d "rag_cache" ]; then
    echo "❌ Error: rag_cache not found (needed for offline test)"
    echo "Please build RAG cache first:"
    echo "  python -m src.rag_engine --build-cache"
    exit 1
fi

echo "✅ Prerequisites checked"
echo ""

# Run online test
if [ "$SKIP_ONLINE" != "true" ]; then
    echo "=================================================================="
    echo "🌐 STEP 1/3: Running Online Mode Test (50 queries)"
    echo "=================================================================="
    echo ""
    ./docker_test_50_queries.sh
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Online test failed, continuing with offline test..."
    fi
    echo ""
else
    echo "⏭️  Skipping online test (no .env file)"
    echo ""
fi

# Run offline test
echo "=================================================================="
echo "📚 STEP 2/3: Running Offline Mode Test (50 queries)"
echo "=================================================================="
echo ""
./docker_test_50_queries_offline.sh

if [ $? -ne 0 ]; then
    echo "❌ Offline test failed"
    exit 1
fi
echo ""

# Compare results
if [ "$SKIP_ONLINE" != "true" ] && [ -f "ONLINE_MODE_TEST_REPORT.md" ] && [ -f "OFFLINE_MODE_TEST_REPORT.md" ]; then
    echo "=================================================================="
    echo "📊 STEP 3/3: Comparing Results"
    echo "=================================================================="
    echo ""
    ./compare_test_results.sh
else
    echo "=================================================================="
    echo "📝 STEP 3/3: Test Results"
    echo "=================================================================="
    echo ""
    echo "Offline test completed successfully!"
    echo ""
    if [ ! -f "ONLINE_MODE_TEST_REPORT.md" ]; then
        echo "💡 Run online test for comparison:"
        echo "   ./docker_test_50_queries.sh"
    fi
fi

echo ""
echo "=================================================================="
echo "✅ All Tests Complete!"
echo "=================================================================="
echo ""
echo "📝 Generated Reports:"
if [ -f "ONLINE_MODE_TEST_REPORT.md" ]; then
    echo "  ✅ ONLINE_MODE_TEST_REPORT.md"
fi
if [ -f "OFFLINE_MODE_TEST_REPORT.md" ]; then
    echo "  ✅ OFFLINE_MODE_TEST_REPORT.md"
fi
echo ""
echo "🎯 Next Steps:"
echo "  1. Review reports:"
echo "     cat OFFLINE_MODE_TEST_REPORT.md"
if [ -f "ONLINE_MODE_TEST_REPORT.md" ]; then
    echo "     cat ONLINE_MODE_TEST_REPORT.md"
fi
echo ""
echo "  2. Check success rates:"
echo "     grep 'Success Rate' *_TEST_REPORT.md"
echo ""
echo "  3. If offline success < 70%, add missing commands from report"
echo ""
echo "=================================================================="

