#!/bin/bash
# Compare online and offline test results

echo "=================================================================="
echo "📊 ONLINE vs OFFLINE MODE COMPARISON"
echo "=================================================================="
echo ""

# Check if both reports exist
if [ ! -f "ONLINE_MODE_TEST_REPORT.md" ]; then
    echo "❌ Online mode report not found"
    echo "Run: ./docker_test_50_queries.sh"
    exit 1
fi

if [ ! -f "OFFLINE_MODE_TEST_REPORT.md" ]; then
    echo "❌ Offline mode report not found"
    echo "Run: ./docker_test_50_queries_offline.sh"
    exit 1
fi

echo "✅ Both reports found"
echo ""

# Extract key metrics
echo "=== SUCCESS RATES ==="
echo ""
echo "Online Mode:"
grep -A 1 "Success Rate" ONLINE_MODE_TEST_REPORT.md | head -2
echo ""
echo "Offline Mode:"
grep -A 1 "Success Rate" OFFLINE_MODE_TEST_REPORT.md | head -2
echo ""

echo "=== PERFORMANCE ==="
echo ""
echo "Online Mode Duration:"
grep "Duration" ONLINE_MODE_TEST_REPORT.md | head -1
echo ""
echo "Offline Mode Duration:"
grep "Duration" OFFLINE_MODE_TEST_REPORT.md | head -1
echo ""

echo "=== CATEGORY BREAKDOWN ==="
echo ""
echo "Online Mode:"
grep -A 10 "| Category" ONLINE_MODE_TEST_REPORT.md | head -11
echo ""
echo "Offline Mode:"
grep -A 10 "| Category" OFFLINE_MODE_TEST_REPORT.md | head -11
echo ""

echo "=== RAG ANALYSIS (Offline Only) ==="
echo ""
grep -A 5 "RAG Hit Rate" OFFLINE_MODE_TEST_REPORT.md | head -6
echo ""

# Calculate difference
online_success=$(grep "Success Rate" ONLINE_MODE_TEST_REPORT.md | head -1 | grep -oP '\d+\.\d+' | head -1)
offline_success=$(grep "Success Rate" OFFLINE_MODE_TEST_REPORT.md | head -1 | grep -oP '\d+\.\d+' | head -1)

if [ ! -z "$online_success" ] && [ ! -z "$offline_success" ]; then
    diff=$(echo "$offline_success - $online_success" | bc)
    echo "=== OVERALL COMPARISON ==="
    echo ""
    echo "Online Success:  ${online_success}%"
    echo "Offline Success: ${offline_success}%"
    echo "Difference:      ${diff}%"
    echo ""
    
    # Interpretation
    if (( $(echo "$diff > 5" | bc -l) )); then
        echo "🎉 Offline mode outperforming online! Excellent RAG cache."
    elif (( $(echo "$diff > -5" | bc -l) )); then
        echo "✅ Offline mode comparable to online. Good RAG cache."
    elif (( $(echo "$diff > -15" | bc -l) )); then
        echo "⚠️  Offline mode below online. RAG cache needs improvement."
    else
        echo "❌ Offline mode significantly behind. Critical RAG gaps."
    fi
    echo ""
fi

echo "=== RECOMMENDATIONS ==="
echo ""
echo "Offline Mode Missing Commands:"
grep -A 20 "### Missing Command Recommendations" OFFLINE_MODE_TEST_REPORT.md | tail -15
echo ""

echo "=================================================================="
echo "📝 Full Reports:"
echo "  - Online:  ONLINE_MODE_TEST_REPORT.md"
echo "  - Offline: OFFLINE_MODE_TEST_REPORT.md"
echo "=================================================================="

