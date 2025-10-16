#!/bin/bash
# Fetch man pages from inside Docker container

echo "📚 Fetching Man Pages from Docker..."
echo "===================================="
echo ""
echo "This will:"
echo "1. Start a Docker container"
echo "2. Fetch 100 common command man pages"
echo "3. Save to doc_cache/linux_docs.json"
echo "4. Exit automatically"
echo ""

docker-compose run --rm lai-nux-tool-shell python src/utils/fetch_man_pages.py --auto --limit 100

echo ""
echo "✅ Man pages fetched!"
echo "📁 Saved to: doc_cache/linux_docs.json"
echo "🔄 RAG will auto-rebuild on next use"

