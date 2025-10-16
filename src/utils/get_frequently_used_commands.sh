#!/bin/bash
# Get most frequently used commands from your shell history

echo "📊 Your Most Frequently Used Commands:"
echo "======================================"

# Get unique commands from history
history | awk '{print $2}' | sort | uniq -c | sort -rn | head -30

echo ""
echo "💡 TIP: Use these commands with fetch_man_pages.py"
echo ""
echo "Example:"
echo "python src/utils/fetch_man_pages.py --commands \\"
history | awk '{print $2}' | sort | uniq -c | sort -rn | head -20 | awk '{print $2}' | tr '\n' ' '
echo ""

