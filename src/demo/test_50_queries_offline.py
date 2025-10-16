"""Comprehensive 50-query test suite for offline mode.

Tests RAG cache performance and generates detailed analysis comparing with online mode.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from src.config import Config
from src.llm_engine import LLMEngine
from src.command_parser import CommandParser

# Load environment variables
load_dotenv()

# Same 50 queries as online test for direct comparison
TEST_QUERIES = [
    # File Operations (10)
    ("File Ops", "list all files including hidden ones"),
    ("File Ops", "create a new file named test.txt"),
    ("File Ops", "copy file1.txt to file2.txt"),
    ("File Ops", "move old.log to archive folder"),
    ("File Ops", "delete all txt files in current directory"),
    ("File Ops", "rename document.pdf to report.pdf"),
    ("File Ops", "create backup of config.json"),
    ("File Ops", "show file permissions of script.sh"),
    ("File Ops", "make script.sh executable"),
    ("File Ops", "change owner of file.txt to root"),
    
    # Directory Operations (8)
    ("Directory", "show current working directory"),
    ("Directory", "create a new directory called projects"),
    ("Directory", "list directory tree structure"),
    ("Directory", "remove empty directory temp"),
    ("Directory", "create nested directories a/b/c"),
    ("Directory", "change to home directory"),
    ("Directory", "count files in current directory"),
    ("Directory", "show disk usage of current directory"),
    
    # Search & Find (10)
    ("Search", "find all jpg files recursively"),
    ("Search", "find files modified in last 7 days"),
    ("Search", "find files larger than 100MB"),
    ("Search", "find empty files"),
    ("Search", "search for text 'error' in all log files"),
    ("Search", "find files owned by current user"),
    ("Search", "locate python executable"),
    ("Search", "find duplicate files by name"),
    ("Search", "search for pattern in all sh files"),
    ("Search", "find files with 777 permissions"),
    
    # Text Processing (8)
    ("Text", "show first 10 lines of log.txt"),
    ("Text", "show last 20 lines of error.log"),
    ("Text", "count lines in file.txt"),
    ("Text", "count words in document.txt"),
    ("Text", "sort lines in names.txt alphabetically"),
    ("Text", "remove duplicate lines from data.txt"),
    ("Text", "replace 'old' with 'new' in file.txt"),
    ("Text", "extract column 2 from csv.txt"),
    
    # System Info (7)
    ("System", "show system information"),
    ("System", "display current date and time"),
    ("System", "show memory usage"),
    ("System", "show cpu information"),
    ("System", "check disk space"),
    ("System", "show logged in users"),
    ("System", "display system uptime"),
    
    # Process Management (7)
    ("Process", "list all running processes"),
    ("Process", "show top 10 cpu consuming processes"),
    ("Process", "find process by name nginx"),
    ("Process", "kill process with pid 1234"),
    ("Process", "show processes using port 8080"),
    ("Process", "monitor system resources in real-time"),
    ("Process", "show memory usage per process"),
]


def categorize_failure(query, command, error_msg):
    """Categorize the type of failure."""
    if not command or command.strip() == "":
        return "EMPTY_OUTPUT", "RAG returned empty response"
    elif "{{" in command or "[[" in command:
        return "TEMPLATE_PLACEHOLDER", "Contains template placeholders"
    elif not any(cmd in command.split()[0] if command.split() else "" for cmd in ['ls', 'find', 'grep', 'cat', 'head', 'tail', 'awk', 'sed', 'ps', 'top', 'mkdir', 'touch', 'cp', 'mv', 'rm', 'cd', 'pwd', 'chmod', 'chown', 'df', 'du', 'kill', 'which', 'locate', 'wc', 'sort', 'uniq', 'date', 'free', 'uname', 'uptime', 'tree', 'rmdir']):
        return "INVALID_COMMAND", "Not a recognized Linux command"
    elif error_msg and "Invalid command structure" in error_msg:
        return "SYNTAX_ERROR", "Command syntax validation failed"
    elif command.startswith(('dir', 'copy', 'del', 'type')):
        return "WINDOWS_COMMAND", "Generated Windows command instead of Linux"
    else:
        return "RAG_MISS", "RAG cache didn't have good match"


def analyze_rag_coverage(query, command, success):
    """Analyze if query was likely found in RAG cache."""
    # Simple heuristic: successful commands likely came from RAG
    if success:
        return "RAG_HIT", "Successfully retrieved from cache"
    else:
        return "RAG_MISS", "No good match in cache"


def run_comprehensive_test():
    """Run comprehensive 50-query test suite in offline mode."""
    
    print("=" * 80)
    print("🧪 COMPREHENSIVE 50-QUERY TEST SUITE - OFFLINE MODE")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: Offline (RAG-based)")
    print(f"Model: Local CodeLlama-7b-Instruct")
    print("=" * 80)
    print()
    
    # Initialize offline mode
    config = Config()
    config.set('mode', 'rag')
    
    try:
        engine = LLMEngine(config)
        print("✅ Offline engine initialized successfully")
        print(f"📚 RAG cache loaded")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize offline engine: {e}")
        print("⚠️  Make sure RAG cache exists and model is available")
        return
    
    parser = CommandParser()
    
    # Results storage
    results = []
    category_stats = {}
    failure_types = {}
    rag_coverage = {"hits": 0, "misses": 0}
    missing_commands = []
    
    start_time = time.time()
    
    # Run tests
    for i, (category, query) in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/50] Testing: {query}")
        
        try:
            # Generate command using RAG
            command = engine.generate_command(query)
            
            # Try to parse
            try:
                parsed = parser.parse_command(command)
                success = True
                error = None
                failure_type = None
                failure_reason = None
                rag_status, rag_reason = analyze_rag_coverage(query, command, success)
                if rag_status == "RAG_HIT":
                    rag_coverage["hits"] += 1
                else:
                    rag_coverage["misses"] += 1
            except Exception as e:
                success = False
                parsed = command
                error = str(e)
                failure_type, failure_reason = categorize_failure(query, command, error)
                rag_status = "RAG_MISS"
                rag_reason = "Failed validation"
                rag_coverage["misses"] += 1
                
                # Track failure types
                if failure_type not in failure_types:
                    failure_types[failure_type] = []
                failure_types[failure_type].append((query, command, failure_reason))
                
                # Track missing commands for recommendations
                missing_commands.append({
                    "query": query,
                    "category": category,
                    "generated": command,
                    "reason": failure_reason
                })
            
            # Track category stats
            if category not in category_stats:
                category_stats[category] = {"total": 0, "success": 0, "failed": 0}
            category_stats[category]["total"] += 1
            if success:
                category_stats[category]["success"] += 1
            else:
                category_stats[category]["failed"] += 1
            
            # Store result
            results.append({
                "id": i,
                "category": category,
                "query": query,
                "command": parsed,
                "success": success,
                "error": error,
                "failure_type": failure_type,
                "failure_reason": failure_reason,
                "rag_status": rag_status,
                "rag_reason": rag_reason
            })
            
            status = "✅" if success else "❌"
            cmd_display = command[:60] if command else "(empty)"
            print(f"   {status} {cmd_display}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "id": i,
                "category": category,
                "query": query,
                "command": "",
                "success": False,
                "error": str(e),
                "failure_type": "EXCEPTION",
                "failure_reason": str(e),
                "rag_status": "ERROR",
                "rag_reason": "Exception during generation"
            })
            
            if category not in category_stats:
                category_stats[category] = {"total": 0, "success": 0, "failed": 0}
            category_stats[category]["total"] += 1
            category_stats[category]["failed"] += 1
            rag_coverage["misses"] += 1
        
        time.sleep(0.1)  # Small delay for clean output
    
    elapsed_time = time.time() - start_time
    
    # Calculate overall stats
    total_success = sum(1 for r in results if r["success"])
    total_failed = len(results) - total_success
    success_rate = (total_success / len(results)) * 100
    rag_hit_rate = (rag_coverage["hits"] / len(results)) * 100
    
    # Generate markdown report
    generate_markdown_report(
        results, 
        category_stats, 
        failure_types, 
        missing_commands,
        rag_coverage,
        elapsed_time, 
        success_rate,
        rag_hit_rate
    )
    
    print()
    print("=" * 80)
    print(f"✅ Test completed in {elapsed_time:.1f} seconds")
    print(f"📊 Success Rate: {success_rate:.1f}% ({total_success}/{len(results)})")
    print(f"📚 RAG Hit Rate: {rag_hit_rate:.1f}% ({rag_coverage['hits']}/{len(results)})")
    print(f"📝 Report saved to: OFFLINE_MODE_TEST_REPORT.md")
    print("=" * 80)


def load_online_results():
    """Load online mode results for comparison if available."""
    report_path = Path("ONLINE_MODE_TEST_REPORT.md")
    if report_path.exists():
        content = report_path.read_text(encoding='utf-8')
        # Parse success rate from online report
        for line in content.split('\n'):
            if 'Success Rate**:' in line:
                try:
                    rate = float(line.split('**Success Rate**: ')[1].split('%')[0])
                    return rate
                except:
                    pass
    return None


def generate_markdown_report(results, category_stats, failure_types, missing_commands, 
                            rag_coverage, elapsed_time, success_rate, rag_hit_rate):
    """Generate detailed markdown report for offline mode."""
    
    online_success_rate = load_online_results()
    
    report = f"""# Offline Mode - 50 Query Test Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Mode**: Offline (RAG-based)  
**Model**: Local CodeLlama-7b-Instruct  
**Duration**: {elapsed_time:.1f} seconds  
**Success Rate**: {success_rate:.1f}% ({sum(1 for r in results if r['success'])}/{len(results)})  
**RAG Hit Rate**: {rag_hit_rate:.1f}% ({rag_coverage['hits']}/{len(results)})
"""
    
    if online_success_rate:
        diff = success_rate - online_success_rate
        report += f"""**Online Comparison**: {diff:+.1f}% difference (Online: {online_success_rate:.1f}%)

"""
    
    report += """
---

## 📊 Summary Statistics

### Overall Results
"""
    report += f"""- ✅ **Successful**: {sum(1 for r in results if r['success'])} queries
- ❌ **Failed**: {sum(1 for r in results if not r['success'])} queries
- ⏱️ **Avg Time**: {elapsed_time/len(results):.2f}s per query
- 📚 **RAG Hits**: {rag_coverage['hits']} queries
- 🔍 **RAG Misses**: {rag_coverage['misses']} queries

"""
    
    report += """### By Category

| Category | Total | Success | Failed | Success Rate |
|----------|-------|---------|--------|--------------|
"""
    
    for category, stats in sorted(category_stats.items()):
        rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
        report += f"| {category} | {stats['total']} | {stats['success']} | {stats['failed']} | {rate:.1f}% |\n"
    
    report += f"""
---

## ✅ Successful Commands

"""
    
    for result in results:
        if result['success']:
            report += f"### {result['id']}. {result['query']}\n"
            report += f"- **Category**: {result['category']}\n"
            report += f"- **Command**: `{result['command']}`\n"
            report += f"- **RAG Status**: {result['rag_status']}\n"
            report += "\n"
    
    report += """---

## ❌ Failed Commands

"""
    
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        for result in failed_results:
            report += f"### {result['id']}. {result['query']}\n"
            report += f"- **Category**: {result['category']}\n"
            report += f"- **Generated**: `{result['command']}`\n"
            report += f"- **Failure Type**: {result['failure_type']}\n"
            report += f"- **Reason**: {result['failure_reason']}\n"
            report += f"- **RAG Status**: {result['rag_status']}\n"
            if result['error']:
                report += f"- **Error**: {result['error']}\n"
            report += "\n"
    else:
        report += "*No failures! All queries generated valid commands.* 🎉\n\n"
    
    report += """---

## 🔍 Failure Analysis

"""
    
    if failure_types:
        report += "### Failure Types Breakdown\n\n"
        for failure_type, failures in failure_types.items():
            report += f"#### {failure_type} ({len(failures)} occurrences)\n\n"
            for query, command, reason in failures:
                report += f"- **Query**: {query}\n"
                report += f"  - Generated: `{command}`\n"
                report += f"  - Reason: {reason}\n\n"
    else:
        report += "*No failures to analyze!* ✅\n\n"
    
    report += """---

## 📚 RAG Cache Analysis

### Cache Coverage
"""
    
    report += f"""
- **Total Queries**: {len(results)}
- **RAG Hits**: {rag_coverage['hits']} ({rag_hit_rate:.1f}%)
- **RAG Misses**: {rag_coverage['misses']} ({100-rag_hit_rate:.1f}%)
- **Cache Effectiveness**: {"Excellent ✅" if rag_hit_rate > 80 else "Good ✓" if rag_hit_rate > 60 else "Needs Improvement ⚠️"}

### Missing Command Recommendations

These queries failed and should be added to RAG cache:

"""
    
    if missing_commands:
        # Group by category
        by_category = {}
        for cmd in missing_commands:
            cat = cmd['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(cmd)
        
        for category, cmds in sorted(by_category.items()):
            report += f"#### {category} ({len(cmds)} missing)\n\n"
            for cmd in cmds:
                report += f"**{cmd['query']}**\n"
                report += f"- Reason: {cmd['reason']}\n"
                report += f"- Generated: `{cmd['generated']}`\n"
                report += f"- *Should add correct command to RAG*\n\n"
    else:
        report += "*No missing commands! RAG cache has excellent coverage.* 🎉\n\n"
    
    report += """---

## 🎯 Quality Analysis

### Command Quality Metrics

"""
    
    # Analyze command quality
    successful = [r for r in results if r['success']]
    if successful:
        avg_length = sum(len(r['command']) for r in successful) / len(successful)
        commands_with_flags = sum(1 for r in successful if '-' in r['command'])
        commands_with_quotes = sum(1 for r in successful if ('"' in r['command'] or "'" in r['command']))
        
        report += f"- **Average Command Length**: {avg_length:.1f} characters\n"
        report += f"- **Commands with Flags**: {commands_with_flags}/{len(successful)} ({commands_with_flags/len(successful)*100:.1f}%)\n"
        report += f"- **Commands with Quotes**: {commands_with_quotes}/{len(successful)} ({commands_with_quotes/len(successful)*100:.1f}%)\n\n"
        
        report += """### Command Complexity Distribution

"""
        
        simple_commands = sum(1 for r in successful if len(r['command'].split()) <= 2)
        medium_commands = sum(1 for r in successful if 3 <= len(r['command'].split()) <= 5)
        complex_commands = sum(1 for r in successful if len(r['command'].split()) > 5)
        
        report += f"- **Simple** (1-2 words): {simple_commands} commands\n"
        report += f"- **Medium** (3-5 words): {medium_commands} commands\n"
        report += f"- **Complex** (6+ words): {complex_commands} commands\n\n"
    
    report += """---

## 📋 Complete Test Results

| # | Category | Query | Command | Status | RAG |
|---|----------|-------|---------|--------|-----|
"""
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        rag_icon = "🎯" if result['rag_status'] == "RAG_HIT" else "❓"
        command_display = result['command'][:40] + "..." if len(result['command']) > 40 else result['command']
        query_display = result['query'][:35] + "..." if len(result['query']) > 35 else result['query']
        report += f"| {result['id']} | {result['category']} | {query_display} | `{command_display}` | {status} | {rag_icon} |\n"
    
    report += f"""

---

## 🆚 Comparison with Online Mode

"""
    
    if online_success_rate:
        diff = success_rate - online_success_rate
        diff_symbol = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
        
        report += f"""### Performance Comparison

| Metric | Offline | Online | Difference |
|--------|---------|--------|------------|
| Success Rate | {success_rate:.1f}% | {online_success_rate:.1f}% | {diff:+.1f}% {diff_symbol} |
| Avg Time/Query | {elapsed_time/len(results):.2f}s | ~1.0s* | {elapsed_time/len(results)-1.0:+.2f}s |

*Estimated based on typical API response times

### Analysis

"""
        if diff > 5:
            report += "✅ **Offline mode performing better than online!** Excellent RAG cache coverage.\n\n"
        elif diff > -5:
            report += "✅ **Offline mode performance comparable to online.** Good RAG cache.\n\n"
        elif diff > -15:
            report += "⚠️ **Offline mode underperforming online.** RAG cache needs improvement.\n\n"
        else:
            report += "❌ **Offline mode significantly behind online.** Critical RAG cache gaps.\n\n"
    else:
        report += "*Online mode results not available for comparison. Run online test first.*\n\n"
    
    report += """---

## 🚀 Recommendations

### Immediate Actions

"""
    
    if rag_hit_rate < 70:
        report += f"1. **Expand RAG Cache** - Current hit rate ({rag_hit_rate:.1f}%) is below target (70%+)\n"
        report += f"2. **Add {len(missing_commands)} Missing Commands** - See failure analysis above\n"
        report += "3. **Run Online+RAG Mode** - Use execution fallback to learn from successful commands\n"
    elif rag_hit_rate < 85:
        report += f"1. **Good Coverage** - {rag_hit_rate:.1f}% hit rate is solid\n"
        report += f"2. **Add {len(missing_commands)} Missing Commands** - To reach 90%+ coverage\n"
        report += "3. **Monitor Usage** - Track which commands users request most\n"
    else:
        report += f"1. **Excellent Coverage!** - {rag_hit_rate:.1f}% hit rate\n"
        report += "2. **Fine-tune Edge Cases** - Address remaining failures\n"
        report += "3. **Maintain Cache** - Keep RAG updated with new patterns\n"
    
    report += f"""

### RAG Cache Improvement Commands

```python
# Add missing commands to RAG cache
from src.rag_engine import get_rag_engine

rag = get_rag_engine()

# Priority additions (failed queries):
"""
    
    for cmd in missing_commands[:5]:
        report += f'# rag.add_manual_entry("{cmd["query"]}", "CORRECT_COMMAND_HERE")\n'
    
    report += """
# Then rebuild cache
rag.rebuild_cache()
```

### Testing Strategy

1. **Compare with Online** - Run online test to see gaps
2. **Add Missing Commands** - Use execution fallback or manual additions
3. **Retest** - Run this test again to measure improvement
4. **Iterate** - Continue until 90%+ success rate achieved

---

## 💡 Insights & Next Steps

### What Worked Well
"""
    
    if success_rate > 80:
        report += f"- ✅ High success rate ({success_rate:.1f}%) shows RAG cache is effective\n"
    elif success_rate > 60:
        report += f"- ✓ Decent success rate ({success_rate:.1f}%) - room for improvement\n"
    else:
        report += f"- ⚠️ Success rate ({success_rate:.1f}%) needs significant improvement\n"
    
    best_category = max(category_stats.items(), key=lambda x: x[1]['success'] / x[1]['total'] if x[1]['total'] > 0 else 0)
    report += f"- ✅ Best category: {best_category[0]} ({best_category[1]['success']}/{best_category[1]['total']})\n"
    report += f"- ✅ No API costs - completely offline\n"
    
    report += """
### Areas for Improvement
"""
    
    if failure_types:
        worst_category = min(category_stats.items(), key=lambda x: x[1]['success'] / x[1]['total'] if x[1]['total'] > 0 else 1)
        report += f"- ⚠️ Weakest category: {worst_category[0]} ({worst_category[1]['success']}/{worst_category[1]['total']})\n"
        report += f"- ⚠️ {len(failure_types)} different failure types\n"
        for failure_type, failures in sorted(failure_types.items(), key=lambda x: len(x[1]), reverse=True):
            report += f"  - {failure_type}: {len(failures)} occurrences\n"
    
    report += f"""
### Performance Impact

- **Speed**: Offline mode is {"faster" if elapsed_time/len(results) < 1.0 else "slower"} than online (no API latency)
- **Cost**: $0.00 (vs ~${len(results) * 0.0001:.4f} for online mode)
- **Reliability**: Works without internet connection
- **Privacy**: All processing local, no data sent externally

---

## 📈 Historical Tracking

*Track improvements over time*

| Date | Success Rate | RAG Hit Rate | Missing Commands |
|------|--------------|--------------|------------------|
| {datetime.now().strftime('%Y-%m-%d')} | {success_rate:.1f}% | {rag_hit_rate:.1f}% | {len(missing_commands)} |

---

**Test Status**: ✅ Complete  
**Mode**: Offline (RAG)  
**Next Steps**: {"Add missing commands to RAG" if missing_commands else "Monitor and maintain"}
"""
    
    # Save report
    report_path = Path("OFFLINE_MODE_TEST_REPORT.md")
    report_path.write_text(report, encoding='utf-8')
    
    return results, category_stats, missing_commands


if __name__ == "__main__":
    run_comprehensive_test()

