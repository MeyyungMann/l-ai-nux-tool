#!/usr/bin/env python3
"""
Comprehensive RAG Test Suite - 100 Command Queries
Tests RAG system with diverse real-world queries
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.llm_engine import LLMEngine
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# 100 diverse test queries organized by category
TEST_QUERIES = {
    "File Listing (10)": [
        "list all files",
        "list files",
        "list files by time",
        "list files by size",
        "show hidden files",
        "list files recursively",
        "list only directories",
        "list files with details",
        "list files newest first",
        "list files oldest first",
    ],
    
    "File Search (10)": [
        "find txt files",
        "find large files",
        "find files modified today",
        "find files by name",
        "find empty files",
        "find files larger than 100MB",
        "find all python files",
        "find files in subdirectories",
        "find files modified in last 7 days",
        "find files by extension",
    ],
    
    "File Operations (10)": [
        "create a file",
        "delete a file",
        "copy file to directory",
        "move file to another location",
        "rename a file",
        "create directory",
        "remove empty directory",
        "copy directory recursively",
        "create multiple directories",
        "remove directory with contents",
    ],
    
    "Text Search (10)": [
        "search for text in files",
        "find word in file",
        "search recursively for pattern",
        "count occurrences of word",
        "search case insensitive",
        "search for multiple patterns",
        "search and show line numbers",
        "search in specific file types",
        "search excluding directories",
        "search with context lines",
    ],
    
    "Text Processing (10)": [
        "display file contents",
        "show first 10 lines",
        "show last 20 lines",
        "count lines in file",
        "count words in file",
        "sort file contents",
        "remove duplicate lines",
        "replace text in file",
        "extract column from file",
        "merge two files",
    ],
    
    "Archive Operations (10)": [
        "compress files",
        "extract tar file",
        "create tar archive",
        "compress directory",
        "extract zip file",
        "create zip archive",
        "compress with gzip",
        "decompress gz file",
        "list archive contents",
        "add files to archive",
    ],
    
    "Process Management (10)": [
        "show running processes",
        "kill process by name",
        "find process by name",
        "show process tree",
        "show top processes",
        "kill process by id",
        "run command in background",
        "show memory usage",
        "show cpu usage",
        "list all processes",
    ],
    
    "System Information (10)": [
        "show disk space",
        "show disk usage",
        "show system information",
        "show current user",
        "show hostname",
        "show kernel version",
        "show uptime",
        "show memory info",
        "show cpu info",
        "show mounted filesystems",
    ],
    
    "Network Operations (10)": [
        "download file from url",
        "check network connectivity",
        "show ip address",
        "test connection to host",
        "download with curl",
        "show network interfaces",
        "show open ports",
        "check dns resolution",
        "show routing table",
        "transfer file via ssh",
    ],
    
    "Permissions (10)": [
        "change file permissions",
        "make file executable",
        "change file owner",
        "show file permissions",
        "set directory permissions",
        "change group ownership",
        "add execute permission",
        "remove write permission",
        "set permissions recursively",
        "show file ownership",
    ],
}

def test_rag_system(mode='rag'):
    """Test RAG system with 100 queries."""
    
    mode_names = {
        'offline': '🔌 Offline',
        'rag': '📚 Offline+RAG',
        'online': '🌐 Online',
        'online-rag': '🌐📚 Online+RAG'
    }
    
    console.print(f"\n[bold blue]🧪 {mode_names.get(mode, mode)} - 100 Command Test Suite[/bold blue]")
    console.print("=" * 70)
    
    # Initialize
    console.print(f"\n[yellow]Initializing {mode} mode...[/yellow]")
    config = Config()
    config.set('mode', mode)
    
    try:
        llm_engine = LLMEngine(config)
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize: {e}[/red]")
        return
    
    # Test results
    results = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'by_category': {},
        'errors': [],
        'timings': []
    }
    
    # Run tests
    console.print("\n[green]Running 100 test queries...[/green]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        
        for category, queries in TEST_QUERIES.items():
            task = progress.add_task(f"Testing {category}", total=len(queries))
            
            category_results = {
                'success': 0,
                'failed': 0,
                'avg_time': 0,
                'commands': []
            }
            
            for query in queries:
                results['total'] += 1
                
                try:
                    # Time the generation
                    start = time.time()
                    command = llm_engine.generate_command(query)
                    elapsed = time.time() - start
                    
                    results['timings'].append(elapsed)
                    category_results['commands'].append({
                        'query': query,
                        'command': command,
                        'time': elapsed,
                        'success': True
                    })
                    
                    results['success'] += 1
                    category_results['success'] += 1
                    
                except Exception as e:
                    results['failed'] += 1
                    category_results['failed'] += 1
                    results['errors'].append({
                        'query': query,
                        'error': str(e)
                    })
                    category_results['commands'].append({
                        'query': query,
                        'command': None,
                        'time': 0,
                        'success': False,
                        'error': str(e)
                    })
                
                progress.update(task, advance=1)
            
            # Calculate category average time
            successful_times = [c['time'] for c in category_results['commands'] if c['success']]
            if successful_times:
                category_results['avg_time'] = sum(successful_times) / len(successful_times)
            
            results['by_category'][category] = category_results
    
    # Display results
    console.print("\n" + "=" * 70)
    console.print("[bold green]📊 TEST RESULTS[/bold green]")
    console.print("=" * 70)
    
    # Summary table
    summary_table = Table(title="Overall Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Queries", str(results['total']))
    summary_table.add_row("Successful", f"{results['success']} ({results['success']/results['total']*100:.1f}%)")
    summary_table.add_row("Failed", f"{results['failed']} ({results['failed']/results['total']*100:.1f}%)")
    
    if results['timings']:
        avg_time = sum(results['timings']) / len(results['timings'])
        min_time = min(results['timings'])
        max_time = max(results['timings'])
        summary_table.add_row("Avg Response Time", f"{avg_time*1000:.1f}ms")
        summary_table.add_row("Min Time", f"{min_time*1000:.1f}ms")
        summary_table.add_row("Max Time", f"{max_time*1000:.1f}ms")
    
    console.print(summary_table)
    
    # Category breakdown
    console.print("\n[bold]📋 Results by Category:[/bold]\n")
    
    category_table = Table()
    category_table.add_column("Category", style="cyan")
    category_table.add_column("Success", style="green")
    category_table.add_column("Failed", style="red")
    category_table.add_column("Avg Time", style="yellow")
    
    for category, cat_results in results['by_category'].items():
        category_table.add_row(
            category,
            str(cat_results['success']),
            str(cat_results['failed']),
            f"{cat_results['avg_time']*1000:.1f}ms" if cat_results['avg_time'] > 0 else "N/A"
        )
    
    console.print(category_table)
    
    # Sample successful commands
    console.print("\n[bold]✅ Sample Successful Commands:[/bold]\n")
    
    sample_table = Table()
    sample_table.add_column("Query", style="cyan", width=30)
    sample_table.add_column("Generated Command", style="green", width=35)
    
    # Show 10 samples from different categories
    sample_count = 0
    for category, cat_results in results['by_category'].items():
        for cmd in cat_results['commands']:
            if cmd['success'] and sample_count < 10:
                sample_table.add_row(
                    cmd['query'][:30],
                    cmd['command'][:35] if cmd['command'] else "N/A"
                )
                sample_count += 1
        if sample_count >= 10:
            break
    
    console.print(sample_table)
    
    # Show errors if any
    if results['errors']:
        console.print(f"\n[bold red]❌ Errors ({len(results['errors'])}):[/bold red]\n")
        for i, error in enumerate(results['errors'][:5], 1):
            console.print(f"{i}. [red]{error['query']}[/red]")
            console.print(f"   Error: {error['error'][:80]}...")
    
    # Performance analysis
    console.print("\n[bold]⚡ Performance Analysis:[/bold]")
    if results['timings']:
        fast_queries = sum(1 for t in results['timings'] if t < 0.1)
        medium_queries = sum(1 for t in results['timings'] if 0.1 <= t < 0.5)
        slow_queries = sum(1 for t in results['timings'] if t >= 0.5)
        
        console.print(f"  Fast (<100ms): {fast_queries} queries")
        console.print(f"  Medium (100-500ms): {medium_queries} queries")
        console.print(f"  Slow (>500ms): {slow_queries} queries")
    
    # Save detailed results
    results_file = Path("tests/rag_test_results.json")
    import json
    with open(results_file, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {
            'summary': {
                'total': results['total'],
                'success': results['success'],
                'failed': results['failed'],
                'success_rate': results['success']/results['total']*100 if results['total'] > 0 else 0,
                'avg_time_ms': sum(results['timings'])/len(results['timings'])*1000 if results['timings'] else 0
            },
            'by_category': {
                cat: {
                    'success': r['success'],
                    'failed': r['failed'],
                    'avg_time_ms': r['avg_time']*1000,
                    'commands': r['commands']
                }
                for cat, r in results['by_category'].items()
            },
            'errors': results['errors']
        }
        json.dump(json_results, f, indent=2)
    
    console.print(f"\n[dim]📁 Detailed results saved to: {results_file}[/dim]")
    
    # Final verdict
    console.print("\n" + "=" * 70)
    success_rate = results['success']/results['total']*100 if results['total'] > 0 else 0
    
    if success_rate >= 90:
        console.print("[bold green]✅ EXCELLENT: RAG system performing very well![/bold green]")
    elif success_rate >= 75:
        console.print("[bold yellow]⚠️  GOOD: RAG system working but needs improvement[/bold yellow]")
    else:
        console.print("[bold red]❌ NEEDS WORK: RAG system requires optimization[/bold red]")
    
    console.print("=" * 70)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG system with 100 queries")
    parser.add_argument('--mode', choices=['offline', 'rag', 'online', 'online-rag'], 
                       default='rag', help='Mode to test (default: rag)')
    args = parser.parse_args()
    
    try:
        results = test_rag_system(mode=args.mode)
        
        # Exit code based on success rate
        success_rate = results['success']/results['total']*100 if results['total'] > 0 else 0
        sys.exit(0 if success_rate >= 75 else 1)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Test failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)

