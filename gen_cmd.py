#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linux AI Command Generator (L-AI-NUX-TOOL)
Main CLI interface for generating Linux commands from natural language.
"""

import click
import os
import sys
import signal
import time
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from src.llm_engine import LLMEngine
from src.command_parser import CommandParser
from src.config import Config
from src.windows_compatibility import WindowsCompatibility
from src.shift_tab import ShiftTabCompletion
from src.enhanced_safety import EnhancedSafetySystem

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    shutdown_requested = True
    print("\n🛑 Shutdown requested. Press Ctrl+C again to force exit.")
    print("💡 You can continue with the next command or type 'exit' to quit.")

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

console = Console()

def determine_mode(mode, offline, online, config):
    """Determine the mode based on arguments and configuration."""
    
    # Priority: explicit mode > flags > config > default
    if mode:
        return mode
    elif offline:
        return 'offline'
    elif online:
        return 'online'
    else:
        # Check if we have an API key
        api_key = config.get('api.api_key') or os.getenv('OPENAI_API_KEY')
        if api_key:
            return 'online'
        else:
            return 'offline'

def show_mode_selection_menu():
    """Show mode selection menu."""
    console.print(Panel(
        "🚀 Linux AI Command Generator - Mode Selection\n\n"
        "Choose your preferred mode:\n"
        "1. 🔌 Offline Mode - Use local Mixtral-8x7B model with 4-bit quantization\n"
        "   • No internet required after first setup\n"
        "   • Faster response times\n"
        "   • No API costs\n"
        "   • Uses your GPU with automatic CPU fallback\n\n"
        "2. 🌐 Online Mode - Use OpenAI-compatible API\n"
        "   • Requires API key\n"
        "   • Internet connection needed\n"
        "   • May have usage costs\n"
        "   • Works on any device\n\n"
        "3. 📚 RAG Mode - Mixtral-8x7B + Linux Documentation Retrieval\n"
        "   • Enhanced with Linux man pages and documentation\n"
        "   • Best accuracy for Linux commands\n"
        "   • Uses RAG (Retrieval-Augmented Generation)\n"
        "   • First run: downloads and indexes documentation\n\n"
        "4. 🔄 Compare All - Run all modes and compare results\n"
        "   • See outputs from all available modes\n"
        "   • Compare accuracy and approaches\n"
        "   • Best for evaluation\n\n"
        "5. ❓ Help - Show more information about modes",
        title="Mode Selection",
        border_style="blue"
    ))

def get_mode_from_user():
    """Get mode selection from user."""
    while True:
        choice = Prompt.ask(
            "\n[bold blue]Select mode (1-5) or 'q' to quit",
            choices=["1", "2", "3", "4", "5", "q", "Q"],
            default="1"
        )
        
        if choice.lower() == 'q':
            return None
        elif choice == "1":
            return 'offline'
        elif choice == "2":
            return 'online'
        elif choice == "3":
            return 'rag'
        elif choice == "4":
            return 'compare'
        elif choice == "5":
            show_mode_help()
        else:
            console.print("[red]Invalid choice. Please select 1-5 or 'q'.[/red]")

def show_mode_help():
    """Show detailed help about modes."""
    console.print(Panel(
        "📚 Mode Help\n\n"
        "🔌 OFFLINE MODE:\n"
        "• Uses Mixtral-8x7B-Instruct with 4-bit quantization\n"
        "• Optimized for GPU with automatic CPU fallback\n"
        "• First run downloads ~15GB model (one-time)\n"
        "• Subsequent runs use cached quantized model (~3GB)\n"
        "• Best for: Regular use, privacy, cost savings\n\n"
        "🌐 ONLINE MODE:\n"
        "• Uses OpenAI-compatible API (GPT-5-mini by default)\n"
        "• Requires API key (set OPENAI_API_KEY env var or use --api-key)\n"
        "• Works with OpenAI, Ollama, LiteLLM, etc.\n"
        "• Best for: Testing, when offline model isn't available\n\n"
        "📚 RAG MODE:\n"
        "• Combines Mixtral-8x7B with Linux documentation retrieval\n"
        "• Sources: man pages, tldr, cheat.sh, man7.org, GNU Coreutils\n"
        "• Uses semantic search to find relevant command examples\n"
        "• First run: downloads and indexes documentation (~10-15 min)\n"
        "• Cached files: ./rag_cache/ and ./doc_cache/\n"
        "• Best for: Highest accuracy, learning Linux commands\n\n"
        "🌐 ONLINE-RAG MODE:\n"
        "• Combines online API with RAG documentation context\n"
        "• Uses API for generation + RAG for context\n"
        "• Requires API key + RAG cache\n"
        "• Best for: Best of both worlds (API + documentation)\n\n"
        "🔄 COMPARE MODE:\n"
        "• Runs all available modes simultaneously\n"
        "• Shows side-by-side comparison of outputs\n"
        "• Useful for evaluating which mode works best\n"
        "• Takes longer as it runs multiple inferences\n"
        "• Best for: Testing, evaluation, comparing approaches\n\n"
        "💡 TIP: Try RAG mode for best accuracy with Linux commands!",
        title="Mode Help",
        border_style="green"
    ))

@click.command()
@click.argument('description', required=False)
@click.option('--mode', type=click.Choice(['online', 'offline', 'rag', 'online-rag', 'compare']), help='Choose mode: online, offline, rag, online-rag, or compare')
@click.option('--offline', is_flag=True, help='Use offline mode with local model (deprecated, use --mode offline)')
@click.option('--online', is_flag=True, help='Use online mode with API (deprecated, use --mode online)')
@click.option('--api-key', help='API key for online mode')
@click.option('--base-url', default='https://api.openai.com/v1', help='Base URL for API')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
@click.option('--show-rag-impact', is_flag=True, help='Show side-by-side comparison of with/without RAG')
def gen_cmd(description, mode, offline, online, api_key, base_url, interactive, show_rag_impact):
    """
    Generate Linux commands from natural language descriptions.
    
    MODES:
    - Offline: Uses local Mixtral-8x7B model with 4-bit quantization (default)
    - Online: Uses OpenAI-compatible API (requires API key)
    - RAG: Mixtral-8x7B + Linux documentation retrieval (best accuracy)
    - Compare: Run all modes and compare results
    
    EXAMPLES:
    gen-cmd "Remove all .tmp files recursively in the folder ./mystuff/"
    gen-cmd --mode rag "find all text files"
    gen-cmd --mode compare "list all files"
    gen-cmd --mode online --api-key YOUR_KEY "list all files"
    gen-cmd --interactive  # Interactive mode with mode selection
    gen-cmd --show-rag-impact "find all text files"  # Show RAG impact
    gen-cmd --show-rag-impact --interactive  # Interactive RAG demo
    """
    
    # Initialize configuration
    config = Config()
    
    # Determine mode from arguments
    selected_mode = determine_mode(mode, offline, online, config)
    
    # Set API key if provided
    if api_key:
        config.set('api.api_key', api_key)
    
    # Set API configuration if provided
    if base_url != 'https://api.openai.com/v1':
        config.set('api.base_url', base_url)
    
    # Set the mode in config
    config.set('mode', selected_mode)
    
    # Initialize Windows compatibility layer
    windows_compat = WindowsCompatibility()
    
    # Initialize LLM engine
    llm_engine = LLMEngine(config)
    
    # Initialize command parser
    command_parser = CommandParser()
    
    # Initialize enhanced systems
    shift_tab_completion = ShiftTabCompletion(llm_engine, command_parser)
    safety_system = EnhancedSafetySystem()
    
    # Handle compare mode
    if selected_mode == 'compare':
        if description:
            run_comparison_mode(description, command_parser, shift_tab_completion, safety_system, config)
            return
        elif interactive:
            run_interactive_comparison_mode(command_parser, shift_tab_completion, safety_system, config)
            return
    
    # Handle show RAG impact mode
    if show_rag_impact:
        if interactive:
            run_interactive_rag_demo_mode(command_parser, shift_tab_completion, safety_system, config)
            return
        elif description:
            show_rag_impact_comparison(description, command_parser, shift_tab_completion, safety_system, config)
            return
        else:
            console.print("[yellow]--show-rag-impact requires a description or --interactive flag[/yellow]")
            return
    
    if interactive:
        run_interactive_mode(llm_engine, command_parser, shift_tab_completion, safety_system, config)
    elif description:
        generate_command(description, llm_engine, command_parser, shift_tab_completion, safety_system, config)
    else:
        # No arguments provided - show mode selection menu
        show_mode_selection_menu()
        selected_mode = get_mode_from_user()
        
        if selected_mode is None:
            console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)
        
        # Update config with selected mode
        config.set('mode', selected_mode)
        
        # Reinitialize LLM engine with selected mode
        llm_engine = LLMEngine(config)
        shift_tab_completion = ShiftTabCompletion(llm_engine, command_parser)
        
        # Start interactive mode
        run_interactive_mode(llm_engine, command_parser, shift_tab_completion, safety_system, config)

def generate_command(description, llm_engine, command_parser, shift_tab_completion, safety_system, config):
    """Generate a single command from description with timeout support."""
    try:
        console.print(f"[blue]Processing:[/blue] {description}")
        
        # Add timeout warning for potentially slow operations
        if any(word in description.lower() for word in ['find', 'search', 'scan', 'duplicate', 'all files']):
            console.print("[dim]⏱️  This operation might take a while. Press Ctrl+C to cancel if needed.[/dim]")
        
        # Generate command using LLM with timeout protection
        generated_command = llm_engine.generate_command(description)
        
        # Check for shutdown request after generation
        if shutdown_requested:
            console.print("[yellow]🛑 Command generation cancelled by user.[/yellow]")
            return
        
        # Parse and validate command
        try:
            parsed_command = command_parser.parse_command(generated_command)
        except ValueError as e:
            # Fallback: attempt LLM-based repair when parser rejects the structure
            if str(e).startswith("Invalid command structure"):
                repaired = llm_engine.repair_command(description, generated_command)
                parsed_command = command_parser.parse_command(repaired)

                # If the repair came from online fallback, ask user to approve adding to RAG cache
                if llm_engine.has_pending_rag_seed():
                    seed = llm_engine.get_and_clear_pending_rag_seed()
                    if seed:
                        seed_description, seed_command = seed
                        # Confirm with the user
                        if click.confirm("Add this online-fixed command to RAG cache for future queries?"):
                            try:
                                # Store minimal helpful context
                                enriched_context = f"Known-good command for this task:\n\n{seed_command}"
                                if llm_engine.rag_engine is not None:
                                    llm_engine.rag_engine.store_context_for_query(seed_description, enriched_context, top_k=3)
                                    console.print("[green]✅ Added to RAG cache[/green]")

                                    # Persist approval for next-start promotion into docs
                                    try:
                                        from pathlib import Path
                                        import json as _json
                                        approvals_path = Path.home() / '.lai-nux-tool' / 'rag_cache' / 'user_approved.jsonl'
                                        approvals_path.parent.mkdir(parents=True, exist_ok=True)
                                        record = {
                                            'description': seed_description,
                                            'command': seed_command
                                        }
                                        with open(approvals_path, 'a', encoding='utf-8') as af:
                                            af.write(_json.dumps(record, ensure_ascii=False) + "\n")
                                    except Exception:
                                        pass
                                else:
                                    console.print("[yellow]RAG engine not available; cannot add to cache now[/yellow]")
                            except Exception as e2:
                                console.print(f"[yellow]Failed to add to RAG cache: {e2}[/yellow]")
                        else:
                            console.print("[dim]Skipped adding to RAG cache[/dim]")
            else:
                raise
        
        # Enhanced safety check
        is_safe, warning, safety_analysis = safety_system.validate_command_safety(parsed_command)
        
        # Display result with enhanced safety information
        if is_safe:
            console.print(Panel(
                parsed_command,
                title="Generated Command",
                border_style="green"
            ))
        else:
            console.print(Panel(
                parsed_command,
                title=f"Generated Command (⚠ {safety_analysis['severity'].upper()} Safety Warning)",
                border_style="red" if safety_analysis['severity'] == 'critical' else "yellow"
            ))
            console.print(f"[red]Warning:[/red] {warning}")
            
            # Show safety suggestions
            if safety_analysis['suggestions']:
                console.print("\n[yellow]Safety Suggestions:[/yellow]")
                for suggestion in safety_analysis['suggestions']:
                    console.print(f"  • {suggestion}")
            
            # Show safer alternatives
            alternatives = safety_system._get_safer_alternatives(parsed_command)
            if alternatives:
                console.print("\n[green]Safer Alternatives:[/green]")
                for alt in alternatives:
                    console.print(f"  • {alt}")
        
        # Ask for confirmation to execute
        if click.confirm("Execute this command?"):
            # Check for shutdown request before execution
            if shutdown_requested:
                console.print("[yellow]🛑 Command execution cancelled by user.[/yellow]")
                return
                
            # Add timeout warning for potentially slow commands
            if any(word in parsed_command.lower() for word in ['find /', 'find .', 'du -', 'df -', 'ps aux']):
                console.print("[dim]⏱️  This command might take a while. Press Ctrl+C to cancel if needed.[/dim]")
            
            # Use Windows compatibility layer for execution
            from src.windows_compatibility import WindowsCompatibility
            windows_compat = WindowsCompatibility()
            stdout, stderr, returncode = windows_compat.execute_command(parsed_command)
            
            if returncode == 0:
                console.print("[green]Command executed successfully![/green]")
                if stdout:
                    console.print(f"Output: {stdout}")
            else:
                console.print(f"[red]Command failed with return code {returncode}[/red]")
                if stderr:
                    console.print(f"Error: {stderr}")
                
                # Execution-based online fallback
                current_mode = config.get('mode', 'offline')
                if current_mode in ['offline', 'rag']:  # Only offer fallback for offline modes
                    console.print("\n[yellow]💡 The command failed to execute properly.[/yellow]")
                    if click.confirm("Would you like to try online mode to generate a better command?"):
                        console.print("[cyan]🌐 Using online mode to generate improved command...[/cyan]")
                        
                        try:
                            # Ensure online client is set up
                            if not hasattr(llm_engine, 'client'):
                                llm_engine._setup_online_client()
                            
                            # Use online repair with context about the failure
                            repaired = llm_engine.repair_command(description, parsed_command)
                            
                            if repaired and repaired.strip():
                                console.print(Panel(
                                    repaired,
                                    title="🌐 Online-Generated Command",
                                    border_style="cyan"
                                ))
                                
                                if click.confirm("Execute this online-generated command?"):
                                    stdout2, stderr2, returncode2 = windows_compat.execute_command(repaired)
                                    
                                    if returncode2 == 0:
                                        console.print("[green]✅ Online command executed successfully![/green]")
                                        if stdout2:
                                            console.print(f"Output: {stdout2}")
                                        
                                        # Offer to add successful online command to RAG cache
                                        if llm_engine.rag_engine and click.confirm("\n💾 Add this successful online command to RAG cache for future use?"):
                                            try:
                                                enriched_context = f"Known-good command for this task:\n\n{repaired}"
                                                llm_engine.rag_engine.store_context_for_query(description, enriched_context, top_k=3)
                                                console.print("[green]✅ Added to RAG cache - this will improve future results![/green]")
                                                
                                                # Save to user_approved.jsonl
                                                try:
                                                    from pathlib import Path
                                                    import json as _json
                                                    approvals_path = Path.home() / '.lai-nux-tool' / 'rag_cache' / 'user_approved.jsonl'
                                                    approvals_path.parent.mkdir(parents=True, exist_ok=True)
                                                    record = {
                                                        'description': description,
                                                        'command': repaired
                                                    }
                                                    with open(approvals_path, 'a', encoding='utf-8') as af:
                                                        af.write(_json.dumps(record, ensure_ascii=False) + "\n")
                                                except Exception:
                                                    pass
                                            except Exception as e3:
                                                console.print(f"[yellow]Failed to add to RAG cache: {e3}[/yellow]")
                                    else:
                                        console.print(f"[red]Online command also failed with return code {returncode2}[/red]")
                                        if stderr2:
                                            console.print(f"Error: {stderr2}")
                            else:
                                console.print("[yellow]⚠️  Online mode failed to generate a valid command[/yellow]")
                        
                        except Exception as e_online:
                            console.print(f"[red]Online fallback error: {e_online}[/red]")
                            console.print("[yellow]💡 Make sure OPENAI_API_KEY is set in your environment or .env file[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")

def run_interactive_mode(llm_engine, command_parser, shift_tab_completion, safety_system, config):
    """Run interactive mode with Shift+Tab support and mode switching."""
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    current_mode = config.get('mode', 'offline')
    
    console.print(Panel(
        f"Linux AI Command Generator - Interactive Mode\n"
        f"Current Mode: {'🔌 Offline' if current_mode == 'offline' else '🌐 Online' if current_mode == 'online' else '📚 Offline+RAG' if current_mode == 'rag' else '🌐📚 Online+RAG'}\n\n"
        "Commands:\n"
        "• Type your natural language description\n"
        "• 'mode' - Switch between offline/online/Offline+RAG/Online+RAG modes or compare all\n"
        "• 'preload' - Preload both modes for instant switching\n"
        "• 'cache' - Manage model cache\n"
        "• 'shift+tab' - Intelligent command completion\n"
        "• 'help' - Show help\n"
        "• 'exit', 'quit', 'q', 'bye', 'goodbye', 'stop', 'end' - Exit the program\n"
        "• Ctrl+C - Graceful shutdown (press twice to force exit)",
        title="Welcome",
        border_style="blue"
    ))
    
    console.print("[dim]💡 Tip: Type 'q', 'exit', or press Ctrl+C for graceful shutdown![/dim]")
    
    while True:
        try:
            # Check for shutdown request
            if shutdown_requested:
                console.print("\n[yellow]👋 Goodbye! Thanks for using Linux AI Command Generator![/yellow]")
                break
                
            # Create nice mode display for prompt
            mode_display = {'offline': '🔌 Offline', 'online': '🌐 Online', 'rag': '📚 Offline+RAG', 'online-rag': '🌐📚 Online+RAG'}.get(current_mode, current_mode)
            description = Prompt.ask(f"\n[bold blue]Describe the command you want to generate (mode: {mode_display})")
            
            # Sanitize input to remove invalid UTF-8 characters
            description = description.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            
            if description.lower() in ['exit', 'quit', 'q', 'bye', 'goodbye', 'stop', 'end']:
                console.print("\n[yellow]👋 Goodbye! Thanks for using Linux AI Command Generator![/yellow]")
                break
            
            # Handle special commands
            if description.lower() == 'mode':
                new_mode = switch_mode_interactive(config)
                if new_mode == 'compare':
                    # Run comparison mode
                    console.print("\n[blue]🔄 Comparison Mode[/blue]")
                    console.print("This will run your query through all available modes and show the differences.")
                    query = Prompt.ask("Enter your command description")
                    if query:
                        run_comparison_mode(query, command_parser, shift_tab_completion, safety_system, config)
                    continue
                elif new_mode:
                    # Switch mode without recreating the engine
                    llm_engine.switch_mode(new_mode)
                    shift_tab_completion = ShiftTabCompletion(llm_engine, command_parser)
                    current_mode = new_mode
                    console.print(f"[green]Switched to {current_mode} mode[/green]")
                continue
            
            elif description.lower() == 'compare':
                # Direct comparison mode
                console.print("\n[blue]🔄 Comparison Mode[/blue]")
                console.print("This will run your query through all available modes and show the differences.")
                query = Prompt.ask("Enter your command description")
                if query:
                    run_comparison_mode(query, command_parser, shift_tab_completion, safety_system, config)
                continue
            
            elif description.lower() == 'preload':
                console.print("[blue]Preloading both modes for instant switching...[/blue]")
                llm_engine.preload_both_modes()
                console.print("[green]✅ Both modes preloaded! You can now switch instantly.[/green]")
                continue
            
            elif description.lower() == 'cache':
                handle_cache_management(llm_engine)
                continue
            
            elif description.lower() == 'help':
                show_interactive_help()
                continue
            
            # Check for Shift+Tab completion request
            elif description.lower() in ['shift+tab', 'completion', 'complete']:
                partial_input = Prompt.ask("Enter partial command for completion")
                completions = shift_tab_completion.complete_command(partial_input)
                
                console.print("\n[green]Completion Suggestions:[/green]")
                for i, completion in enumerate(completions, 1):
                    console.print(f"  {i}. {completion}")
                
                if completions:
                    choice = Prompt.ask("Select completion (number) or press Enter to skip", default="")
                    if choice.isdigit() and 1 <= int(choice) <= len(completions):
                        selected_command = completions[int(choice) - 1]
                        shift_tab_completion.add_to_history(selected_command)
                        console.print(f"[green]Selected:[/green] {selected_command}")
                        continue
                
            generate_command(description, llm_engine, command_parser, shift_tab_completion, safety_system, config)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Goodbye! (Ctrl+C detected)[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")

def switch_mode_interactive(config):
    """Switch mode in interactive session."""
    current_mode = config.get('mode', 'offline')
    
    console.print(f"\n[blue]Current mode: {current_mode}[/blue]")
    console.print("1. Switch to offline mode")
    console.print("2. Switch to online mode")
    console.print("3. Switch to Offline+RAG mode")
    console.print("4. Switch to Online+RAG mode")
    console.print("5. Compare all modes")
    console.print("6. Cancel")
    
    choice = Prompt.ask("Select option (1-6)", choices=["1", "2", "3", "4", "5", "6"], default="6")
    
    if choice == "1":
        config.set('mode', 'offline')
        return 'offline'
    elif choice == "2":
        # Check for API key
        api_key = config.get('api.api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            api_key = Prompt.ask("Enter API key for online mode", password=True)
            if api_key:
                config.set('api.api_key', api_key)
            else:
                console.print("[red]API key required for online mode[/red]")
                return None
        config.set('mode', 'online')
        return 'online'
    elif choice == "3":
        config.set('mode', 'rag')
        return 'rag'
    elif choice == "4":
        # Check for API key
        api_key = config.get('api.api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            api_key = Prompt.ask("Enter API key for Online+RAG mode", password=True)
            if api_key:
                config.set('api.api_key', api_key)
            else:
                console.print("[red]API key required for Online+RAG mode[/red]")
                return None
        config.set('mode', 'online-rag')
        return 'online-rag'
    elif choice == "5":
        return 'compare'
    else:
        return None

def handle_cache_management(llm_engine):
    """Handle cache management commands."""
    console.print("\n[blue]Cache Management[/blue]")
    console.print("1. Show cache info")
    console.print("2. Clear cache")
    console.print("3. Back to main menu")
    
    choice = Prompt.ask("Select option (1-3)", choices=["1", "2", "3"], default="3")
    
    if choice == "1":
        cache_info = llm_engine.get_cache_info()
        if 'error' in cache_info:
            console.print(f"[red]Error: {cache_info['error']}[/red]")
        else:
            size_mb = cache_info['total_size'] / (1024 * 1024)
            console.print(f"[green]Cache Directory:[/green] {cache_info['cache_dir']}")
            console.print(f"[green]Cached Models:[/green] {cache_info['cached_models']}")
            console.print(f"[green]Total Size:[/green] {size_mb:.1f} MB")
            if cache_info['cache_files']:
                console.print(f"[green]Cache Files:[/green]")
                for file in cache_info['cache_files']:
                    console.print(f"  • {file}")
    
    elif choice == "2":
        if click.confirm("Are you sure you want to clear the cache?"):
            llm_engine.clear_model_cache()
        else:
            console.print("[yellow]Cache clearing cancelled[/yellow]")

def show_rag_impact_comparison(description, command_parser, shift_tab_completion, safety_system, config):
    """Show side-by-side comparison of command generation with and without RAG."""
    from rich.table import Table
    
    console.print(Panel(
        f"[bold cyan]RAG Impact Demonstration[/bold cyan]\n"
        f"Query: [yellow]{description}[/yellow]\n\n"
        f"Comparing generation with and without RAG documentation...",
        border_style="cyan"
    ))
    
    try:
        # Generate WITHOUT RAG (offline mode)
        console.print("\n[dim]Generating without RAG...[/dim]")
        config_no_rag = Config()
        config_no_rag.set('mode', 'offline')
        engine_no_rag = LLMEngine(config_no_rag)
        command_no_rag = engine_no_rag.generate_command(description)
        command_no_rag = command_parser.parse_command(command_no_rag)
        
        # Generate WITH RAG
        console.print("[dim]Generating with RAG...[/dim]")
        config_with_rag = Config()
        config_with_rag.set('mode', 'rag')
        engine_with_rag = LLMEngine(config_with_rag)
        command_with_rag = engine_with_rag.generate_command(description)
        command_with_rag = command_parser.parse_command(command_with_rag)
        
        # Create comparison table
        table = Table(title=f"\n🔍 RAG Impact Comparison", show_header=True, header_style="bold magenta", border_style="cyan")
        table.add_column("Mode", style="cyan", width=20)
        table.add_column("Generated Command", style="green", width=60)
        table.add_column("Impact", style="yellow", width=20)
        
        # Analyze differences
        impact = "No change"
        if command_no_rag != command_with_rag:
            if len(command_with_rag) > len(command_no_rag):
                impact = "✅ More detailed"
            else:
                impact = "✅ More concise"
            
            # Check for specific improvements
            if "-type" in command_with_rag and "-type" not in command_no_rag:
                impact = "✅ Added -type flag"
            elif any(flag in command_with_rag for flag in ["-i", "--interactive", "-v", "--verbose"]) and \
                 not any(flag in command_no_rag for flag in ["-i", "--interactive", "-v", "--verbose"]):
                impact = "✅ Added safety flags"
        
        table.add_row("🔌 Without RAG", command_no_rag, "Baseline")
        table.add_row("📚 With RAG", command_with_rag, impact)
        
        console.print(table)
        
        # Show detailed analysis
        if command_no_rag != command_with_rag:
            console.print("\n[bold yellow]📊 Analysis:[/bold yellow]")
            console.print(f"  • Without RAG: {len(command_no_rag)} characters, {len(command_no_rag.split())} words")
            console.print(f"  • With RAG: {len(command_with_rag)} characters, {len(command_with_rag.split())} words")
            
            # Highlight what RAG added
            rag_additions = set(command_with_rag.split()) - set(command_no_rag.split())
            if rag_additions:
                console.print(f"\n  [green]RAG added:[/green] {' '.join(rag_additions)}")
            
            rag_removals = set(command_no_rag.split()) - set(command_with_rag.split())
            if rag_removals:
                console.print(f"  [red]RAG removed:[/red] {' '.join(rag_removals)}")
        else:
            console.print("\n[yellow]ℹ️  RAG produced the same result as the base model for this query.[/yellow]")
            console.print("[dim]   Try more complex queries to see RAG's impact![/dim]")
        
        # Show what RAG provides
        console.print("\n[bold cyan]💡 What RAG Provides:[/bold cyan]")
        console.print("  ✅ More accurate command options")
        console.print("  ✅ Safety flags (-i, -type f, etc.)")
        console.print("  ✅ Complete command syntax")
        console.print("  ✅ Context from Linux documentation")
        console.print("  ✅ Best practices and modern alternatives")
        
    except Exception as e:
        console.print(f"[red]Error during comparison: {e}[/red]")
        import traceback
        traceback.print_exc()


def run_interactive_rag_demo_mode(command_parser, shift_tab_completion, safety_system, config):
    """Interactive mode that always shows RAG impact comparison."""
    console.print(Panel(
        "[bold cyan]🎯 Interactive RAG Demonstration Mode[/bold cyan]\n\n"
        "This mode shows side-by-side comparison of command generation\n"
        "WITH and WITHOUT RAG documentation for every query.\n\n"
        "[yellow]Perfect for demonstrating RAG's value![/yellow]\n\n"
        "Type your command descriptions to see the difference.\n"
        "Type 'exit' or 'q' to quit.",
        title="RAG Demo Mode",
        border_style="cyan"
    ))
    
    while True:
        try:
            description = Prompt.ask(f"\n[bold blue]Describe the command you want (or 'exit' to quit)")
            
            if description.lower() in ['exit', 'quit', 'q', 'bye', 'goodbye']:
                console.print("[yellow]👋 Thanks for exploring RAG![/yellow]")
                break
            
            if not description.strip():
                continue
            
            # Show comparison
            show_rag_impact_comparison(description, command_parser, shift_tab_completion, safety_system, config)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Thanks for exploring RAG![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def show_interactive_help():
    """Show help for interactive mode."""
    console.print(Panel(
        "📚 Interactive Mode Help\n\n"
        "🔧 COMMANDS:\n"
        "• Type any natural language description to generate a Linux command\n"
        "• 'mode' - Switch between offline/online/Offline+RAG/Online+RAG modes or compare all\n"
        "• 'compare' - Direct access to comparison mode\n"
        "• 'preload' - Preload both modes for instant switching\n"
        "• 'cache' - Manage model cache\n"
        "• 'shift+tab' - Get intelligent command completions\n"
        "• 'help' - Show this help message\n"
        "• 'exit', 'quit', 'q', 'bye', 'goodbye', 'stop', 'end' - Exit the program\n\n"
        "💡 EXAMPLES:\n"
        "• 'list all files in current directory'\n"
        "• 'find all .txt files recursively'\n"
        "• 'show running processes'\n"
        "• 'compress all log files'\n\n"
        "🛡️ SAFETY:\n"
        "• All commands are validated for safety\n"
        "• You'll be warned about potentially dangerous commands\n"
        "• Safer alternatives will be suggested when available\n\n"
        "💾 CACHE:\n"
        "• Models are cached for faster loading\n"
        "• Use 'cache' command to manage cache\n"
        "• Cache persists between app restarts",
        title="Interactive Help",
        border_style="green"
    ))

def run_comparison_mode(description, command_parser, shift_tab_completion, safety_system, config):
    """Run comparison mode - generate commands with all available modes."""
    from rich.table import Table
    
    console.print(f"\n[bold blue]Comparing all modes for:[/bold blue] {description}\n")
    
    results = {}
    modes_to_test = ['offline', 'rag', 'online', 'online-rag']
    
    # Mode display names
    mode_names = {
        'offline': '🔌 Offline',
        'rag': '📚 Offline+RAG', 
        'online': '🌐 Online',
        'online-rag': '🌐📚 Online+RAG'
    }
    
    # Test each mode
    for mode in modes_to_test:
        console.print(f"[dim]Testing {mode} mode...[/dim]")
        try:
            # Create engine for this mode
            config.set('mode', mode)
            
            # Skip online mode if no API key
            if mode == 'online':
                api_key = config.get('api.api_key') or os.getenv('OPENAI_API_KEY')
                if not api_key:
                    results[mode] = {'command': 'N/A', 'error': 'No API key configured'}
                    continue
            
            llm_engine = LLMEngine(config)
            command = llm_engine.generate_command(description)
            results[mode] = {'command': command, 'error': None}
            
        except Exception as e:
            results[mode] = {'command': 'N/A', 'error': str(e)}
    
    # Display results in a table
    table = Table(title="Mode Comparison Results")
    table.add_column("Mode", style="cyan", no_wrap=True)
    table.add_column("Generated Command", style="green")
    table.add_column("Status", style="yellow")
    
    for mode in modes_to_test:
        result = results.get(mode, {})
        command = result.get('command', 'N/A')
        error = result.get('error')
        
        status = "✅ Success" if not error else f"❌ {error}"
        
        table.add_row(
            mode_names.get(mode, mode.upper()),
            command,
            status
        )
    
    console.print(table)
    
    # Show recommendation
    console.print("\n[bold yellow]💡 Recommendation:[/bold yellow]")
    if results.get('online-rag', {}).get('command') and not results.get('online-rag', {}).get('error'):
        console.print("  Online+RAG mode provides the most accurate Linux commands.")
    elif results.get('rag', {}).get('command') and not results.get('rag', {}).get('error'):
        console.print("  Offline+RAG mode typically provides accurate Linux commands.")
    else:
        console.print("  Try different modes to see which works best for your query.")

def run_interactive_comparison_mode(command_parser, shift_tab_completion, safety_system, config):
    """Run interactive comparison mode."""
    console.print(Panel(
        "🔄 Comparison Mode - Interactive\n\n"
        "This mode will run your query through all available modes\n"
        "and show the results side-by-side for comparison.\n\n"
        "Type 'exit' or 'q' to quit.",
        title="Interactive Comparison Mode",
        border_style="blue"
    ))
    
    while True:
        try:
            description = Prompt.ask(f"\n[bold blue]Enter command description (or 'exit' to quit)")
            
            if description.lower() in ['exit', 'quit', 'q']:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                break
            
            run_comparison_mode(description, command_parser, shift_tab_completion, safety_system, config)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Goodbye! (Ctrl+C detected)[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")

if __name__ == '__main__':
    gen_cmd()

