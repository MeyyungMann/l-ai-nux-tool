"""
Shift+Tab Intelligent Completion System
Provides intelligent transformation and completion of partial commands
"""

import re
import subprocess
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import platform

class ShiftTabCompletion:
    """Intelligent command completion and transformation system."""
    
    def __init__(self, llm_engine, command_parser):
        self.llm_engine = llm_engine
        self.command_parser = command_parser
        self.context_buffer = []
        self.completion_history = []
        
        # Common command patterns for completion
        self.command_patterns = {
            'file_operations': {
                'ls': ['ls -la', 'ls -lt', 'ls -lh', 'ls -R'],
                'find': ['find . -name', 'find . -type f', 'find . -size +100M'],
                'cp': ['cp -r', 'cp -v', 'cp -p'],
                'mv': ['mv -v', 'mv -i'],
                'rm': ['rm -rf', 'rm -i', 'rm -v']
            },
            'text_processing': {
                'grep': ['grep -r', 'grep -i', 'grep -n', 'grep -v'],
                'awk': ['awk \'{print $1}\'', 'awk -F: \'{print $1}\''],
                'sed': ['sed -i', 'sed -n', 'sed \'s/old/new/g\''],
                'cut': ['cut -d: -f1', 'cut -c1-10'],
                'sort': ['sort -n', 'sort -r', 'sort -u']
            },
            'system_info': {
                'ps': ['ps aux', 'ps -ef', 'ps aux --sort=-%cpu'],
                'top': ['top', 'htop'],
                'df': ['df -h', 'df -T'],
                'du': ['du -h', 'du -sh', 'du -h --max-depth=1']
            }
        }
    
    def complete_command(self, partial_input: str) -> List[str]:
        """Generate completion suggestions for partial input."""
        
        # Clean input
        partial_input = partial_input.strip()
        
        if not partial_input:
            return self._get_common_commands()
        
        # Analyze the partial input
        analysis = self._analyze_input(partial_input)
        
        # Generate completions based on analysis
        completions = []
        
        # 1. Pattern-based completions
        pattern_completions = self._get_pattern_completions(partial_input)
        completions.extend(pattern_completions)
        
        # 2. LLM-based completions
        llm_completions = self._get_llm_completions(partial_input)
        completions.extend(llm_completions)
        
        # 3. Context-based completions
        context_completions = self._get_context_completions(partial_input)
        completions.extend(context_completions)
        
        # 4. History-based completions
        history_completions = self._get_history_completions(partial_input)
        completions.extend(history_completions)
        
        # Remove duplicates and validate
        unique_completions = list(dict.fromkeys(completions))
        validated_completions = [comp for comp in unique_completions if self._validate_completion(comp)]
        
        # Sort by relevance
        scored_completions = self._score_completions(partial_input, validated_completions)
        
        return scored_completions[:5]  # Return top 5 completions
    
    def _analyze_input(self, input_text: str) -> Dict[str, Any]:
        """Analyze the input to understand what the user is trying to do."""
        
        analysis = {
            'command_type': 'unknown',
            'has_command': False,
            'has_arguments': False,
            'has_options': False,
            'has_paths': False,
            'is_partial': False,
            'suggested_command': None
        }
        
        # Check if it starts with a command
        words = input_text.split()
        if words:
            first_word = words[0].lower()
            
            # Check against known commands
            for category, commands in self.command_patterns.items():
                if first_word in commands:
                    analysis['command_type'] = category
                    analysis['has_command'] = True
                    analysis['suggested_command'] = first_word
                    break
            
            # Check if it's a partial command
            if len(words) == 1 and not analysis['has_command']:
                analysis['is_partial'] = True
            
            # Check for arguments and options
            if len(words) > 1:
                analysis['has_arguments'] = True
                
                # Check for options (starting with -)
                for word in words[1:]:
                    if word.startswith('-'):
                        analysis['has_options'] = True
                    elif '/' in word or '.' in word:
                        analysis['has_paths'] = True
        
        return analysis
    
    def _get_pattern_completions(self, partial_input: str) -> List[str]:
        """Get completions based on command patterns."""
        
        completions = []
        words = partial_input.split()
        
        if not words:
            return completions
        
        first_word = words[0].lower()
        
        # Find matching patterns
        for category, commands in self.command_patterns.items():
            if first_word in commands:
                # Get common completions for this command
                command_completions = commands[first_word]
                
                # If we have more words, try to complete based on context
                if len(words) > 1:
                    context = ' '.join(words[1:])
                    for completion in command_completions:
                        if context in completion:
                            completions.append(completion)
                else:
                    completions.extend(command_completions)
        
        return completions
    
    def _get_llm_completions(self, partial_input: str) -> List[str]:
        """Get completions using the LLM engine."""
        
        try:
            # Create a completion prompt
            prompt = f"""Complete this Linux command:

Partial command: {partial_input}

Provide 3 possible completions that are:
1. Safe and follow best practices
2. Commonly used
3. Complete the intended task

Completions:"""
            
            # Generate completions
            response = self.llm_engine.generate_command(prompt)
            
            # Parse the response
            completions = self._parse_llm_response(response)
            
            return completions
            
        except Exception as e:
            print(f"Error getting LLM completions: {e}")
            return []
    
    def _get_context_completions(self, partial_input: str) -> List[str]:
        """Get completions based on current context."""
        
        completions = []
        
        # Check current directory context
        try:
            current_dir = Path.cwd()
            
            # If input contains current directory patterns
            if '.' in partial_input or '..' in partial_input:
                completions.extend([
                    f"ls {current_dir}",
                    f"find {current_dir} -name",
                    f"du -h {current_dir}"
                ])
            
            # Check for common file patterns
            if any(pattern in partial_input for pattern in ['*.txt', '*.log', '*.py', '*.js']):
                completions.extend([
                    f"find . -name '*.txt'",
                    f"grep -r 'pattern' .",
                    f"ls -la *.txt"
                ])
                
        except Exception as e:
            print(f"Error getting context completions: {e}")
        
        return completions
    
    def _get_history_completions(self, partial_input: str) -> List[str]:
        """Get completions based on command history."""
        
        completions = []
        
        # Simple history matching
        for history_item in self.completion_history[-10:]:  # Last 10 commands
            if partial_input.lower() in history_item.lower():
                completions.append(history_item)
        
        return completions
    
    def _get_common_commands(self) -> List[str]:
        """Get common Linux commands when no input is provided."""
        
        return [
            'ls -la',
            'find . -name',
            'grep -r',
            'ps aux',
            'df -h',
            'du -h',
            'top',
            'htop',
            'cat',
            'head',
            'tail'
        ]
    
    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse LLM response to extract completions."""
        
        completions = []
        
        # Split by lines and look for commands
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for lines that look like commands
            if self._looks_like_command(line):
                # Clean up the command
                command = self._clean_command(line)
                if command:
                    completions.append(command)
        
        return completions
    
    def _looks_like_command(self, text: str) -> bool:
        """Check if text looks like a Linux command."""
        
        if not text or len(text) < 2:
            return False
        
        # Should start with a command name
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*', text):
            return False
        
        # Should not be too long
        if len(text) > 200:
            return False
        
        return True
    
    def _clean_command(self, command: str) -> str:
        """Clean up a command string."""
        
        # Remove common prefixes
        prefixes_to_remove = ['$', '#', '>', 'Command:', 'Completion:']
        
        for prefix in prefixes_to_remove:
            if command.startswith(prefix):
                command = command[len(prefix):].strip()
        
        # Remove quotes
        command = command.strip('"\'')
        
        return command.strip()
    
    def _validate_completion(self, completion: str) -> bool:
        """Validate that a completion is safe and reasonable."""
        
        if not completion or len(completion) < 2:
            return False
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'rm\s+-rf\s+/',
            r'dd\s+if=',
            r'mkfs\s+',
            r'fdisk\s+',
            r'>\s+/dev/'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, completion, re.IGNORECASE):
                return False
        
        return True
    
    def _score_completions(self, partial_input: str, completions: List[str]) -> List[str]:
        """Score and sort completions by relevance."""
        
        scored = []
        
        for completion in completions:
            score = 0
            
            # Exact match gets highest score
            if completion.lower().startswith(partial_input.lower()):
                score += 100
            
            # Partial match gets medium score
            elif partial_input.lower() in completion.lower():
                score += 50
            
            # Length similarity
            length_diff = abs(len(completion) - len(partial_input))
            score += max(0, 20 - length_diff)
            
            # Common commands get bonus
            common_commands = ['ls', 'find', 'grep', 'ps', 'df', 'du']
            if any(cmd in completion for cmd in common_commands):
                score += 10
            
            scored.append((score, completion))
        
        # Sort by score (descending) and return commands
        scored.sort(key=lambda x: x[0], reverse=True)
        return [completion for score, completion in scored]
    
    def add_to_history(self, command: str):
        """Add a command to the completion history."""
        
        if command and command not in self.completion_history:
            self.completion_history.append(command)
            
            # Keep only last 50 commands
            if len(self.completion_history) > 50:
                self.completion_history = self.completion_history[-50:]
    
    def get_completion_stats(self) -> Dict[str, Any]:
        """Get statistics about completion usage."""
        
        return {
            'total_completions': len(self.completion_history),
            'unique_commands': len(set(self.completion_history)),
            'most_used_command': max(set(self.completion_history), key=self.completion_history.count) if self.completion_history else None,
            'completion_categories': self._get_category_stats()
        }
    
    def _get_category_stats(self) -> Dict[str, int]:
        """Get statistics by command category."""
        
        categories = {}
        
        for command in self.completion_history:
            first_word = command.split()[0].lower()
            
            for category, commands in self.command_patterns.items():
                if first_word in commands:
                    categories[category] = categories.get(category, 0) + 1
                    break
        else:
                categories['other'] = categories.get('other', 0) + 1

        return categories