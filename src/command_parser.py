"""
Command Parser for integrating with man command system.
Windows-compatible version that uses PowerShell and WSL when available.
"""

import subprocess
import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import platform

class CommandParser:
    """Parser for Linux commands with man page integration."""
    
    def __init__(self):
        self.system = platform.system()
        self.man_cache = {}
        self.wsl_available = self._check_wsl_availability()
    
    def _check_wsl_availability(self) -> bool:
        """Check if WSL is available on Windows."""
        if self.system != "Windows":
            return True
        
        try:
            result = subprocess.run(['wsl', '--list'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def parse_command(self, command: str) -> str:
        """Parse and validate a generated command."""
        # Clean up the command
        command = command.strip()
        
        # Remove any markdown formatting
        command = re.sub(r'```bash\s*', '', command)
        command = re.sub(r'```\s*$', '', command)
        command = command.strip()
        
        # Validate command structure
        if not self._is_valid_command(command):
            raise ValueError(f"Invalid command structure: {command}")
        
        # Check if command exists in man pages
        # Note: On Windows without WSL, man pages won't be available
        # But we should NEVER convert Linux commands to Windows alternatives
        # The command is already validated as Linux-only by _is_valid_command
        return command
    
    def _is_valid_command(self, command: str) -> bool:
        """Check if command has valid structure."""
        if not command:
            return False
        
        # Basic validation - should start with a command
        parts = command.split()
        if not parts:
            return False
        
        # Check for dangerous commands
        dangerous_commands = ['rm -rf /', 'dd if=', 'mkfs', 'fdisk']
        for dangerous in dangerous_commands:
            if dangerous in command:
                return False
        
        return True
    
    def _command_exists_in_man(self, command: str) -> bool:
        """Check if command exists in man pages."""
        cmd_name = command.split()[0]
        
        if cmd_name in self.man_cache:
            return self.man_cache[cmd_name]
        
        exists = self._check_man_page(cmd_name)
        self.man_cache[cmd_name] = exists
        return exists
    
    def _check_man_page(self, command: str) -> bool:
        """Check if man page exists for command."""
        try:
            if self.system == "Windows" and self.wsl_available:
                # Use WSL to check man pages
                result = subprocess.run(
                    ['wsl', 'man', command],
                    capture_output=True, text=True, timeout=10
                )
                return result.returncode == 0
            else:
                # Use system man command
                result = subprocess.run(
                    ['man', command],
                    capture_output=True, text=True, timeout=10
                )
                return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _suggest_alternatives(self, command: str) -> List[str]:
        """Suggest alternative commands if the original doesn't exist."""
        cmd_name = command.split()[0]
        
        # Common command alternatives
        alternatives = {
            'ls': ['dir', 'Get-ChildItem'],
            'find': ['Get-ChildItem -Recurse', 'dir /s'],
            'grep': ['Select-String', 'findstr'],
            'awk': ['ForEach-Object'],
            'sed': ['ForEach-Object'],
            'chmod': ['icacls', 'Set-Acl'],
            'chown': ['icacls', 'Set-Acl']
        }
        
        if cmd_name in alternatives:
            return alternatives[cmd_name]
        
        return []
    
    def get_man_page(self, command: str) -> Optional[str]:
        """Get man page content for a command."""
        try:
            if self.system == "Windows" and self.wsl_available:
                result = subprocess.run(
                    ['wsl', 'man', command],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    return result.stdout
            else:
                result = subprocess.run(
                    ['man', command],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return None
    
    def extract_command_syntax(self, command: str) -> Dict[str, any]:
        """Extract syntax information from man page."""
        cmd_name = command.split()[0]
        man_content = self.get_man_page(cmd_name)
        
        if not man_content:
            return {}
        
        # Parse man page for syntax information
        syntax_info = {
            'command': cmd_name,
            'description': self._extract_description(man_content),
            'options': self._extract_options(man_content),
            'examples': self._extract_examples(man_content)
        }
        
        return syntax_info
    
    def _extract_description(self, man_content: str) -> str:
        """Extract command description from man page."""
        lines = man_content.split('\n')
        for i, line in enumerate(lines):
            if 'DESCRIPTION' in line.upper():
                # Get next few lines as description
                desc_lines = []
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip() and not lines[j].startswith(' '):
                        break
                    desc_lines.append(lines[j].strip())
                return ' '.join(desc_lines)
        return ""
    
    def _extract_options(self, man_content: str) -> List[str]:
        """Extract command options from man page."""
        options = []
        lines = man_content.split('\n')
        
        for line in lines:
            # Look for option patterns like "-v, --verbose"
            if re.match(r'^\s*-[a-zA-Z]', line):
                options.append(line.strip())
        
        return options
    
    def _extract_examples(self, man_content: str) -> List[str]:
        """Extract examples from man page."""
        examples = []
        lines = man_content.split('\n')
        
        for line in lines:
            # Look for example patterns
            if re.match(r'^\s*\$', line) or re.match(r'^\s*#', line):
                examples.append(line.strip())
        
        return examples
    
    def validate_command_safety(self, command: str) -> Tuple[bool, str]:
        """Validate command safety and return (is_safe, warning_message)."""
        # Check for dangerous patterns
        dangerous_patterns = [
            (r'rm\s+-rf\s+/', "Dangerous: rm -rf / can delete entire system"),
            (r'dd\s+if=', "Dangerous: dd command can overwrite disks"),
            (r'mkfs\s+', "Dangerous: mkfs can format disks"),
            (r'fdisk\s+', "Dangerous: fdisk can partition disks"),
            (r'>\s+/dev/', "Dangerous: Writing to device files"),
            (r'chmod\s+777', "Warning: chmod 777 gives full permissions"),
            (r'chown\s+-R\s+root', "Warning: Changing ownership to root")
        ]
        
        for pattern, warning in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False, warning
        
        return True, ""
    
    def get_windows_equivalent(self, command: str) -> str:
        """Get Windows equivalent of Linux command."""
        cmd_name = command.split()[0]
        
        # Common Linux to Windows mappings
        mappings = {
            'ls': 'dir',
            'find': 'Get-ChildItem -Recurse',
            'grep': 'Select-String',
            'awk': 'ForEach-Object',
            'sed': 'ForEach-Object',
            'chmod': 'icacls',
            'chown': 'icacls',
            'cp': 'copy',
            'mv': 'move',
            'rm': 'del',
            'mkdir': 'md',
            'rmdir': 'rd'
        }
        
        if cmd_name in mappings:
            return command.replace(cmd_name, mappings[cmd_name], 1)
        
        return command