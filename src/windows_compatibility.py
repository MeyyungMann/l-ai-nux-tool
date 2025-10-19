"""
Windows-specific enhancements for the Linux AI Command Generator.
"""

import os
import subprocess
import platform
from typing import List, Dict, Optional

class WindowsCompatibility:
    """Windows compatibility layer for Linux commands."""
    
    def __init__(self):
        self.system = platform.system()
        self.is_docker = self._check_docker()
        self.wsl_available = self._check_wsl()
        self.powershell_available = self._check_powershell()
    
    def _check_docker(self) -> bool:
        """Check if running inside Docker container."""
        return os.path.exists('/.dockerenv')
    
    def _check_wsl(self) -> bool:
        """Check if WSL is available."""
        if self.system != "Windows":
            return True
        
        try:
            result = subprocess.run(['wsl', '--list'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _check_powershell(self) -> bool:
        """Check if PowerShell is available."""
        try:
            result = subprocess.run(['powershell', '-Command', 'Get-Host'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def convert_linux_to_windows(self, command: str) -> str:
        """Convert Linux command to Windows equivalent."""
        if self.system != "Windows":
            return command
        
        # Common Linux to Windows command mappings
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
            'rmdir': 'rd',
            'cat': 'type',
            'head': 'Get-Content -TotalCount',
            'tail': 'Get-Content -Tail',
            'wc': 'Measure-Object',
            'sort': 'Sort-Object',
            'uniq': 'Get-Unique',
            'which': 'where',
            'pwd': 'Get-Location'
        }
        
        # Convert command
        parts = command.split()
        if parts:
            cmd_name = parts[0]
            if cmd_name in mappings:
                parts[0] = mappings[cmd_name]
                command = ' '.join(parts)
        
        return command
    
    def execute_command(self, command: str) -> tuple[str, str, int]:
        """Execute command with Windows compatibility."""
        # If running in Docker, execute commands directly in Linux environment
        if self.is_docker:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.stdout, result.stderr, result.returncode
        
        if self.system != "Windows":
            # Use standard subprocess for Unix systems
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.stdout, result.stderr, result.returncode
        
        # Windows-specific execution
        if self.wsl_available and self._is_linux_command(command):
            # Execute in WSL
            wsl_command = f"wsl {command}"
            result = subprocess.run(wsl_command, shell=True, capture_output=True, text=True)
            return result.stdout, result.stderr, result.returncode
        else:
            # Convert to Windows command
            windows_command = self.convert_linux_to_windows(command)
            result = subprocess.run(windows_command, shell=True, capture_output=True, text=True)
            return result.stdout, result.stderr, result.returncode
    
    def _is_linux_command(self, command: str) -> bool:
        """Check if command is Linux-specific."""
        linux_commands = [
            'ls', 'find', 'grep', 'awk', 'sed', 'chmod', 'chown',
            'cp', 'mv', 'rm', 'mkdir', 'rmdir', 'cat', 'head',
            'tail', 'wc', 'sort', 'uniq', 'which', 'pwd'
        ]
        
        cmd_name = command.split()[0]
        return cmd_name in linux_commands
    
    def get_system_info(self) -> Dict[str, any]:
        """Get system information for compatibility."""
        return {
            'system': self.system,
            'is_docker': self.is_docker,
            'wsl_available': self.wsl_available,
            'powershell_available': self.powershell_available,
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0]
        }
    
    def suggest_alternatives(self, command: str) -> List[str]:
        """Suggest Windows alternatives for Linux commands."""
        if self.system != "Windows":
            return []
        
        alternatives = []
        cmd_name = command.split()[0]
        
        # Command-specific alternatives
        if cmd_name == 'find':
            alternatives.extend([
                'Get-ChildItem -Recurse -Name "*.tmp"',
                'dir /s *.tmp',
                'forfiles /s /m *.tmp'
            ])
        elif cmd_name == 'grep':
            alternatives.extend([
                'Select-String -Pattern "pattern"',
                'findstr "pattern"',
                'findstr /r "pattern"'
            ])
        elif cmd_name == 'awk':
            alternatives.extend([
                'ForEach-Object { $_.Property }',
                'Select-Object -Property Property',
                'Where-Object { $_.Property -eq "value" }'
            ])
        
        return alternatives






