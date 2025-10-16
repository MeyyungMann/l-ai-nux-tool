#!/usr/bin/env python3
"""
Fetch man pages from Linux system and add to RAG documentation
"""

import subprocess
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

class ManPageCollector:
    """Collect man pages from Linux system for RAG."""
    
    def __init__(self):
        self.doc_cache_dir = Path("doc_cache")
        self.doc_cache_dir.mkdir(exist_ok=True)
        self.docs_file = self.doc_cache_dir / "linux_docs.json"
    
    def get_available_commands(self) -> List[str]:
        """Get list of all available commands on the system."""
        try:
            # Get all executables in PATH
            result = subprocess.run(
                ['bash', '-c', 'compgen -c | sort -u'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                commands = result.stdout.strip().split('\n')
                # Filter common commands (avoid aliases and shell builtins)
                common_commands = [cmd for cmd in commands if cmd and len(cmd) > 1 and not cmd.startswith('.')]
                return common_commands[:500]  # Limit to first 500
            
        except Exception as e:
            print(f"Error getting commands: {e}")
        
        return []
    
    def get_man_page(self, command: str) -> Optional[Dict]:
        """Get man page content for a command."""
        try:
            # Get man page
            result = subprocess.run(
                ['man', command],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return None
            
            man_text = result.stdout
            
            # Parse man page
            parsed = self._parse_man_page(command, man_text)
            return parsed
            
        except Exception as e:
            return None
    
    def _parse_man_page(self, command: str, man_text: str) -> Dict:
        """Parse man page text into structured format."""
        
        # Extract NAME section (description)
        name_match = re.search(r'NAME\s+(.*?)(?=\n\S|\n\n)', man_text, re.DOTALL)
        description = ""
        if name_match:
            desc_text = name_match.group(1).strip()
            # Clean up the description
            desc_text = re.sub(r'\s+', ' ', desc_text)
            description = desc_text[:200]  # Limit length
        
        # Extract SYNOPSIS section (usage examples)
        synopsis_match = re.search(r'SYNOPSIS\s+(.*?)(?=\n[A-Z]|\n\n\S)', man_text, re.DOTALL)
        examples = []
        if synopsis_match:
            synopsis = synopsis_match.group(1).strip()
            # Extract command patterns
            lines = synopsis.split('\n')
            for line in lines:
                line = line.strip()
                if line and command in line:
                    # Clean up the line
                    clean_line = re.sub(r'\s+', ' ', line)
                    examples.append(clean_line[:100])
        
        # Extract OPTIONS section
        options = {}
        options_match = re.search(r'OPTIONS\s+(.*?)(?=\n[A-Z]{2,}|\Z)', man_text, re.DOTALL)
        if options_match:
            options_text = options_match.group(1)
            # Parse options (simple parsing)
            option_lines = re.findall(r'(-[a-zA-Z0-9-]+)\s+(.{0,100})', options_text)
            for opt, desc in option_lines[:10]:  # Limit to 10 options
                desc = re.sub(r'\s+', ' ', desc.strip())
                options[opt.strip()] = desc[:80]
        
        # Determine category
        category = self._categorize_command(command, description)
        
        return {
            "command": command,
            "description": description or f"{command} - Linux command",
            "examples": examples[:5] if examples else [command],
            "options": options,
            "category": category,
            "man_page": "fetched_from_system"
        }
    
    def _categorize_command(self, command: str, description: str) -> str:
        """Categorize command based on name and description."""
        
        categories = {
            'file': ['ls', 'cp', 'mv', 'rm', 'mkdir', 'touch', 'cat', 'ln', 'chmod', 'chown'],
            'search': ['find', 'locate', 'which', 'whereis', 'grep', 'ack', 'ag'],
            'text': ['sed', 'awk', 'cut', 'sort', 'uniq', 'tr', 'wc', 'head', 'tail'],
            'archive': ['tar', 'zip', 'unzip', 'gzip', 'bzip2', '7z', 'rar'],
            'network': ['curl', 'wget', 'ssh', 'scp', 'ping', 'netstat', 'ip', 'ifconfig'],
            'process': ['ps', 'top', 'htop', 'kill', 'killall', 'pkill', 'nice', 'renice'],
            'system': ['systemctl', 'service', 'uname', 'uptime', 'dmesg', 'journalctl'],
            'disk': ['df', 'du', 'mount', 'umount', 'fdisk', 'lsblk'],
            'user': ['su', 'sudo', 'useradd', 'usermod', 'passwd', 'whoami', 'id'],
            'package': ['apt', 'yum', 'dnf', 'pacman', 'snap', 'pip', 'npm'],
        }
        
        for category, commands in categories.items():
            if command in commands:
                return category
        
        # Check description for keywords
        desc_lower = description.lower()
        if any(word in desc_lower for word in ['file', 'directory', 'folder']):
            return 'file'
        elif any(word in desc_lower for word in ['search', 'find', 'locate']):
            return 'search'
        elif any(word in desc_lower for word in ['network', 'internet', 'download']):
            return 'network'
        elif any(word in desc_lower for word in ['process', 'task', 'running']):
            return 'process'
        
        return 'general'
    
    def fetch_and_add_commands(self, commands: List[str]) -> int:
        """Fetch man pages for commands and add to documentation."""
        
        # Load existing docs
        if self.docs_file.exists():
            with open(self.docs_file, 'r') as f:
                existing_docs = json.load(f)
        else:
            existing_docs = []
        
        # Get existing command names
        existing_commands = {doc['command'] for doc in existing_docs}
        
        added_count = 0
        failed_count = 0
        
        print(f"📚 Fetching man pages for {len(commands)} commands...")
        print("=" * 60)
        
        for i, command in enumerate(commands, 1):
            # Skip if already exists
            if command in existing_commands:
                continue
            
            # Show progress
            if i % 10 == 0:
                print(f"Progress: {i}/{len(commands)} ({added_count} added, {failed_count} failed)")
            
            # Fetch man page
            doc = self.get_man_page(command)
            
            if doc:
                existing_docs.append(doc)
                existing_commands.add(command)
                added_count += 1
                print(f"✅ {command}: {doc['description'][:50]}...")
            else:
                failed_count += 1
        
        # Save updated docs
        with open(self.docs_file, 'w') as f:
            json.dump(existing_docs, f, indent=2)
        
        print("\n" + "=" * 60)
        print(f"✅ COMPLETE!")
        print(f"   Added: {added_count} new commands")
        print(f"   Failed: {failed_count} commands")
        print(f"   Total docs: {len(existing_docs)}")
        print(f"   Saved to: {self.docs_file}")
        
        return added_count

def main():
    """Main function to fetch and add man pages."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch man pages for RAG documentation")
    parser.add_argument('--commands', nargs='+', help='Specific commands to fetch')
    parser.add_argument('--auto', action='store_true', help='Auto-fetch common commands')
    parser.add_argument('--limit', type=int, default=50, help='Max commands to fetch')
    
    args = parser.parse_args()
    
    collector = ManPageCollector()
    
    if args.commands:
        # Fetch specific commands
        print(f"Fetching man pages for specific commands: {args.commands}")
        collector.fetch_and_add_commands(args.commands)
    
    elif args.auto:
        # Auto-fetch common commands
        print("Auto-fetching common Linux commands...")
        
        # Prioritized list of common commands
        common_commands = [
            # File operations
            'ls', 'cp', 'mv', 'rm', 'mkdir', 'rmdir', 'touch', 'cat', 'less', 'more',
            'head', 'tail', 'ln', 'chmod', 'chown', 'chgrp',
            
            # Search and find
            'find', 'locate', 'which', 'whereis', 'grep', 'egrep', 'fgrep',
            
            # Text processing
            'sed', 'awk', 'cut', 'sort', 'uniq', 'tr', 'wc', 'diff', 'patch',
            
            # Archives
            'tar', 'zip', 'unzip', 'gzip', 'gunzip', 'bzip2', 'xz',
            
            # Network
            'curl', 'wget', 'ssh', 'scp', 'rsync', 'ping', 'netstat', 'ss', 'ip',
            
            # Process management
            'ps', 'top', 'htop', 'kill', 'killall', 'pkill', 'pgrep', 'nice',
            
            # System
            'systemctl', 'service', 'uname', 'uptime', 'dmesg', 'journalctl',
            'df', 'du', 'mount', 'umount', 'lsblk', 'fdisk',
            
            # User management
            'su', 'sudo', 'useradd', 'usermod', 'userdel', 'passwd', 'whoami', 'id', 'groups',
            
            # Package management
            'apt', 'apt-get', 'dpkg', 'yum', 'dnf', 'pacman', 'snap',
        ]
        
        # Limit commands
        commands_to_fetch = common_commands[:args.limit]
        collector.fetch_and_add_commands(commands_to_fetch)
    
    else:
        print("Usage:")
        print("  Fetch specific commands:")
        print("    python src/utils/fetch_man_pages.py --commands ls find grep")
        print()
        print("  Auto-fetch common commands:")
        print("    python src/utils/fetch_man_pages.py --auto --limit 50")

if __name__ == "__main__":
    main()

