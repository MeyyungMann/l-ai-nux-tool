"""
Enhanced Safety System for Linux Command Generation
Provides comprehensive safety checks and warnings
"""

import re
import subprocess
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import platform

class EnhancedSafetySystem:
    """Enhanced safety system for command validation and warnings."""
    
    def __init__(self):
        # Dangerous patterns with severity levels
        self.dangerous_patterns = {
            'critical': [
                (r'rm\s+-rf\s+/', "CRITICAL: rm -rf / can delete entire system"),
                (r'dd\s+if=', "CRITICAL: dd command can overwrite disks"),
                (r'mkfs\s+', "CRITICAL: mkfs can format disks"),
                (r'fdisk\s+', "CRITICAL: fdisk can partition disks"),
                (r'>\s+/dev/', "CRITICAL: Writing to device files"),
                (r'echo\s+.*>\s*/dev/', "CRITICAL: Writing to device files"),
            ],
            'high': [
                (r'rm\s+-rf\s+[^/]', "HIGH: rm -rf can delete large directory trees"),
                (r'chmod\s+777', "HIGH: chmod 777 gives full permissions to everyone"),
                (r'chown\s+-R\s+root', "HIGH: Changing ownership to root"),
                (r'sudo\s+rm\s+-rf', "HIGH: sudo rm -rf with elevated privileges"),
                (r'mv\s+.*\s+/', "HIGH: Moving files to root directory"),
                (r'cp\s+.*\s+/', "HIGH: Copying files to root directory"),
            ],
            'medium': [
                (r'rm\s+-rf', "MEDIUM: rm -rf can delete directories"),
                (r'chmod\s+[0-9]{3}', "MEDIUM: Changing file permissions"),
                (r'chown\s+', "MEDIUM: Changing file ownership"),
                (r'sudo\s+', "MEDIUM: Using sudo with elevated privileges"),
                (r'mount\s+', "MEDIUM: Mounting filesystems"),
                (r'umount\s+', "MEDIUM: Unmounting filesystems"),
            ],
            'low': [
                (r'rm\s+', "LOW: File deletion"),
                (r'mv\s+', "LOW: Moving files"),
                (r'cp\s+', "LOW: Copying files"),
                (r'touch\s+', "LOW: Creating files"),
                (r'mkdir\s+', "LOW: Creating directories"),
            ]
        }
        
        # Safe commands that are generally okay
        self.safe_commands = [
            'ls', 'pwd', 'echo', 'cat', 'head', 'tail', 'grep', 'find',
            'ps', 'top', 'htop', 'df', 'du', 'free', 'uptime', 'who',
            'ping', 'curl', 'wget', 'netstat', 'ss', 'traceroute',
            'tar', 'zip', 'unzip', 'gzip', 'gunzip', 'bzip2', 'bunzip2',
            'sort', 'uniq', 'cut', 'awk', 'sed', 'less', 'more'
        ]
        
        # Commands that require special attention
        self.special_attention_commands = [
            'rm', 'mv', 'cp', 'chmod', 'chown', 'chgrp', 'sudo', 'su',
            'mount', 'umount', 'mkfs', 'fdisk', 'dd', 'parted'
        ]
    
    def validate_command_safety(self, command: str) -> Tuple[bool, str, Dict[str, any]]:
        """Validate command safety and return detailed analysis."""
        
        analysis = {
            'is_safe': True,
            'severity': 'none',
            'warnings': [],
            'suggestions': [],
            'risk_factors': [],
            'confidence': 100
        }
        
        # Clean the command
        clean_command = self._clean_command(command)
        
        if not clean_command:
            return False, "Empty or invalid command", analysis
        
        # Check for dangerous patterns
        for severity, patterns in self.dangerous_patterns.items():
            for pattern, message in patterns:
                if re.search(pattern, clean_command, re.IGNORECASE):
                    analysis['is_safe'] = False
                    analysis['severity'] = severity
                    analysis['warnings'].append(message)
                    analysis['risk_factors'].append(pattern)
        
        # Check command-specific safety
        command_safety = self._check_command_specific_safety(clean_command)
        if not command_safety['is_safe']:
            analysis['is_safe'] = False
            analysis['severity'] = max(analysis['severity'], command_safety['severity'])
            analysis['warnings'].extend(command_safety['warnings'])
            analysis['risk_factors'].extend(command_safety['risk_factors'])
        
        # Generate suggestions
        analysis['suggestions'] = self._generate_safety_suggestions(clean_command, analysis)
        
        # Calculate confidence
        analysis['confidence'] = self._calculate_confidence(clean_command, analysis)
        
        # Determine overall safety
        is_safe = analysis['is_safe'] and analysis['severity'] in ['none', 'low']
        warning_message = '; '.join(analysis['warnings']) if analysis['warnings'] else ''
        
        return is_safe, warning_message, analysis
    
    def _clean_command(self, command: str) -> str:
        """Clean and normalize a command string."""
        
        # Remove common prefixes
        prefixes_to_remove = ['$', '#', '>', 'Command:', 'Generated:']
        
        for prefix in prefixes_to_remove:
            if command.startswith(prefix):
                command = command[len(prefix):].strip()
        
        # Remove quotes
        command = command.strip('"\'')
        
        # Remove extra whitespace
        command = ' '.join(command.split())
        
        return command.strip()
    
    def _check_command_specific_safety(self, command: str) -> Dict[str, any]:
        """Check safety for specific command types."""
        
        safety = {
            'is_safe': True,
            'severity': 'none',
            'warnings': [],
            'risk_factors': []
        }
        
        words = command.split()
        if not words:
            return safety
        
        command_name = words[0].lower()
        
        # Check for dangerous command combinations
        if command_name == 'rm':
            safety = self._check_rm_safety(command, words)
        elif command_name == 'chmod':
            safety = self._check_chmod_safety(command, words)
        elif command_name == 'chown':
            safety = self._check_chown_safety(command, words)
        elif command_name == 'sudo':
            safety = self._check_sudo_safety(command, words)
        elif command_name == 'dd':
            safety = self._check_dd_safety(command, words)
        elif command_name == 'mount':
            safety = self._check_mount_safety(command, words)
        
        return safety
    
    def _check_rm_safety(self, command: str, words: List[str]) -> Dict[str, any]:
        """Check safety of rm commands."""
        
        safety = {
            'is_safe': True,
            'severity': 'none',
            'warnings': [],
            'risk_factors': []
        }
        
        # Check for recursive deletion
        if '-r' in words or '-rf' in words or '--recursive' in words:
            safety['severity'] = 'high'
            safety['warnings'].append("Recursive deletion can remove entire directory trees")
            safety['risk_factors'].append('recursive_deletion')
        
        # Check for force deletion
        if '-f' in words or '--force' in words:
            safety['severity'] = 'high'
            safety['warnings'].append("Force deletion skips confirmation prompts")
            safety['risk_factors'].append('force_deletion')
        
        # Check for dangerous paths
        for word in words[1:]:
            if word.startswith('-'):
                continue
            
            if word == '/' or word.startswith('/'):
                safety['is_safe'] = False
                safety['severity'] = 'critical'
                safety['warnings'].append(f"Dangerous path: {word}")
                safety['risk_factors'].append('dangerous_path')
        
        return safety
    
    def _check_chmod_safety(self, command: str, words: List[str]) -> Dict[str, any]:
        """Check safety of chmod commands."""
        
        safety = {
            'is_safe': True,
            'severity': 'none',
            'warnings': [],
            'risk_factors': []
        }
        
        # Check for dangerous permissions
        for word in words[1:]:
            if word.startswith('-'):
                continue
            
            if word in ['777', '666', '000']:
                safety['severity'] = 'high'
                safety['warnings'].append(f"Dangerous permissions: {word}")
                safety['risk_factors'].append('dangerous_permissions')
        
        return safety
    
    def _check_chown_safety(self, command: str, words: List[str]) -> Dict[str, any]:
        """Check safety of chown commands."""
        
        safety = {
            'is_safe': True,
            'severity': 'none',
            'warnings': [],
            'risk_factors': []
        }
        
        # Check for recursive ownership change
        if '-R' in words or '--recursive' in words:
            safety['severity'] = 'high'
            safety['warnings'].append("Recursive ownership change affects entire directory trees")
            safety['risk_factors'].append('recursive_ownership')
        
        return safety
    
    def _check_sudo_safety(self, command: str, words: List[str]) -> Dict[str, any]:
        """Check safety of sudo commands."""
        
        safety = {
            'is_safe': True,
            'severity': 'medium',
            'warnings': [],
            'risk_factors': []
        }
        
        safety['warnings'].append("Using sudo with elevated privileges")
        safety['risk_factors'].append('elevated_privileges')
        
        # Check what command is being run with sudo
        if len(words) > 1:
            sudo_command = ' '.join(words[1:])
            if any(dangerous in sudo_command for dangerous in ['rm -rf', 'dd', 'mkfs', 'fdisk']):
                safety['is_safe'] = False
                safety['severity'] = 'critical'
                safety['warnings'].append("Dangerous command with sudo privileges")
                safety['risk_factors'].append('dangerous_sudo_command')
        
        return safety
    
    def _check_dd_safety(self, command: str, words: List[str]) -> Dict[str, any]:
        """Check safety of dd commands."""
        
        safety = {
            'is_safe': False,
            'severity': 'critical',
            'warnings': [],
            'risk_factors': []
        }
        
        safety['warnings'].append("dd command can overwrite disks and cause data loss")
        safety['risk_factors'].append('disk_overwrite')
        
        return safety
    
    def _check_mount_safety(self, command: str, words: List[str]) -> Dict[str, any]:
        """Check safety of mount commands."""
        
        safety = {
            'is_safe': True,
            'severity': 'medium',
            'warnings': [],
            'risk_factors': []
        }
        
        safety['warnings'].append("Mount operations can affect system filesystems")
        safety['risk_factors'].append('filesystem_operation')
        
        return safety
    
    def _generate_safety_suggestions(self, command: str, analysis: Dict[str, any]) -> List[str]:
        """Generate safety suggestions based on analysis."""
        
        suggestions = []
        
        # General suggestions based on severity
        if analysis['severity'] == 'critical':
            suggestions.append("Consider using a safer alternative command")
            suggestions.append("Double-check the command before executing")
            suggestions.append("Consider running in a test environment first")
        elif analysis['severity'] == 'high':
            suggestions.append("Add confirmation prompts (-i flag)")
            suggestions.append("Test the command on a small subset first")
        elif analysis['severity'] == 'medium':
            suggestions.append("Review the command parameters carefully")
        
        # Command-specific suggestions
        if 'rm' in command:
            suggestions.append("Consider using 'rm -i' for interactive deletion")
            suggestions.append("Use 'find' with '-delete' for more control")
        
        if 'chmod' in command:
            suggestions.append("Use more restrictive permissions (e.g., 755 instead of 777)")
        
        if 'sudo' in command:
            suggestions.append("Consider if sudo is really necessary")
            suggestions.append("Review what the command will do with elevated privileges")
        
        return suggestions
    
    def _calculate_confidence(self, command: str, analysis: Dict[str, any]) -> int:
        """Calculate confidence in the safety assessment."""
        
        confidence = 100
        
        # Reduce confidence for dangerous commands
        if analysis['severity'] == 'critical':
            confidence -= 50
        elif analysis['severity'] == 'high':
            confidence -= 30
        elif analysis['severity'] == 'medium':
            confidence -= 15
        
        # Reduce confidence for complex commands
        if len(command.split()) > 5:
            confidence -= 10
        
        # Reduce confidence for commands with pipes
        if '|' in command:
            confidence -= 5
        
        # Reduce confidence for commands with redirection
        if '>' in command or '<' in command:
            confidence -= 5
        
        return max(0, confidence)
    
    def get_safety_report(self, command: str) -> Dict[str, any]:
        """Generate a comprehensive safety report for a command."""
        
        is_safe, warning, analysis = self.validate_command_safety(command)
        
        report = {
            'command': command,
            'is_safe': is_safe,
            'warning': warning,
            'analysis': analysis,
            'recommendations': self._get_safety_recommendations(command, analysis),
            'alternatives': self._get_safer_alternatives(command),
            'timestamp': self._get_timestamp()
        }
        
        return report
    
    def _get_safety_recommendations(self, command: str, analysis: Dict[str, any]) -> List[str]:
        """Get safety recommendations for a command."""
        
        recommendations = []
        
        if not analysis['is_safe']:
            recommendations.append("DO NOT EXECUTE - Command is unsafe")
            recommendations.append("Review the command and use safer alternatives")
        
        if analysis['severity'] in ['high', 'critical']:
            recommendations.append("Add confirmation prompts")
            recommendations.append("Test in a safe environment first")
        
        recommendations.extend(analysis['suggestions'])
        
        return recommendations
    
    def _get_safer_alternatives(self, command: str) -> List[str]:
        """Get safer alternatives for dangerous commands."""
        
        alternatives = []
        words = command.split()
        
        if not words:
            return alternatives
        
        command_name = words[0].lower()
        
        # Provide safer alternatives
        if command_name == 'rm':
            if '-rf' in words:
                alternatives.append(command.replace('-rf', '-ri'))  # Interactive recursive
                alternatives.append(command.replace('rm -rf', 'find . -name'))
        
        elif command_name == 'chmod':
            if '777' in command:
                alternatives.append(command.replace('777', '755'))
                alternatives.append(command.replace('777', '644'))
        
        elif command_name == 'sudo':
            alternatives.append(command.replace('sudo ', ''))  # Try without sudo first
        
        return alternatives
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_safety_stats(self) -> Dict[str, any]:
        """Get statistics about safety checks."""
        
        return {
            'dangerous_patterns_count': sum(len(patterns) for patterns in self.dangerous_patterns.values()),
            'safe_commands_count': len(self.safe_commands),
            'special_attention_commands_count': len(self.special_attention_commands),
            'severity_levels': list(self.dangerous_patterns.keys())
        }
