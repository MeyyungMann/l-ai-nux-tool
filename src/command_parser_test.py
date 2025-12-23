"""
Unit tests for command parser.
"""

import pytest
from .command_parser import CommandParser


@pytest.mark.unit
class TestCommandParser:
    """Test command parsing and validation."""
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = CommandParser()
        assert parser is not None
    
    def test_parse_valid_command(self):
        """Test parsing valid commands."""
        parser = CommandParser()
        result = parser.parse_command("ls -la")
        assert result == "ls -la"
    
    def test_parse_command_with_markdown(self):
        """Test parsing commands with markdown formatting."""
        parser = CommandParser()
        result = parser.parse_command("```bash\nls -la\n```")
        assert "ls -la" in result
        assert "```" not in result
    
    def test_parse_command_strips_whitespace(self):
        """Test that parser strips whitespace."""
        parser = CommandParser()
        result = parser.parse_command("  ls -la  ")
        assert result == "ls -la"
    
    def test_validate_dangerous_commands(self):
        """Test detection of dangerous commands."""
        parser = CommandParser()
        
        dangerous = [
            "rm -rf /",
            "dd if=/dev/zero",
            "mkfs /dev/sda",
            "fdisk /dev/sda"
        ]
        
        for cmd in dangerous:
            with pytest.raises(ValueError):
                parser.parse_command(cmd)
    
    def test_validate_empty_command(self):
        """Test validation of empty commands."""
        parser = CommandParser()
        
        with pytest.raises(ValueError):
            parser.parse_command("")
        
        with pytest.raises(ValueError):
            parser.parse_command("   ")
    
    def test_is_valid_command_structure(self):
        """Test command structure validation."""
        parser = CommandParser()
        
        valid = [
            "ls",
            "find . -name '*.txt'",
            "grep -r pattern /path"
        ]
        
        for cmd in valid:
            assert parser._is_valid_command(cmd) is True
        
        invalid = [
            "",
            "   ",
            "rm -rf /"
        ]
        
        for cmd in invalid:
            assert parser._is_valid_command(cmd) is False

