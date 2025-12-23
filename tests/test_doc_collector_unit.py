"""
Unit tests for document collector.
"""

import pytest
import json
from pathlib import Path
from src.doc_collector import LinuxDocCollector, CommandDoc


@pytest.mark.unit
class TestCommandDoc:
    """Test CommandDoc class."""
    
    def test_command_doc_creation(self):
        """Test creating a CommandDoc."""
        doc = CommandDoc(
            command="ls",
            description="List files",
            usage="ls [options]",
            examples=["ls -la"],
            options={"-l": "Long format"},
            category="file_operations"
        )
        
        assert doc.command == "ls"
        assert doc.description == "List files"
        assert len(doc.examples) == 1
    
    def test_command_doc_to_dict(self):
        """Test converting CommandDoc to dictionary."""
        doc = CommandDoc(
            command="find",
            description="Find files",
            examples=["find . -name '*.txt'"]
        )
        
        doc_dict = doc.to_dict()
        assert isinstance(doc_dict, dict)
        assert doc_dict['command'] == "find"
        assert 'examples' in doc_dict


@pytest.mark.unit
class TestLinuxDocCollector:
    """Test LinuxDocCollector."""
    
    def test_collector_initialization(self, temp_cache_dir):
        """Test collector initialization."""
        collector = LinuxDocCollector(temp_cache_dir)
        assert collector.cache_dir == temp_cache_dir
    
    def test_save_and_load_docs(self, temp_cache_dir, sample_command_docs):
        """Test saving and loading documents."""
        collector = LinuxDocCollector(temp_cache_dir)
        
        # Save docs
        success = collector.save_docs(sample_command_docs)
        assert success is True
        
        # Load docs
        loaded_docs = collector.load_docs()
        assert len(loaded_docs) == len(sample_command_docs)
        assert loaded_docs[0].command == sample_command_docs[0].command
    
    def test_load_nonexistent_cache(self, temp_cache_dir):
        """Test loading from non-existent cache."""
        collector = LinuxDocCollector(temp_cache_dir)
        docs = collector.load_docs()
        assert docs == []
    
    def test_collector_cache_file_path(self, temp_cache_dir):
        """Test cache file path construction."""
        collector = LinuxDocCollector(temp_cache_dir)
        expected_path = temp_cache_dir / "linux_docs.json"
        assert collector.docs_file == expected_path

