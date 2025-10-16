"""
Document collector for Linux command documentation.
"""

from typing import List, Dict, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class CommandDoc:
    """Represents a command documentation entry."""
    
    def __init__(self, command: str, description: str, usage: str = "", examples: str = "", options: Dict[str, str] = None, category: str = ""):
        self.command = command
        self.description = description
        self.usage = usage
        # Convert examples to list if it's a string
        if isinstance(examples, str):
            self.examples = [examples] if examples else []
        else:
            self.examples = examples or []
        self.options = options or {}
        self.category = category
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "command": self.command,
            "description": self.description,
            "usage": self.usage,
            "examples": self.examples,
            "options": self.options,
            "category": self.category
        }


class LinuxDocCollector:
    """Collects and manages Linux command documentation."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.docs_file = cache_dir / "linux_docs.json"
        
    def load_docs(self) -> List[CommandDoc]:
        """Load documentation from cache file."""
        if not self.docs_file.exists():
            logger.warning(f"No documentation cache found at {self.docs_file}")
            return []
        
        try:
            with open(self.docs_file, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)
            
            docs = []
            for doc_data in docs_data:
                doc = CommandDoc(
                    command=doc_data.get('command', ''),
                    description=doc_data.get('description', ''),
                    usage=doc_data.get('usage', ''),
                    examples=doc_data.get('examples', ''),
                    options=doc_data.get('options', {}),
                    category=doc_data.get('category', '')
                )
                docs.append(doc)
            
            logger.info(f"Loaded {len(docs)} command documents from cache")
            return docs
            
        except Exception as e:
            logger.error(f"Error loading documentation: {e}")
            return []
    
    def save_docs(self, docs: List[CommandDoc]) -> bool:
        """Save documentation to cache file."""
        try:
            docs_data = [doc.to_dict() for doc in docs]
            
            with open(self.docs_file, 'w', encoding='utf-8') as f:
                json.dump(docs_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(docs)} command documents to cache")
            return True
            
        except Exception as e:
            logger.error(f"Error saving documentation: {e}")
            return False
    
    def load_from_cache(self) -> List[CommandDoc]:
        """Load documentation from cache file (alias for load_docs)."""
        return self.load_docs()
    
    def collect_docs(self) -> List[CommandDoc]:
        """Collect Linux command documentation."""
        # For now, just load from existing cache
        # In a full implementation, this would fetch from man pages, etc.
        return self.load_docs()
    
    def collect_all_docs(self) -> List[CommandDoc]:
        """Collect all Linux command documentation (alias for collect_docs)."""
        return self.collect_docs()
