#!/usr/bin/env python3
"""
Update RAG cache when documentation changes
"""

import os
import shutil
from pathlib import Path
import hashlib
import json

def get_docs_hash():
    """Get hash of documentation file to detect changes."""
    docs_file = Path("doc_cache/linux_docs.json")
    if not docs_file.exists():
        return None
    
    with open(docs_file, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_cached_hash():
    """Get stored hash from last build."""
    hash_file = Path("rag_cache/.docs_hash")
    if not hash_file.exists():
        return None
    
    return hash_file.read_text().strip()

def save_docs_hash(doc_hash):
    """Save current docs hash."""
    hash_file = Path("rag_cache/.docs_hash")
    hash_file.parent.mkdir(exist_ok=True)
    hash_file.write_text(doc_hash)

def clear_rag_cache():
    """Clear RAG cache to force rebuild."""
    cache_dir = Path("rag_cache")
    if cache_dir.exists():
        # Keep the directory but remove cache files
        for file in cache_dir.glob("*"):
            if file.name != ".docs_hash":
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
        print("🔄 RAG cache cleared - will rebuild on next use")
        return True
    return False

def check_and_update_cache():
    """Check if docs changed and update cache if needed."""
    current_hash = get_docs_hash()
    cached_hash = get_cached_hash()
    
    if current_hash is None:
        print("❌ No documentation file found")
        return False
    
    if current_hash != cached_hash:
        print("📝 Documentation changed - updating RAG cache...")
        clear_rag_cache()
        save_docs_hash(current_hash)
        print("✅ RAG cache will rebuild on next use")
        return True
    else:
        print("✅ RAG cache is up to date")
        return False

if __name__ == "__main__":
    check_and_update_cache()

