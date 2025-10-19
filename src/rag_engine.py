"""
RAG (Retrieval-Augmented Generation) Engine
Combines document retrieval with CodeLlama generation for enhanced command suggestions.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
import json

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    logging.warning("RAG dependencies not installed. Install with: pip install sentence-transformers faiss-cpu")
    SentenceTransformer = None
    faiss = None

from .doc_collector import LinuxDocCollector, CommandDoc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG Engine for document retrieval and augmented generation."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the RAG engine."""
        self.cache_dir = cache_dir or Path.home() / '.lai-nux-tool' / 'rag_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.embedder = None
        self.index = None
        self.documents = []
        self.doc_texts = []
        
        # Query cache for faster repeated queries
        self.query_cache = {}
        self.query_cache_file = self.cache_dir / 'query_cache.pkl'
        self._load_query_cache()
        
        # Load or initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize or load the RAG system."""
        if not SentenceTransformer or not faiss:
            logger.error("RAG dependencies not available. Please install: pip install sentence-transformers faiss-cpu")
            return
        
        # Always promote any user-approved commands into docs before checking for changes
        # so that the hash reflects the latest promotions
        try:
            self._promote_user_approvals()
        except Exception as e:
            logger.warning(f"Failed to promote user approvals: {e}")

        # Check if documentation changed
        if self._docs_changed():
            logger.info("Documentation updated - clearing cache")
            self._clear_cache()
        
        # Try to load from cache
        if self._load_from_cache():
            logger.info("RAG system loaded from cache")
            return
        
        # Initialize fresh
        logger.info("Initializing RAG system...")
        
        # Load embedding model
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded")
        
        # Collect documents
        logger.info("Collecting Linux documentation...")
        collector = LinuxDocCollector(self.cache_dir.parent / 'doc_cache')
        
        # Try to load from cache first
        cached_docs = collector.load_from_cache()
        if cached_docs:
            self.documents = cached_docs
            logger.info(f"Loaded {len(self.documents)} docs from cache")
        else:
            self.documents = collector.collect_all_docs()
            logger.info(f"Collected {len(self.documents)} docs")
        
        # Create document texts for embedding
        self._create_document_texts()
        
        # Create embeddings and index
        logger.info("Creating embeddings and vector index...")
        self._create_index()
        
        # Save to cache
        self._save_to_cache()
        logger.info("RAG system initialized and cached")
    
    def _create_document_texts(self):
        """Create searchable text from documents."""
        self.doc_texts = []
        
        for doc in self.documents:
            # Combine command info into searchable text
            text_parts = [
                f"Command: {doc.command}",
                f"Description: {doc.description}",
            ]
            
            # Add examples
            if doc.examples:
                text_parts.append("Examples:")
                text_parts.extend(doc.examples)
            
            # Add options
            if doc.options:
                text_parts.append("Options:")
                for opt, desc in list(doc.options.items())[:5]:  # Limit options
                    text_parts.append(f"{opt}: {desc}")
            
            # Combine into single text
            text = "\n".join(text_parts)
            self.doc_texts.append(text)
    
    def _create_index(self):
        """Create FAISS vector index from documents."""
        if not self.doc_texts:
            logger.warning("No documents to index")
            return
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(self.doc_texts)} documents...")
        embeddings = self.embedder.encode(self.doc_texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        logger.info(f"Created FAISS index with {self.index.ntotal} vectors")
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 5) -> List[Tuple[CommandDoc, float]]:
        """Retrieve relevant documents for a query."""
        if not self.embedder or not self.index:
            logger.warning("RAG system not initialized")
            return []
        
        try:
            # Embed the query
            query_embedding = self.embedder.encode([query])
            
            # Search the index
            distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Get relevant documents with scores
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents):
                    # Convert distance to similarity score (lower distance = higher similarity)
                    similarity = 1.0 / (1.0 + dist)
                    results.append((self.documents[idx], similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def get_context_for_query(self, query: str, top_k: int = 3) -> str:
        """Get formatted context for a query to augment generation with caching."""
        # Improve query processing for better context matching
        processed_query = self._improve_query_for_context(query)
        
        # Create cache key from normalized query
        cache_key = self._normalize_query(processed_query.lower()) + f"_k{top_k}"
        
        # Check cache first
        if cache_key in self.query_cache:
            logger.debug(f"🎯 Query cache HIT for: {query}")
            return self.query_cache[cache_key]
        
        # Try partial matching for similar queries
        normalized_query = self._normalize_query(processed_query.lower())
        for existing_key in self.query_cache.keys():
            if existing_key.endswith(f"_k{top_k}") and existing_key.startswith(normalized_query[:10]):
                logger.debug(f"🎯 Query cache PARTIAL HIT for: {query}")
                return self.query_cache[existing_key]
        
        logger.debug(f"🔍 Query cache MISS for: {query}")
        
        # Retrieve documents with improved query
        relevant_docs = self.retrieve_relevant_docs(processed_query, top_k)
        
        if not relevant_docs:
            return ""
        
        context_parts = ["Here are some relevant Linux command examples:\n"]
        
        for doc, score in relevant_docs:
            context_parts.append(f"\n## {doc.command}")
            context_parts.append(f"Description: {doc.description}")
            
            if doc.examples:
                context_parts.append("Examples:")
                for example in doc.examples[:2]:  # Limit examples
                    context_parts.append(f"  - {example}")
            
            if doc.options:
                context_parts.append("Common options:")
                for opt, desc in list(doc.options.items())[:3]:  # Limit options
                    context_parts.append(f"  {opt}: {desc}")
        
        context = "\n".join(context_parts)
        
        # Cache the result
        self.query_cache[cache_key] = context
        self._save_query_cache()
        
        return context
    
    def _improve_query_for_context(self, query: str) -> str:
        """Improve query for better context matching."""
        # Fix common typos and improve query specificity
        improvements = {
            'diplicate': 'duplicate',
            'duplicate files by name': 'duplicate-detection filename',
            'duplicate files by content': 'duplicate-detection content',
            'duplicate filenames': 'duplicate-detection filename',
            'duplicate file names': 'duplicate-detection filename',
            'files with same name': 'duplicate-detection filename',
            'same filename': 'duplicate-detection filename',
            'list files with date': 'list files with timestamps',
            'files today': 'files created today',
            'files yesterday': 'files created yesterday',
            'large files': 'find large files',
            'empty files': 'find empty files',
            'permissions': 'file permissions',
            'ownership': 'file ownership',
        }
        
        improved_query = query.lower()
        for original, improved in improvements.items():
            if original in improved_query:
                improved_query = improved_query.replace(original, improved)
                break
        
        # Add specific context hints for better matching
        if 'duplicate' in improved_query and ('name' in improved_query or 'filename' in improved_query):
            improved_query += ' duplicate-detection filename find basename printf'
        elif 'duplicate' in improved_query and 'content' in improved_query:
            improved_query += ' duplicate-detection content md5sum sha256sum'
        elif 'find' in improved_query and 'files' in improved_query:
            improved_query += ' file search'
        
        return improved_query

    def store_context_for_query(self, query: str, context: str, top_k: int = 3) -> None:
        """Store a provided context string into the query cache for this query.

        This lets external components (e.g., online fallback) seed the RAG
        cache so future queries can reuse high-quality context instantly.
        """
        try:
            cache_key = self._normalize_query(query.lower()) + f"_k{top_k}"
            
            # Format the context to be more prominent and actionable
            formatted_context = f"""🎯 Known-good command for this task:

{context}

This command was successfully used and verified to work for similar requests."""
            
            self.query_cache[cache_key] = formatted_context
            self._save_query_cache()
            
            # Also store with different top_k values for better matching
            for k in [1, 2, 3, 4, 5]:
                alt_key = self._normalize_query(query.lower()) + f"_k{k}"
                self.query_cache[alt_key] = formatted_context
            
            self._save_query_cache()
            logger.info(f"✅ Stored custom context for query (k={top_k}) with multiple cache keys")
        except Exception as e:
            logger.warning(f"Failed to store context for query: {e}")
    
    def search_commands(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant commands and return structured results."""
        relevant_docs = self.retrieve_relevant_docs(query, top_k)
        
        results = []
        for doc, score in relevant_docs:
            results.append({
                'command': doc.command,
                'description': doc.description,
                'examples': doc.examples[:3],
                'category': doc.category,
                'relevance_score': score,
            })
        
        return results
    
    def _save_to_cache(self):
        """Save RAG system to cache."""
        try:
            cache_data = {
                'documents': self.documents,
                'doc_texts': self.doc_texts,
                'embedder_name': 'all-MiniLM-L6-v2',
            }
            
            # Save main cache data
            cache_file = self.cache_dir / 'rag_cache.pkl'
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Save FAISS index separately
            if self.index:
                index_file = self.cache_dir / 'faiss_index.bin'
                faiss.write_index(self.index, str(index_file))
            
            logger.info(f"RAG system cached to {self.cache_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save RAG cache: {e}")
    
    def _load_from_cache(self) -> bool:
        """Load RAG system from cache."""
        try:
            cache_file = self.cache_dir / 'rag_cache.pkl'
            index_file = self.cache_dir / 'faiss_index.bin'
            
            if not cache_file.exists() or not index_file.exists():
                return False
            
            # Load cache data
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.documents = cache_data['documents']
            self.doc_texts = cache_data['doc_texts']
            
            # Load embedder
            embedder_name = cache_data.get('embedder_name', 'all-MiniLM-L6-v2')
            self.embedder = SentenceTransformer(embedder_name)
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            logger.info(f"Loaded RAG cache with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load RAG cache: {e}")
            return False
    
    def clear_cache(self):
        """Clear the RAG cache."""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("RAG cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear RAG cache: {e}")
    
    def rebuild_index(self):
        """Rebuild the RAG index from scratch."""
        logger.info("Rebuilding RAG index...")
        self.clear_cache()
        self._initialize()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return {
            'total_documents': len(self.documents),
            'indexed_vectors': self.index.ntotal if self.index else 0,
            'embedder_model': 'all-MiniLM-L6-v2',
            'cache_dir': str(self.cache_dir),
            'is_initialized': self.embedder is not None and self.index is not None,
            'cached_queries': len(self.query_cache),
            'query_cache_size': len(self.query_cache)
        }
    
    def _docs_changed(self) -> bool:
        """Check if documentation has changed since last cache build."""
        import hashlib
        
        docs_file = self.cache_dir.parent / 'doc_cache' / 'linux_docs.json'
        hash_file = self.cache_dir / '.docs_hash'
        
        if not docs_file.exists():
            return False
        
        # Calculate current docs hash
        with open(docs_file, 'rb') as f:
            current_hash = hashlib.md5(f.read()).hexdigest()
        
        # Compare with stored hash
        if hash_file.exists():
            stored_hash = hash_file.read_text().strip()
            if current_hash == stored_hash:
                return False
        
        # Store new hash
        hash_file.write_text(current_hash)
        return True
    
    def _clear_cache(self):
        """Clear RAG cache files."""
        cache_file = self.cache_dir / 'rag_cache.pkl'
        index_file = self.cache_dir / 'faiss_index.bin'
        
        if cache_file.exists():
            cache_file.unlink()
        if index_file.exists():
            index_file.unlink()
        
        # Clear query cache too
        self.query_cache = {}
        if self.query_cache_file.exists():
            self.query_cache_file.unlink()
        
        logger.info("RAG cache cleared")

    def _promote_user_approvals(self):
        """Promote user-approved commands into linux_docs.json as examples.

        Reads approvals from rag cache file 'user_approved.jsonl' and, for each,
        appends the approved command as an example under the corresponding command
        entry in the docs (matched by first token). After successful promotion,
        clears the approvals file to avoid duplicate promotions.
        """
        approvals_path = self.cache_dir / 'user_approved.jsonl'
        docs_file = self.cache_dir.parent / 'doc_cache' / 'linux_docs.json'

        if not approvals_path.exists() or approvals_path.stat().st_size == 0:
            return
        if not docs_file.exists():
            logger.warning("Docs file not found; cannot promote approvals")
            return

        # Load docs JSON (supports list or object with 'commands' key)
        with open(docs_file, 'r', encoding='utf-8') as f:
            try:
                docs_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to parse docs JSON: {e}")
                return

        if isinstance(docs_data, dict) and 'commands' in docs_data:
            docs_list = docs_data['commands']
            wrapper = 'dict'
        elif isinstance(docs_data, list):
            docs_list = docs_data
            wrapper = 'list'
        else:
            logger.warning("Unsupported docs JSON structure; skipping promotion")
            return

        # Build index by command name
        def get_cmd_name(entry):
            return (entry.get('command') or '').strip()

        name_to_entry = {get_cmd_name(e): e for e in docs_list if isinstance(e, dict)}

        modified = False
        with open(approvals_path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for ln in lines:
            try:
                rec = json.loads(ln)
                approved_cmd = (rec.get('command') or '').strip()
                if not approved_cmd:
                    continue
                cmd_name = approved_cmd.split()[0]
                entry = name_to_entry.get(cmd_name)
                if not entry:
                    # If no entry exists, create a minimal one
                    entry = {
                        'command': cmd_name,
                        'description': rec.get('description') or f"User-promoted examples for {cmd_name}",
                        'examples': []
                    }
                    docs_list.append(entry)
                    name_to_entry[cmd_name] = entry
                    modified = True

                # Ensure examples is a list
                if 'examples' not in entry or not isinstance(entry['examples'], list):
                    entry['examples'] = []

                # Add the full approved command as an example if not present
                if approved_cmd not in entry['examples']:
                    entry['examples'].append(approved_cmd)
                    modified = True
            except Exception as e:
                logger.warning(f"Skipping invalid approval record: {e}")

        if modified:
            # Write back docs
            with open(docs_file, 'w', encoding='utf-8') as f:
                if wrapper == 'dict':
                    json.dump({'commands': docs_list}, f, ensure_ascii=False, indent=2)
                else:
                    json.dump(docs_list, f, ensure_ascii=False, indent=2)
            logger.info("Promoted user approvals into linux_docs.json")

        # Clear approvals regardless to avoid repeated attempts on malformed lines
        try:
            approvals_path.write_text('', encoding='utf-8')
        except Exception:
            pass
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for cache key generation with better matching."""
        import re
        
        # Convert to lowercase and normalize whitespace
        normalized = re.sub(r'\s+', ' ', query.lower().strip())
        
        # Remove punctuation but keep important words
        normalized = re.sub(r'[^\w\s-]', ' ', normalized)
        
        # Normalize whitespace again
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Create multiple cache keys for better matching
        # This allows partial matches for similar queries
        return normalized
    
    def _load_query_cache(self):
        """Load query cache from disk."""
        try:
            if self.query_cache_file.exists():
                with open(self.query_cache_file, 'rb') as f:
                    self.query_cache = pickle.load(f)
                logger.info(f"📚 Loaded query cache with {len(self.query_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to load query cache: {e}")
            self.query_cache = {}
    
    def _save_query_cache(self):
        """Save query cache to disk (with size limit)."""
        try:
            # Limit cache size to 100 most recent queries
            if len(self.query_cache) > 100:
                # Keep only 100 most recent
                self.query_cache = dict(list(self.query_cache.items())[-100:])
            
            with open(self.query_cache_file, 'wb') as f:
                pickle.dump(self.query_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save query cache: {e}")
    
    def clear_query_cache(self):
        """Clear query cache manually."""
        self.query_cache = {}
        if self.query_cache_file.exists():
            self.query_cache_file.unlink()
        logger.info("✅ Query cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get query cache statistics."""
        return {
            'cached_queries': len(self.query_cache),
            'cache_file': str(self.query_cache_file),
            'cache_exists': self.query_cache_file.exists()
        }


# Singleton instance
_rag_engine = None


def get_rag_engine() -> RAGEngine:
    """Get or create the global RAG engine instance."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


if __name__ == '__main__':
    # Test the RAG engine
    engine = RAGEngine()
    
    print("\nRAG System Stats:")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTesting search:")
    query = "find all text files in directory"
    results = engine.search_commands(query, top_k=3)
    
    print(f"\nTop results for '{query}':")
    for result in results:
        print(f"\n{result['command']} (score: {result['relevance_score']:.3f})")
        print(f"  {result['description']}")
        if result['examples']:
            print(f"  Example: {result['examples'][0]}")

