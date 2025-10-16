#!/usr/bin/env python3
"""
Dedicated RAG (Retrieval-Augmented Generation) tests
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_rag_dependencies():
    """Test that RAG dependencies are available."""
    print("\n" + "="*70)
    print("TEST: RAG Dependencies")
    print("="*70)
    
    dependencies = {
        'sentence_transformers': 'Sentence Transformers',
        'faiss': 'FAISS',
        'beautifulsoup4': 'BeautifulSoup4',
        'lxml': 'LXML',
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            if module == 'beautifulsoup4':
                __import__('bs4')
            else:
                __import__(module)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed")
            missing.append(module)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Install with:")
        print("  pip install sentence-transformers faiss-cpu beautifulsoup4 lxml")
        return False
    
    print("\n✅ All RAG dependencies available")
    return True


def test_doc_collector():
    """Test documentation collector."""
    print("\n" + "="*70)
    print("TEST: Documentation Collector")
    print("="*70)
    
    try:
        from src.doc_collector import LinuxDocCollector
        
        collector = LinuxDocCollector()
        print("✅ LinuxDocCollector initialized")
        
        # Check cache
        docs = collector.load_from_cache()
        
        if docs:
            print(f"✅ Loaded {len(docs)} documents from cache")
            
            # Show categories
            categories = set(doc.category for doc in docs)
            print(f"\n📚 Categories ({len(categories)}):")
            for cat in sorted(categories)[:10]:
                count = sum(1 for doc in docs if doc.category == cat)
                print(f"   • {cat}: {count} commands")
            
            # Show sample commands
            print(f"\n📄 Sample Commands:")
            for doc in docs[:5]:
                print(f"   • {doc.command}: {doc.description[:60]}...")
            
            return True
        else:
            print("⚠️  No cache found")
            print("Run to create cache:")
            print("  python test_rag_system.py")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_engine():
    """Test RAG engine functionality."""
    print("\n" + "="*70)
    print("TEST: RAG Engine")
    print("="*70)
    
    try:
        from src.rag_engine import RAGEngine
        
        engine = RAGEngine()
        print("✅ RAG engine initialized")
        
        # Get stats
        stats = engine.get_stats()
        print(f"\n📊 Statistics:")
        print(f"   Documents: {stats['total_documents']}")
        print(f"   Vectors: {stats['indexed_vectors']}")
        print(f"   Model: {stats['embedder_model']}")
        print(f"   Cache: {stats['cache_dir']}")
        
        if stats['total_documents'] == 0:
            print("\n⚠️  No documents indexed")
            return False
        
        # Test search
        queries = [
            "find all text files",
            "list files by size",
            "show disk usage",
            "compress directory",
            "change file permissions",
        ]
        
        print(f"\n🔍 Testing searches:")
        for query in queries:
            results = engine.search_commands(query, top_k=3)
            print(f"\n   Query: '{query}'")
            
            if results:
                for i, result in enumerate(results, 1):
                    cmd = result['command']
                    score = result['relevance_score']
                    print(f"      {i}. {cmd} (score: {score:.3f})")
            else:
                print(f"      No results")
        
        print("\n✅ RAG search working")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_context_generation():
    """Test RAG context generation for LLM."""
    print("\n" + "="*70)
    print("TEST: RAG Context Generation")
    print("="*70)
    
    try:
        from src.rag_engine import RAGEngine
        
        engine = RAGEngine()
        stats = engine.get_stats()
        
        if stats['total_documents'] == 0:
            print("⚠️  Skipping (no documents)")
            return False
        
        # Test context generation
        query = "find all text files recursively"
        print(f"\nQuery: '{query}'")
        
        context = engine.get_context_for_query(query, top_k=3)
        
        if context:
            print(f"\n✅ Generated context ({len(context)} chars)")
            print(f"\n📝 Context preview:")
            print("-" * 70)
            print(context[:500] + "..." if len(context) > 500 else context)
            print("-" * 70)
            return True
        else:
            print("❌ No context generated")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Run all RAG tests."""
    print("="*70)
    print("L-AI-NUX-TOOL - RAG TESTS")
    print("="*70)
    
    tests = [
        ("RAG Dependencies", test_rag_dependencies),
        ("Documentation Collector", test_doc_collector),
        ("RAG Engine", test_rag_engine),
        ("Context Generation", test_rag_context_generation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("RAG TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 All RAG tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

