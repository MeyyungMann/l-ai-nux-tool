"""Verify online-rag mode works end-to-end."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, 'src')

from src.llm_engine import LLMEngine
from src.config import Config

print("=" * 80)
print("🧪 Testing Online-RAG Mode")
print("=" * 80)

# Setup
config = Config()
config.set('api.model', 'gpt-5-mini')

engine = LLMEngine(config=config)
engine.switch_mode('online-rag')

print("\n✅ Engine initialized in online-rag mode")

# Test queries
test_queries = [
    "list all files",
    "show disk usage",
    "find jpg files",
]

print(f"\n🧪 Testing {len(test_queries)} queries...")
print("=" * 80)

success_count = 0
for i, query in enumerate(test_queries, 1):
    print(f"\n{i}. Query: '{query}'")
    try:
        result = engine.generate_command(query)
        if result and result.strip():
            print(f"   ✅ Generated: {result}")
            success_count += 1
        else:
            print(f"   ❌ Empty result")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 80)
print(f"📊 Results: {success_count}/{len(test_queries)} successful")
print("=" * 80)

if success_count == len(test_queries):
    print("\n🎉 All tests passed! Online-RAG mode is working correctly.")
    sys.exit(0)
else:
    print(f"\n❌ {len(test_queries) - success_count} tests failed.")
    sys.exit(1)


