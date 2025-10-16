"""Debug online-rag in Docker to see what API returns."""

from src.config import Config
from src.llm_engine import LLMEngine
import os

# Set up config
config = Config()
config.set('mode', 'online-rag')
config.set('api.model', 'gpt-5-mini')

engine = LLMEngine(config)

query = "find duplicate files by name"

print("=" * 80)
print(f"Debugging: {query}")
print("=" * 80)

# Get RAG context
context = engine.rag_engine.get_context_for_query(query, top_k=3)
print(f"\n1. RAG Context (first 300 chars):")
print(context[:300])
print("...")

# Clean templates
cleaned = engine._clean_rag_templates(context)
print(f"\n2. Cleaned (first 300 chars):")
print(cleaned[:300])
print("...")

# Create prompt
prompt = engine._create_online_rag_prompt(query, cleaned)
print(f"\n3. Prompt length: {len(prompt)} chars")
print(f"   Last 200 chars:")
print(prompt[-200:])

# Make API call directly to see raw response
print(f"\n4. Making API call...")
response = engine.client.chat.completions.create(
    model='gpt-5-mini',
    messages=[
        {"role": "system", "content": "You are a Linux command expert. Generate accurate Linux commands based on the provided context and user request."},
        {"role": "user", "content": prompt}
    ],
    max_completion_tokens=200
)

raw = response.choices[0].message.content
print(f"\n5. Raw API Response:")
print(f"   Type: {type(raw)}")
print(f"   Length: {len(raw)}")
print(f"   Content: '{raw}'")

# Extract
print(f"\n6. Extracting command...")
extracted = engine._extract_command(raw, prompt)
print(f"   Extracted: '{extracted}'")
print(f"   Length: {len(extracted)}")

if extracted and extracted.strip():
    print(f"\n✅ Extraction worked")
    
    # Try validation
    from src.command_parser import CommandParser
    parser = CommandParser()
    try:
        parsed = parser.parse_command(extracted)
        print(f"✅ Validation passed: {parsed}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
else:
    print(f"\n❌ Extraction returned empty!")
    print(f"\nDebugging extraction:")
    
    # Try manual extraction
    lines = raw.split('\n')
    print(f"   Raw has {len(lines)} lines")
    for i, line in enumerate(lines):
        print(f"   Line {i}: {repr(line)}")


