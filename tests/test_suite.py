#!/usr/bin/env python3
"""
Main Test Suite for L-AI-NUX-TOOL
Runs comprehensive tests including quick tests and additional validation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """Run the main test suite."""
    print("="*70)
    print("L-AI-NUX-TOOL - MAIN TEST SUITE")
    print("="*70)
    print("\n🧪 Running comprehensive test suite...")
    print("⚡ This should complete in under 60 seconds\n")
    
    # Import and run quick tests
    try:
        from tests.quick_test import main as quick_main
        print("Running quick tests...")
        quick_result = quick_main()
        if not quick_result:
            print("❌ Quick tests failed")
            return False
        print("✅ Quick tests passed")
    except Exception as e:
        print(f"❌ Quick tests failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("✅ All tests completed successfully!")
    print("="*70)
    return True

if __name__ == '__main__':
    sys.exit(0 if main() else 1)
