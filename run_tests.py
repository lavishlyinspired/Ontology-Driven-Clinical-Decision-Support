"""
Test Runner for LCA System

Runs all tests and provides summary report
"""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run all test suites"""
    print("=" * 80)
    print("LUNG CANCER ASSISTANT - TEST SUITE")
    print("=" * 80)
    print()

    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
        import pytest

    # Run tests
    test_dir = Path(__file__).parent / "tests"

    print(f"Running tests from: {test_dir}")
    print()

    # Run component tests
    print("📦 Component Tests")
    print("-" * 80)
    result_component = pytest.main([
        str(test_dir / "test_components.py"),
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ])

    print()
    print("=" * 80)

    if result_component == 0:
        print("✅ ALL TESTS PASSED!")
        print()
        print("Test Coverage:")
        print("  ✓ Dynamic Orchestrator (complexity assessment, workflow routing)")
        print("  ✓ Context Graphs (nodes, edges, conflict detection)")
        print("  ✓ NSCLC Agent (stage-based recommendations)")
        print("  ✓ SCLC Agent (limited/extensive stage protocols)")
        print("  ✓ Biomarker Agent (EGFR, ALK detection)")
        print("  ✓ Negotiation Protocol (evidence hierarchy, safety-first, consensus)")
        print("  ✓ Module Imports (all agents and analytics)")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
