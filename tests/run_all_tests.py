#!/usr/bin/env python3
"""
Comprehensive test runner for SurvivalProject.
Runs all test modules and provides a summary.
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_test_module(module_name, test_function):
    """Run a test module and return success status."""
    try:
        print(f"\n{'='*60}")
        print(f"Running {module_name}")
        print(f"{'='*60}")
        test_function()
        print(f"✅ {module_name} PASSED")
        return True
    except Exception as e:
        print(f"❌ {module_name} FAILED")
        print(f"Error: {str(e)}")
        print(f"Traceback:")
        traceback.print_exc()
        return False

def main():
    """Run all tests and provide summary."""
    print("🧪 SurvivalProject Test Suite")
    print("="*60)
    
    # Import test modules
    try:
        from test_dataloaders import main as test_dataloaders
        from test_architectures import main as test_architectures  
        from test_data_preprocessing import main as test_data_preprocessing
    except ImportError as e:
        print(f"❌ Failed to import test modules: {e}")
        return 1
    
    # Define test modules
    test_modules = [
        ("Data Preprocessing Tests", test_data_preprocessing),
        ("Architecture Tests", test_architectures),
        ("Dataloader Tests", test_dataloaders),
    ]
    
    # Run all tests
    results = []
    for module_name, test_function in test_modules:
        success = run_test_module(module_name, test_function)
        results.append((module_name, success))
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for module_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{module_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} test modules passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
