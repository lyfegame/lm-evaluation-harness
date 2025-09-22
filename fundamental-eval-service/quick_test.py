#!/usr/bin/env python3
"""
Quick test to verify the web service setup works.
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test that we can import our modules."""
    print("🔍 Testing basic imports...")
    
    try:
        from config import settings
        print(f"✅ Config: {settings.app_name}")
        
        from routes import health, evaluation
        print("✅ Routes imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_existing_results():
    """Test that we have existing results to analyze."""
    print("📊 Testing existing results...")
    
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        print("❌ No artifacts directory")
        return False
    
    result_dirs = list(artifacts_dir.glob("*"))
    if not result_dirs:
        print("❌ No evaluation results found")
        return False
    
    print(f"✅ Found {len(result_dirs)} evaluation results")
    
    # Check the most recent result
    latest_result = max(result_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 Latest result: {latest_result.name}")
    
    # Check for results.json
    results_file = latest_result / "results.json"
    if results_file.exists():
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        solve_rate = results.get("solve_rate", 0.0)
        print(f"   Solve rate: {solve_rate:.2%}")
        return True
    else:
        print("❌ No results.json found")
        return False

def test_analysis_script():
    """Test that our analysis script works."""
    print("📈 Testing analysis script...")
    
    if not Path("analyze_results.py").exists():
        print("❌ analyze_results.py not found")
        return False
    
    # Test that we can import the analysis script
    try:
        import analyze_results
        print("✅ Analysis script imports successfully")
        return True
    except Exception as e:
        print(f"❌ Analysis script error: {e}")
        return False

def main():
    """Run quick tests."""
    print("🚀 Quick Test of SWE-bench Evaluation Setup")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_existing_results,
        test_analysis_script
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Setup looks good!")
        print()
        print("📋 To run a full test:")
        print("1. Start service: python app.py")
        print("2. Open: http://localhost:8000/docs")
        print("3. Submit evaluation request")
        print("4. Check results in artifacts/")
    else:
        print("⚠️  Some issues found. Check the errors above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
