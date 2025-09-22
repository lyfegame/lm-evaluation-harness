#!/usr/bin/env python3
"""
Test script to verify the lightweight Docker evaluation setup.
This tests the setup without requiring Docker to be running.
"""
import os
import sys
import json
from pathlib import Path

def test_file_structure():
    """Test that all required files exist."""
    print("🔍 Testing file structure...")
    
    required_files = [
        "app.py",
        "config.py", 
        "requirements.txt",
        "docker-compose.yml",
        "Dockerfile.evaluator",
        "evaluate_local.py",
        "analyze_results.py",
        "setup_local.sh",
        "README-LOCAL-DOCKER.md",
        "routes/evaluation.py",
        "routes/health.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_directories():
    """Test that required directories exist."""
    print("📁 Testing directory structure...")
    
    required_dirs = [
        "routes",
        "artifacts"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir.mkdir(exist_ok=True)
        print("📁 Created results directory")
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All required directories present")
        return True

def test_config():
    """Test that configuration is valid."""
    print("⚙️  Testing configuration...")
    
    try:
        from config import settings
        print(f"✅ Config loaded: {settings.app_name} v{settings.app_version}")
        print(f"   Artifacts dir: {settings.artifacts_dir}")
        print(f"   Dataset: {settings.swebench_dataset}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_imports():
    """Test that all Python imports work."""
    print("🐍 Testing Python imports...")
    
    try:
        # Test core imports
        import fastapi
        print("✅ FastAPI imported")
        
        import uvicorn
        print("✅ Uvicorn imported")
        
        # Test our modules
        from routes import health, evaluation
        print("✅ Route modules imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_docker_files():
    """Test Docker configuration files."""
    print("🐳 Testing Docker configuration...")
    
    # Check docker-compose.yml
    if Path("docker-compose.yml").exists():
        print("✅ docker-compose.yml exists")
    else:
        print("❌ docker-compose.yml missing")
        return False
    
    # Check Dockerfile.evaluator
    if Path("Dockerfile.evaluator").exists():
        print("✅ Dockerfile.evaluator exists")
    else:
        print("❌ Dockerfile.evaluator missing")
        return False
    
    # Check evaluate_local.py
    if Path("evaluate_local.py").exists():
        print("✅ evaluate_local.py exists")
    else:
        print("❌ evaluate_local.py missing")
        return False
    
    return True

def test_analysis_tools():
    """Test analysis tools."""
    print("📊 Testing analysis tools...")
    
    # Check analyze_results.py
    if Path("analyze_results.py").exists():
        print("✅ analyze_results.py exists")
    else:
        print("❌ analyze_results.py missing")
        return False
    
    # Check if pandas is available (for analysis)
    try:
        import pandas
        print("✅ Pandas available for analysis")
    except ImportError:
        print("⚠️  Pandas not available - analysis features may be limited")
    
    return True

def test_existing_results():
    """Test existing evaluation results."""
    print("📋 Testing existing results...")
    
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        result_dirs = list(artifacts_dir.glob("*"))
        print(f"✅ Found {len(result_dirs)} existing evaluation results")
        
        # Show sample results
        for result_dir in result_dirs[:3]:  # Show first 3
            print(f"   📁 {result_dir.name}")
            
            # Check for results.json
            results_file = result_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    solve_rate = results.get("solve_rate", 0.0)
                    print(f"      Solve rate: {solve_rate:.2%}")
                except Exception as e:
                    print(f"      Error reading results: {e}")
    else:
        print("⚠️  No existing artifacts found")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Testing Lightweight Docker SWE-bench Evaluation Setup")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_directories,
        test_config,
        test_imports,
        test_docker_files,
        test_analysis_tools,
        test_existing_results
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is ready.")
        print()
        print("📋 Next steps:")
        print("1. Install Docker (if not already installed)")
        print("2. Run: ./setup_local.sh")
        print("3. Start service: python app.py")
        print("4. Test API: http://localhost:8000/docs")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
