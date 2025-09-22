#!/usr/bin/env python3
"""
Test script to run a complete SWE-bench evaluation.
This tests the evaluation workflow without needing the web service.
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import lm_eval modules
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_patch_generation():
    """Test patch generation using Hugging Face remote inference."""
    print("🤖 Testing patch generation...")
    
    try:
        from lm_eval.tasks.swebench_leader.run_hf_remote_swebench import run_hf_remote_swebench_evaluation
        
        task_id = "django__django-11299"
        model_name = "google/gemma-2-9b-it"
        
        print(f"   Task: {task_id}")
        print(f"   Model: {model_name}")
        
        # Create output directory
        output_dir = Path("test_artifacts")
        output_dir.mkdir(exist_ok=True)
        
        # Run patch generation
        result = run_hf_remote_swebench_evaluation(
            task_id=task_id,
            model_name=model_name,
            max_iterations=3,
            artifact_dir=str(output_dir)
        )
        
        if result.get("success", False):
            patch_content = result.get("patch", "")
            duration = result.get("duration", 0.0)
            
            print(f"   ✅ Patch generated successfully!")
            print(f"   📝 Patch size: {len(patch_content)} characters")
            print(f"   ⏱️  Duration: {duration:.2f} seconds")
            
            # Save patch for inspection
            patch_file = output_dir / "generated_patch.patch"
            with open(patch_file, 'w') as f:
                f.write(patch_content)
            print(f"   💾 Patch saved to: {patch_file}")
            
            return patch_content, duration
        else:
            print(f"   ❌ Patch generation failed: {result.get('error', 'Unknown error')}")
            return None, 0.0
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None, 0.0

def test_local_evaluation(patch_content):
    """Test local evaluation (simplified version)."""
    print("🧪 Testing local evaluation...")
    
    if not patch_content:
        print("   ❌ No patch content to evaluate")
        return None
    
    try:
        from lm_eval.tasks.swebench_leader.local_evaluator import LocalSWEBenchEvaluator
        
        # Create evaluator
        evaluator = LocalSWEBenchEvaluator("test_evaluation")
        
        # Create predictions.jsonl
        predictions_file = Path("test_evaluation/predictions.jsonl")
        predictions_file.parent.mkdir(exist_ok=True)
        
        prediction = {
            "instance_id": "django__django-11299",
            "model_name_or_path": "google/gemma-2-9b-it",
            "model_patch": patch_content
        }
        
        with open(predictions_file, 'w') as f:
            f.write(json.dumps(prediction) + "\n")
        
        # Run evaluation
        results = evaluator.run_evaluation(str(predictions_file))
        
        solve_rate = results.get("solve_rate", 0.0)
        num_solved = results.get("num_solved", 0)
        total_instances = results.get("total_instances", 1)
        
        print(f"   ✅ Evaluation completed!")
        print(f"   📊 Solve Rate: {solve_rate:.2%} ({num_solved}/{total_instances})")
        print(f"   🔧 Method: {results.get('evaluation_method', 'unknown')}")
        
        # Save results
        results_file = Path("test_evaluation/results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   💾 Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"   ❌ Evaluation error: {e}")
        return None

def test_analysis():
    """Test results analysis."""
    print("📊 Testing results analysis...")
    
    try:
        # Check if we have test results
        results_file = Path("test_evaluation/results.json")
        if not results_file.exists():
            print("   ⚠️  No test results found, using existing results")
            return test_existing_analysis()
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        solve_rate = results.get("solve_rate", 0.0)
        method = results.get("evaluation_method", "unknown")
        
        print(f"   ✅ Analysis completed!")
        print(f"   📈 Solve Rate: {solve_rate:.2%}")
        print(f"   🔧 Method: {method}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Analysis error: {e}")
        return False

def test_existing_analysis():
    """Test analysis with existing results."""
    print("   📋 Using existing results for analysis...")
    
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        print("   ❌ No artifacts directory found")
        return False
    
    result_dirs = list(artifacts_dir.glob("*"))
    if not result_dirs:
        print("   ❌ No existing results found")
        return False
    
    # Use the most recent result
    latest_result = max(result_dirs, key=lambda x: x.stat().st_mtime)
    results_file = latest_result / "results.json"
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        solve_rate = results.get("solve_rate", 0.0)
        method = results.get("evaluation_method", "unknown")
        
        print(f"   ✅ Found existing result: {latest_result.name}")
        print(f"   📈 Solve Rate: {solve_rate:.2%}")
        print(f"   🔧 Method: {method}")
        
        return True
    else:
        print("   ❌ No results.json in latest result")
        return False

def main():
    """Run complete evaluation test."""
    print("🚀 SWE-bench Evaluation Test")
    print("=" * 50)
    
    start_time = time.time()
    
    # Test 1: Patch Generation
    print("\n1. Testing Patch Generation")
    print("-" * 30)
    patch_content, patch_duration = test_patch_generation()
    
    # Test 2: Local Evaluation
    print("\n2. Testing Local Evaluation")
    print("-" * 30)
    if patch_content:
        evaluation_results = test_local_evaluation(patch_content)
    else:
        print("   ⚠️  Skipping evaluation (no patch generated)")
        evaluation_results = None
    
    # Test 3: Results Analysis
    print("\n3. Testing Results Analysis")
    print("-" * 30)
    analysis_success = test_analysis()
    
    # Summary
    total_duration = time.time() - start_time
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("-" * 30)
    
    print(f"⏱️  Total Duration: {total_duration:.2f} seconds")
    print(f"🤖 Patch Generation: {'✅ Success' if patch_content else '❌ Failed'}")
    print(f"🧪 Local Evaluation: {'✅ Success' if evaluation_results else '❌ Failed'}")
    print(f"📊 Analysis: {'✅ Success' if analysis_success else '❌ Failed'}")
    
    if patch_content and evaluation_results:
        solve_rate = evaluation_results.get("solve_rate", 0.0)
        print(f"🎯 Final Solve Rate: {solve_rate:.2%}")
        
        if solve_rate > 0:
            print("🎉 Evaluation test PASSED!")
        else:
            print("⚠️  Evaluation completed but no tasks solved")
    else:
        print("❌ Evaluation test FAILED")
    
    print("\n📁 Generated Files:")
    test_files = [
        "test_artifacts/generated_patch.patch",
        "test_evaluation/predictions.jsonl", 
        "test_evaluation/results.json"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (not created)")
    
    return patch_content is not None and evaluation_results is not None

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
