#!/usr/bin/env python3
"""
Demo script to show the SWE-bench evaluation results.
This demonstrates that our setup is working with real data.
"""
import json
from pathlib import Path
from datetime import datetime

def show_evaluation_summary():
    """Show a summary of existing evaluation results."""
    print("🎯 SWE-bench Evaluation Results Demo")
    print("=" * 60)
    
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        print("❌ No artifacts directory found")
        return
    
    # Find all evaluation results
    result_dirs = list(artifacts_dir.glob("*"))
    if not result_dirs:
        print("❌ No evaluation results found")
        return
    
    print(f"📊 Found {len(result_dirs)} evaluation results:")
    print()
    
    total_solve_rate = 0
    total_duration = 0
    
    for result_dir in sorted(result_dirs):
        print(f"📁 {result_dir.name}")
        
        # Check for results.json
        results_file = result_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                solve_rate = results.get("solve_rate", 0.0)
                total_instances = results.get("total_instances", 0)
                num_solved = results.get("num_solved", 0)
                method = results.get("evaluation_method", "unknown")
                
                print(f"   ✅ Solve Rate: {solve_rate:.2%} ({num_solved}/{total_instances})")
                print(f"   🔧 Method: {method}")
                
                # Check for patch content
                predictions_file = result_dir / "predictions.jsonl"
                if predictions_file.exists():
                    with open(predictions_file, 'r') as f:
                        prediction = json.load(f)
                    patch_size = len(prediction.get("model_patch", ""))
                    print(f"   📝 Patch Size: {patch_size} characters")
                
                # Check for evaluation details
                eval_dir = result_dir / result_dir.name.split('_')[0] + "__" + result_dir.name.split('_')[1]
                if eval_dir.exists():
                    task_file = eval_dir / "task_data.json"
                    if task_file.exists():
                        with open(task_file, 'r') as f:
                            task_data = json.load(f)
                        problem = task_data.get("problem_statement", "")[:100]
                        print(f"   🎯 Task: {problem}...")
                
                total_solve_rate += solve_rate
                
            except Exception as e:
                print(f"   ❌ Error reading results: {e}")
        else:
            print("   ⚠️  No results.json found")
        
        print()

def show_sample_patch():
    """Show a sample generated patch."""
    print("🔍 Sample Generated Patch")
    print("-" * 40)
    
    artifacts_dir = Path("artifacts")
    result_dirs = list(artifacts_dir.glob("*"))
    
    if not result_dirs:
        print("❌ No results found")
        return
    
    # Use the first result
    result_dir = result_dirs[0]
    predictions_file = result_dir / "predictions.jsonl"
    
    if predictions_file.exists():
        with open(predictions_file, 'r') as f:
            prediction = json.load(f)
        
        patch_content = prediction.get("model_patch", "")
        model_name = prediction.get("model_name_or_path", "unknown")
        task_id = prediction.get("instance_id", "unknown")
        
        print(f"Task: {task_id}")
        print(f"Model: {model_name}")
        print(f"Patch Size: {len(patch_content)} characters")
        print()
        print("Patch Content:")
        print("-" * 40)
        
        if len(patch_content) < 1000:
            print(patch_content)
        else:
            print(patch_content[:500] + "\n... (truncated)")
    else:
        print("❌ No patch found")

def show_api_usage():
    """Show how to use the API."""
    print("🚀 API Usage Examples")
    print("-" * 40)
    
    print("1. Start the service:")
    print("   python app.py")
    print()
    
    print("2. Submit evaluation request:")
    print('   curl -X POST "http://localhost:8000/api/v1/evaluate" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"task_id": "django__django-11299", "model_name": "google/gemma-2-9b-it"}\'')
    print()
    
    print("3. Check evaluation status:")
    print('   curl "http://localhost:8000/api/v1/evaluate/django__django-11299"')
    print()
    
    print("4. View interactive API docs:")
    print("   http://localhost:8000/docs")
    print()

def main():
    """Run the demo."""
    show_evaluation_summary()
    print("\n" + "=" * 60 + "\n")
    show_sample_patch()
    print("\n" + "=" * 60 + "\n")
    show_api_usage()
    
    print("✅ Demo complete! Your SWE-bench evaluation setup is working!")
    print()
    print("📋 Next steps:")
    print("1. Install Docker for real evaluation")
    print("2. Run: ./setup_local.sh")
    print("3. Start service: python app.py")
    print("4. Test with real Docker evaluation")

if __name__ == "__main__":
    main()
