#!/usr/bin/env python3
"""
Lightweight SWE-bench evaluation script for local Docker execution.
Stores results externally for analysis.
"""
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

def run_lightweight_evaluation(task_id: str, patch_content: str, model_name: str):
    """
    Run lightweight SWE-bench evaluation with minimal caching.
    Results are stored externally for analysis.
    """
    print(f"🚀 Starting lightweight evaluation for task: {task_id}")
    print(f"📝 Model: {model_name}")
    print(f"⏱️  Started at: {datetime.now().isoformat()}")
    
    # Create results directory
    results_dir = Path("/app/results")
    results_dir.mkdir(exist_ok=True)
    
    # Create task-specific directory
    task_dir = results_dir / f"{task_id}_{model_name.replace('/', '_')}"
    task_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Create predictions.jsonl
        print("📄 Creating predictions file...")
        predictions_file = task_dir / "predictions.jsonl"
        
        prediction = {
            "instance_id": task_id,
            "model_name_or_path": model_name,
            "model_patch": patch_content
        }
        
        with open(predictions_file, 'w') as f:
            f.write(json.dumps(prediction) + "\n")
        
        # Step 2: Run SWE-bench evaluation with minimal caching
        print("🐳 Running SWE-bench evaluation in Docker...")
        start_time = time.time()
        
        # Generate a consistent run_id
        run_id = f"local_{int(time.time())}"
        
        eval_cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", "princeton-nlp/SWE-bench_Verified",
            "--predictions_path", str(predictions_file),
            "--report_dir", str(task_dir),  # Store reports in task directory
            "--max_workers", "1",
            "--cache_level", "base",  # Minimal caching
            "--run_id", run_id,
            "--timeout", "300"  # 5 minute timeout per instance
        ]
        
        # Run evaluation
        result = subprocess.run(
            eval_cmd, 
            cwd=str(task_dir),
            capture_output=True, 
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        duration = time.time() - start_time
        print(f"⏱️  Evaluation completed in {duration:.2f} seconds")
        
        # Step 3: Process results - look for SWE-bench output files
        # SWE-bench creates reports in the report_dir with specific naming
        report_files = list(task_dir.glob(f"*{run_id}*.json"))
        
        if report_files:
            # Use the first report file found
            results_file = report_files[0]
            print(f"📊 Found results file: {results_file}")
            
            with open(results_file, 'r') as f:
                results_data = json.load(f)
            
            # Extract key metrics from SWE-bench results
            # SWE-bench results have different structure
            solve_rate = 0.0
            num_solved = 0
            total_instances = 1
            
            # Try to extract metrics from the results
            if isinstance(results_data, dict):
                if "results" in results_data:
                    results_list = results_data["results"]
                    if isinstance(results_list, list) and len(results_list) > 0:
                        result_item = results_list[0]
                        if "passed" in result_item:
                            num_solved = 1 if result_item["passed"] else 0
                            solve_rate = num_solved / total_instances
            
            print(f"✅ Evaluation Results:")
            print(f"   Solve Rate: {solve_rate:.2%}")
            print(f"   Solved: {num_solved}/{total_instances}")
            
            # Save summary for easy analysis
            summary = {
                "task_id": task_id,
                "model_name": model_name,
                "solve_rate": solve_rate,
                "num_solved": num_solved,
                "total_instances": total_instances,
                "duration_seconds": duration,
                "evaluation_method": "docker_lightweight",
                "timestamp": datetime.now().isoformat(),
                "results_file": str(results_file),
                "predictions_file": str(predictions_file),
                "raw_results": results_data  # Include full results for debugging
            }
            
            summary_file = task_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📊 Results saved to: {task_dir}")
            return summary
            
        else:
            print("❌ No results file found")
            return {"error": "No results file generated"}
            
    except subprocess.TimeoutExpired:
        print("⏰ Evaluation timed out after 30 minutes")
        return {"error": "Evaluation timeout"}
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        error_summary = {
            "task_id": task_id,
            "model_name": model_name,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "evaluation_method": "docker_lightweight_failed"
        }
        
        error_file = task_dir / "error.json"
        with open(error_file, 'w') as f:
            json.dump(error_summary, f, indent=2)
        
        return error_summary

if __name__ == "__main__":
    # Get parameters from environment variables or command line
    task_id = os.getenv("TASK_ID", "django__django-11299")
    patch_content = os.getenv("PATCH_CONTENT", "")
    model_name = os.getenv("MODEL_NAME", "google/gemma-2-9b-it")
    
    if not patch_content:
        print("❌ No patch content provided")
        sys.exit(1)
    
    # Run evaluation
    result = run_lightweight_evaluation(task_id, patch_content, model_name)
    
    # Print final result
    print(f"\n🎯 Final Result: {json.dumps(result, indent=2)}")
