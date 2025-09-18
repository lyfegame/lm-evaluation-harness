#!/usr/bin/env python3
"""
Main script to run SWE-bench evaluation with remote inference.
Supports both Hugging Face remote inference and API endpoints.
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lm_eval.tasks.swebench_leader.run_hf_remote_swebench import run_hf_remote_swebench_evaluation
from lm_eval.tasks.swebench_leader.local_evaluator import LocalSWEBenchEvaluator

def create_run_summary(cfg, result, timestamp):
    """Create/update a summary file for boss presentation."""
    summary_file = Path("artifacts") / "execution_summary.json"
    
    # Load existing summary or create new one
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
    else:
        summary = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "runs": []
        }
    
    # Add this run to summary
    run_info = {
        "timestamp": timestamp,
        "task_id": cfg["task_id"],
        "model_name": cfg["model_name"],
        "status": "success" if result.get('results_json') and result.get('patch_generated') else "failed",
        "duration": result.get('duration', 0),
        "patch_generated": result.get('patch_generated', False),
        "patch_size": result.get('patch_size', 0),
        "artifact_dir": cfg["artifact_dir"],
        "solve_rate": 0.0
    }
    
    # Get solve rate from results if available
    if result.get('results_json'):
        results_path = Path(result['results_json'])
        if results_path.exists():
            with open(results_path, 'r') as f:
                results_data = json.load(f)
                run_info["solve_rate"] = results_data.get('solve_rate', 0.0)
    
    summary["runs"].append(run_info)
    summary["total_runs"] += 1
    
    if run_info["status"] == "success":
        summary["successful_runs"] += 1
    else:
        summary["failed_runs"] += 1
    
    # Save updated summary
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Run summary updated: {summary_file}")
    print(f"   Total runs: {summary['total_runs']}")
    print(f"   Successful: {summary['successful_runs']}")
    print(f"   Failed: {summary['failed_runs']}")
    print(f"   Success rate: {summary['successful_runs']/summary['total_runs']*100:.1f}%")

def main():
    """Main function to run SWE-bench evaluation with Hugging Face models."""
    parser = argparse.ArgumentParser(description="Run SWE-bench evaluation with remote inference")
    parser.add_argument("--task-id", required=True, help="SWE-bench task ID (e.g., django__django-11299)")
    parser.add_argument("--model-name", help="Hugging Face model name (e.g., google/gemma-3-12b-it)")
    parser.add_argument("--model-endpoint", help="API endpoint (e.g., https://api.anthropic.com/v1/messages)")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum iterations for patch generation")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum workers for evaluation")
    parser.add_argument("--artifact-dir", default="./artifacts/swebench_leader", help="Artifact directory")
    parser.add_argument("--run-id", default="hf_run", help="Run ID for this evaluation")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_name and not args.model_endpoint:
        print("Error: Either --model-name or --model-endpoint must be provided")
        sys.exit(1)
    
    if args.model_name and args.model_endpoint:
        print("Error: Provide either --model-name OR --model-endpoint, not both")
        sys.exit(1)
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_short = args.task_id.replace("__", "_").replace("/", "_")
    
    # Configuration
    if args.model_name:
        # Hugging Face remote inference
        model_endpoint = f"hf_remote:{args.model_name}"
        model_name = args.model_name
        run_id = f"hf_remote_{timestamp}"
    else:
        # API endpoint
        model_endpoint = args.model_endpoint
        model_name = "api_leader_agent"
        run_id = f"api_{timestamp}"
    
    # Create timestamped artifact directory
    timestamped_artifact_dir = Path(args.artifact_dir) / f"{task_short}_{model_name.replace('/', '_')}_{timestamp}"
    
    cfg = {
        "task_id": args.task_id,
        "model_endpoint": model_endpoint,
        "model_name": model_name,
        "strategy": "systematic",
        "truncation_strategy": "ast_llm_compaction",
        "max_tokens": 16000,
        "max_workers": args.max_workers,
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "artifact_dir": str(timestamped_artifact_dir),
        "run_id": run_id
    }
    
    print("🚀 Starting SWE-bench Remote Inference Evaluation")
    print("=" * 50)
    print(f"Task ID: {cfg['task_id']}")
    print(f"Model: {cfg['model_name']}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Timestamped Artifact Dir: {cfg['artifact_dir']}")
    print(f"Run ID: {cfg['run_id']}")
    print(f"Timestamp: {timestamp}")
    print("=" * 50)
    
    try:
        # Run the evaluation directly
        if args.model_name:
            # Hugging Face remote inference
            result = run_hf_remote_swebench_evaluation(
                task_id=args.task_id,
                model_name=args.model_name,
                max_iterations=args.max_iterations,
                artifact_dir=cfg["artifact_dir"]
            )
            
            # Run local evaluation
            if result["success"]:
                evaluator = LocalSWEBenchEvaluator(cfg["artifact_dir"])
                predictions_path = result["predictions_file"]
                results_data = evaluator.run_evaluation(predictions_path, cfg["dataset_name"])
                
                # Save results.json
                results_json = Path(cfg["artifact_dir"]) / "results.json"
                with open(results_json, 'w') as f:
                    json.dump(results_data, f, indent=2)
                
                result["results_json"] = str(results_json)
                result["patch_generated"] = bool(result.get("patch"))
                result["patch_size"] = len(result.get("patch", ""))
                result["artifacts_dir"] = cfg["artifact_dir"]
        else:
            # API endpoint - not implemented yet
            print("❌ API endpoint support not implemented in simplified version")
            return None
        
        print(f"\n📊 Evaluation Results")
        print("=" * 50)
        print(f"Status: {'✅ SUCCESS' if result.get('results_json') else '❌ NO_RESULTS'}")
        print(f"Task ID: {result.get('task_id')}")
        print(f"Model: {result.get('model_name')}")
        print(f"Patch Generated: {'✅' if result.get('patch_generated') else '❌'}")
        print(f"Patch Size: {result.get('patch_size', 0)} characters")
        print(f"Artifacts Dir: {result.get('artifacts_dir')}")
        print(f"Results JSON: {result.get('results_json')}")
        
        if result.get('results_json'):
            # Check results.json content
            results_path = Path(result['results_json'])
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results_data = json.load(f)
                
                print(f"\n📈 Success Metrics")
                print("=" * 50)
                print(f"Total Instances: {results_data.get('total_instances', 'N/A')}")
                print(f"Num Solved: {results_data.get('num_solved', 'N/A')}")
                print(f"Solve Rate: {results_data.get('solve_rate', 'N/A')}")
                
                if 'instances' in results_data:
                    print(f"\n📋 Instance Results")
                    print("=" * 50)
                    for instance in results_data['instances']:
                        status_emoji = "✅" if instance.get('status') == 'solved' else "❌"
                        print(f"{status_emoji} {instance.get('instance_id')}: {instance.get('status')} (tests_passed: {instance.get('tests_passed', 'N/A')})")
            else:
                print("❌ results.json file not found")
        
        # Check predictions.jsonl
        predictions_path = Path(result.get('predictions_file', ''))
        if predictions_path.exists():
            print(f"\n📝 Predictions")
            print("=" * 50)
            print(f"✅ predictions.jsonl generated: {predictions_path}")
            with open(predictions_path, 'r') as f:
                prediction = json.loads(f.read().strip())
            print(f"Instance ID: {prediction['instance_id']}")
            print(f"Model: {prediction['model_name_or_path']}")
            print(f"Patch Length: {len(prediction['model_patch'])} characters")
        else:
            print("❌ predictions.jsonl not generated")
        
        print(f"\n🎯 Summary")
        print("=" * 50)
        if result.get('results_json') and result.get('patch_generated'):
            print("✅ IDEAL OUTPUT ACHIEVED!")
            print("   - solution.patch exists")
            print("   - predictions.jsonl exists")
            print("   - results.json exists")
            print("   - SWE-bench evaluation completed")
        else:
            print("⚠️  Partial success - check individual components")
        
        # Create/update summary file for boss presentation
        create_run_summary(cfg, result, timestamp)
        
        return result
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    
    if result and result.get('results_json'):
        print("\n🎉 SWE-bench remote inference evaluation completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 SWE-bench remote inference evaluation failed!")
        sys.exit(1)
