#!/usr/bin/env python3
"""
Main script to run SWE-bench evaluation with Hugging Face models.
"""
import sys
import json
import argparse
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lm_eval.tasks.swebench_leader.task import run_task

def main():
    """Main function to run SWE-bench evaluation with Hugging Face models."""
    parser = argparse.ArgumentParser(description="Run SWE-bench evaluation with Hugging Face models")
    parser.add_argument("--task-id", required=True, help="SWE-bench task ID (e.g., django__django-11299)")
    parser.add_argument("--model-name", required=True, help="Hugging Face model name (e.g., microsoft/DialoGPT-medium)")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum iterations for patch generation")
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum workers for evaluation")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--artifact-dir", default="./artifacts/swebench_leader", help="Artifact directory")
    parser.add_argument("--run-id", default="hf_run", help="Run ID for this evaluation")
    
    args = parser.parse_args()
    
    # Configuration
    cfg = {
        "task_id": args.task_id,
        "model_endpoint": f"hf:{args.model_name}",  # Special format for HF models
        "model_name": args.model_name,
        "strategy": "systematic",
        "truncation_strategy": "ast_llm_compaction",
        "max_tokens": 16000,
        "max_workers": args.max_workers,
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "artifact_dir": args.artifact_dir,
        "run_id": args.run_id
    }
    
    print("🤗 Starting Hugging Face SWE-bench Evaluation")
    print("=" * 50)
    print(f"Task ID: {cfg['task_id']}")
    print(f"Model: {cfg['model_name']}")
    print(f"Device: {args.device}")
    print(f"Max Iterations: {args.max_iterations}")
    print(f"Artifact Dir: {cfg['artifact_dir']}")
    print(f"Run ID: {cfg['run_id']}")
    print("=" * 50)
    
    try:
        # Run the task
        result = run_task(cfg)
        
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
        predictions_path = Path(result.get('predictions_path', ''))
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
        
        return result
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    
    if result and result.get('results_json'):
        print("\n🎉 Hugging Face SWE-bench evaluation completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Hugging Face SWE-bench evaluation failed!")
        sys.exit(1)
