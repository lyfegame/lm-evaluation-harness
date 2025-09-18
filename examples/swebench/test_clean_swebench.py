#!/usr/bin/env python3
"""
Test script for the clean SWE-bench implementation.
"""
import sys
import json
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lm_eval.tasks.swebench_leader.task import run_task

def test_clean_swebench():
    """Test the clean SWE-bench implementation."""
    
    # Test configuration
    cfg = {
        "task_id": "django__django-11299",
        "model_endpoint": "http://localhost:8000/v1/chat/completions",  # Update this to your actual endpoint
        "model_name": "clean_leader_agent",
        "strategy": "systematic",
        "truncation_strategy": "ast_llm_compaction",
        "max_tokens": 16000,
        "max_workers": 8,
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "artifact_dir": "./artifacts/swebench_leader/clean_test",
        "run_id": "clean_test"
    }
    
    print("Testing clean SWE-bench implementation...")
    print(f"Configuration: {json.dumps(cfg, indent=2)}")
    
    try:
        # Run the task
        result = run_task(cfg)
        
        print(f"\n=== Test Results ===")
        print(f"Status: {'SUCCESS' if result.get('results_json') else 'NO_RESULTS'}")
        print(f"Task ID: {result.get('task_id')}")
        print(f"Model: {result.get('model_name')}")
        print(f"Patch generated: {result.get('patch_generated', False)}")
        print(f"Patch size: {result.get('patch_size', 0)} characters")
        print(f"Artifacts dir: {result.get('artifacts_dir')}")
        print(f"Results JSON: {result.get('results_json')}")
        
        if result.get('results_json'):
            # Check if results.json exists and has the expected format
            results_path = Path(result['results_json'])
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results_data = json.load(f)
                
                print(f"\n=== Results.json Content ===")
                print(f"Total instances: {results_data.get('total_instances', 'N/A')}")
                print(f"Num solved: {results_data.get('num_solved', 'N/A')}")
                print(f"Solve rate: {results_data.get('solve_rate', 'N/A')}")
                
                if 'instances' in results_data:
                    for instance in results_data['instances']:
                        print(f"Instance {instance.get('instance_id')}: {instance.get('status')} (tests_passed: {instance.get('tests_passed', 'N/A')})")
            else:
                print("❌ results.json file not found")
        else:
            print("❌ No results.json generated")
        
        # Check if predictions.jsonl exists
        predictions_path = Path(result.get('predictions_path', ''))
        if predictions_path.exists():
            print(f"\n✅ predictions.jsonl generated: {predictions_path}")
            with open(predictions_path, 'r') as f:
                prediction = json.loads(f.read().strip())
            print(f"Prediction: {prediction['instance_id']} -> {len(prediction['model_patch'])} chars")
        else:
            print("❌ predictions.jsonl not generated")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_clean_swebench()
    
    if result and result.get('results_json'):
        print("\n✅ Clean SWE-bench implementation test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Clean SWE-bench implementation test FAILED")
        sys.exit(1)
