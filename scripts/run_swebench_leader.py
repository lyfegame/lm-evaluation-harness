#!/usr/bin/env python3
"""
Direct runner script for SWE-bench leader agent task.
This provides an alternative to using the harness runner.
"""
import json
import sys
from pathlib import Path

# Add the project root to the path so we can import lm_eval modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lm_eval.tasks.swebench_leader.task import run_task

def main():
    cfg = {
        "task_id": "django__django-11299",
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "model_endpoint": "https://fairies--deploy-checkpoint-9c7df3-9c7d-modelserver-generate.modal.run",
        "model_name": "gemma-3-27b-it@modal",
        "strategy": "systematic",
        "truncation_strategy": "ast_llm_compaction",
        "max_tokens": 16000,
        "max_workers": 8,
        "artifact_dir": "./artifacts/swebench_leader",
        "run_id": "leader_agent_proto"
    }
    
    print("Running SWE-bench leader agent task...")
    print(f"Configuration: {json.dumps(cfg, indent=2)}")
    print()
    
    try:
        result = run_task(cfg)
        print("Task completed successfully!")
        print("Results:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Task failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
