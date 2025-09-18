#!/usr/bin/env python3
"""
Render background worker for SWE-bench evaluations.
This script handles the main execution logic for Render workers.
"""
import os
import sys
import json
import time
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lm_eval.tasks.swebench_leader.task import run_task

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main worker function."""
    logger.info("Starting SWE-bench LeaderAgent worker on Render...")
    
    # Get configuration from environment variables
    task_id = os.getenv('TASK_ID')
    if not task_id:
        logger.error("TASK_ID environment variable is required")
        sys.exit(1)
    
    max_iterations = int(os.getenv('MAX_ITERATIONS', '10'))
    model_endpoint = os.getenv('MODEL_ENDPOINT')
    if not model_endpoint:
        logger.error("MODEL_ENDPOINT environment variable is required")
        sys.exit(1)
    
    # Build configuration
    cfg = {
        "task_id": task_id,
        "dataset_name": "princeton-nlp/SWE-bench_Verified",
        "model_endpoint": model_endpoint,
        "model_name": os.getenv('MODEL_NAME', 'gemma-3-27b-it@modal'),
        "strategy": os.getenv('STRATEGY', 'systematic'),
        "truncation_strategy": os.getenv('TRUNCATION_STRATEGY', 'ast_llm_compaction'),
        "max_tokens": int(os.getenv('MAX_TOKENS', '16000')),
        "max_workers": int(os.getenv('MAX_WORKERS', '1')),  # Render workers should use 1
        "artifact_dir": os.getenv('ARTIFACT_DIR', '/app/artifacts/swebench_leader'),
        "run_id": os.getenv('RUN_ID', 'render_worker'),
        "use_prebuilt_image": False,  # We'll handle Docker differently on Render
        "max_iterations": max_iterations
    }
    
    logger.info(f"Configuration: {json.dumps(cfg, indent=2)}")
    
    try:
        # Run the task
        logger.info(f"Starting evaluation for task: {task_id}")
        start_time = time.time()
        
        result = run_task(cfg)
        
        duration = time.time() - start_time
        logger.info(f"Task completed in {duration:.2f} seconds")
        logger.info(f"Results: {json.dumps(result, indent=2)}")
        
        # Check if we have results
        if result.get('results_json'):
            logger.info("✅ Evaluation completed successfully with results")
        else:
            logger.warning("⚠️ Evaluation completed but no results generated")
        
        # Output results to stdout for Render logs
        print(f"\n=== SWE-bench Evaluation Results ===")
        print(f"Task ID: {task_id}")
        print(f"Duration: {duration:.2f}s")
        print(f"Status: {'SUCCESS' if result.get('results_json') else 'NO_RESULTS'}")
        print(f"Artifacts: {result.get('artifacts_dir', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Task failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
