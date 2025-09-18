#!/usr/bin/env python3
"""
Hugging Face Remote Inference runner for SWE-bench tasks.
"""
import os
import sys
import json
import time
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lm_eval.tasks.swebench_leader.hf_remote_agent import HuggingFaceRemoteAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_swebench_task(task_id: str):
    """Load SWE-bench task data from the official dataset."""
    from datasets import load_dataset
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    
    # Find the task
    for item in dataset:
        if item["instance_id"] == task_id:
            return item
    
    raise ValueError(f"Task {task_id} not found in SWE-bench dataset. Please verify the task ID is correct.")

def create_swebench_prompt(task_data):
    """Create a proper SWE-bench prompt from task data."""
    prompt = f"""Repository: {task_data.get('repo', 'unknown')}
Base Commit: {task_data.get('base_commit', 'unknown')}

Problem Statement:
{task_data.get('problem_statement', 'No problem statement available')}

Test Cases:
FAIL_TO_PASS: {task_data.get('FAIL_TO_PASS', '[]')}
PASS_TO_PASS: {task_data.get('PASS_TO_PASS', '[]')}

Please generate a patch to fix this issue.
"""
    
    return prompt

def run_hf_remote_swebench_evaluation(task_id: str, model_name: str, max_iterations: int = 3, artifact_dir: str = None):
    """
    Run SWE-bench evaluation using Hugging Face remote inference.
    """
    
    logger.info(f"Starting Hugging Face Remote SWE-bench evaluation for task: {task_id}")
    logger.info(f"Model: {model_name}")
    
    # Step 1: Load the SWE-bench task
    logger.info("Step 1: Loading SWE-bench task data...")
    task_data = load_swebench_task(task_id)
    
    # Step 2: Create output directory (use timestamped path if provided)
    if artifact_dir:
        output_dir = Path(artifact_dir)
    else:
        output_dir = Path("artifacts/swebench_leader/hf_remote")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    working_dir = output_dir / task_id
    working_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 3: Save task data
    task_file = working_dir / "task_data.json"
    with open(task_file, 'w') as f:
        json.dump(task_data, f, indent=2)
    
    try:
        # Step 4: Create SWE-bench prompt
        logger.info("Step 2: Creating SWE-bench prompt...")
        prompt = create_swebench_prompt(task_data)
        
        # Save the prompt
        prompt_file = working_dir / "swebench_prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(prompt)
        
        # Step 5: Initialize HuggingFaceRemoteAgent
        logger.info("Step 3: Initializing HuggingFaceRemoteAgent...")
        agent = HuggingFaceRemoteAgent(
            model_name=model_name,
            working_dir=str(working_dir),
            max_iterations=max_iterations
        )
        
        # Step 6: Generate patch
        logger.info("Step 4: Generating patch using remote inference...")
        start_time = time.time()
        
        result = agent.solve_task(prompt)
        
        duration = time.time() - start_time
        logger.info(f"Patch generation completed in {duration:.2f} seconds")
        
        # Step 7: Create SWE-bench format predictions.jsonl
        logger.info("Step 5: Creating SWE-bench predictions...")
        predictions_file = output_dir / "predictions.jsonl"
        
        # Extract patch from the result
        patch_content = result.get("patch", "")
        
        # Create SWE-bench format prediction
        prediction = {
            "instance_id": task_id,
            "model_name_or_path": model_name,
            "model_patch": patch_content
        }
        
        with open(predictions_file, 'w') as f:
            f.write(json.dumps(prediction) + "\n")
        
        # Step 8: Save comprehensive results
        results_file = working_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "task_id": task_id,
                "model_name": model_name,
                "duration": duration,
                "patch_generated": bool(patch_content),
                "patch_size": len(patch_content),
                "api_endpoint": result.get("api_endpoint"),
                "conversation_file": result.get("conversation_file"),
                "patch_file": result.get("patch_file"),
                "log_file": result.get("log_file"),
                "predictions_file": str(predictions_file),
                "evaluation_method": "hf_remote_inference"
            }, f, indent=2)
        
        logger.info(f"Remote inference evaluation completed successfully!")
        logger.info(f"Patch generated: {bool(patch_content)}")
        logger.info(f"Patch size: {len(patch_content)} characters")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Results saved to: {working_dir}")
        
        return {
            "task_id": task_id,
            "model_name": model_name,
            "patch": patch_content,
            "duration": duration,
            "working_dir": str(working_dir),
            "predictions_file": str(predictions_file),
            "results_file": str(results_file),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Remote inference evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error results
        error_file = working_dir / "error_log.json"
        with open(error_file, 'w') as f:
            json.dump({
                "task_id": task_id,
                "model_name": model_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "evaluation_method": "hf_remote_inference_failed"
            }, f, indent=2)
        
        return {
            "task_id": task_id,
            "model_name": model_name,
            "error": str(e),
            "working_dir": str(working_dir),
            "error_file": str(error_file),
            "success": False
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SWE-bench evaluation with Hugging Face remote inference")
    parser.add_argument("--task-id", required=True, help="SWE-bench task ID")
    parser.add_argument("--model-name", required=True, help="Hugging Face model name")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum iterations")
    
    args = parser.parse_args()
    
    result = run_hf_remote_swebench_evaluation(
        task_id=args.task_id,
        model_name=args.model_name,
        max_iterations=args.max_iterations
    )
    
    if result["success"]:
        print("✅ Remote inference evaluation completed successfully!")
        sys.exit(0)
    else:
        print("❌ Remote inference evaluation failed!")
        sys.exit(1)
