#!/usr/bin/env python3
"""
Hugging Face runner for SWE-bench tasks.
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

from lm_eval.tasks.swebench_leader.hf_leader_agent import HuggingFaceLeaderAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_swebench_task(task_id: str):
    """Load SWE-bench task data."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        
        # Find the task
        for item in dataset:
            if item["instance_id"] == task_id:
                return item
        
        raise ValueError(f"Task {task_id} not found in dataset")
    except Exception as e:
        logger.error(f"Error loading SWE-bench task: {e}")
        # Return a mock task for testing
        return {
            "instance_id": task_id,
            "repo": "django/django",
            "base_commit": "6866c91b638de5368c18713fa851bfe56253ea55",
            "patch": "",
            "test_patch": "",
            "problem_statement": "Test problem statement",
            "FAIL_TO_PASS": "[]",
            "PASS_TO_PASS": "[]"
        }

def create_swebench_prompt(task_data):
    """Create a proper SWE-bench prompt from task data."""
    prompt = f"""You are working on a SWE-bench task. Here are the details:

Repository: {task_data['repo']}
Instance ID: {task_data['instance_id']}
Base Commit: {task_data['base_commit']}

Problem Statement:
{task_data['problem_statement']}

Your task is to generate a unified diff patch that fixes the described bug. The patch should be in standard diff format and should address the issue described in the problem statement.

Generate only the patch content, starting with "--- a/" and ending with the last line of the patch."""
    
    return prompt

def run_hf_swebench_evaluation(task_id: str, model_name: str, max_iterations: int = 3, device: str = "auto"):
    """Run a Hugging Face SWE-bench evaluation."""
    
    logger.info(f"Starting Hugging Face SWE-bench evaluation for task: {task_id}")
    logger.info(f"Model: {model_name}")
    
    # Step 1: Load the SWE-bench task
    logger.info("Step 1: Loading SWE-bench task data...")
    task_data = load_swebench_task(task_id)
    
    # Step 2: Create output directory
    output_dir = Path("artifacts/swebench_leader/hf")
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
        
        # Step 5: Initialize HuggingFaceLeaderAgent
        logger.info("Step 3: Initializing HuggingFaceLeaderAgent...")
        agent = HuggingFaceLeaderAgent(
            model_name=model_name,
            working_dir=str(working_dir),
            max_iterations=max_iterations,
            device=device
        )
        
        # Step 6: Generate patch
        logger.info("Step 4: Generating patch...")
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
                "task_data": task_data,
                "duration": duration,
                "agent_result": result,
                "prediction": prediction,
                "timestamp": time.time()
            }, f, indent=2)
        
        logger.info(f"✅ Hugging Face SWE-bench evaluation completed!")
        logger.info(f"Task: {task_id}")
        logger.info(f"Repository: {task_data['repo']}")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Patch generated: {len(patch_content)} characters")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Predictions saved to: {predictions_file}")
        
        return {
            "status": "success",
            "task_id": task_id,
            "repository": task_data['repo'],
            "duration": duration,
            "patch_length": len(patch_content),
            "results_file": str(results_file),
            "predictions_file": str(predictions_file)
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Hugging Face SWE-bench evaluation")
    parser.add_argument("--task-id", required=True, help="SWE-bench task ID")
    parser.add_argument("--model-name", required=True, help="Hugging Face model name")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum iterations")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--strategy", default="systematic", help="Strategy (ignored)")
    parser.add_argument("--truncation-strategy", default="ast_llm_compaction", help="Truncation strategy (ignored)")
    parser.add_argument("--max-tokens", type=int, default=16000, help="Max tokens (ignored)")
    parser.add_argument("--output_dir_name", default="harness", help="Output dir name (ignored)")
    parser.add_argument("--leader-agent-path", help="Leader agent path (ignored)")
    
    args = parser.parse_args()
    
    result = run_hf_swebench_evaluation(
        task_id=args.task_id,
        model_name=args.model_name,
        max_iterations=args.max_iterations,
        device=args.device
    )
    
    print(f"\n=== Hugging Face SWE-bench Evaluation Results ===")
    print(json.dumps(result, indent=2))
    
    if result["status"] == "error":
        sys.exit(1)

if __name__ == "__main__":
    main()
