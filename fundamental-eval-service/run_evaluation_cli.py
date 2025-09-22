#!/usr/bin/env python3
"""
CLI tool to run SWE-bench evaluations manually.
This replicates the same functionality as the web API but runs directly from command line.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path to import lm_eval modules
sys.path.append(str(Path(__file__).parent.parent))

from lm_eval.tasks.swebench_leader.run_hf_remote_swebench import run_hf_remote_swebench_evaluation
from routes.evaluation import run_local_docker_evaluation


async def run_manual_evaluation(task_id: str, model_name: str, max_iterations: int = 3):
    """
    Run a complete SWE-bench evaluation manually from CLI.
    
    Args:
        task_id: SWE-bench task ID (e.g., "django__django-11299")
        model_name: Model name for inference (e.g., "google/gemma-2-9b-it")
        max_iterations: Maximum iterations for patch generation
    """
    print(f"🚀 Starting manual SWE-bench evaluation...")
    print(f"📋 Task ID: {task_id}")
    print(f"🤖 Model: {model_name}")
    print(f"🔄 Max Iterations: {max_iterations}")
    print(f"⏰ Started at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Create artifacts directory
    artifacts_dir = Path("./artifacts") / f"{task_id}_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Generate patch using Hugging Face remote inference
        print("🧠 Step 1: Generating patch with Hugging Face API...")
        start_time = datetime.now()
        
        patch_result = run_hf_remote_swebench_evaluation(
            task_id=task_id,
            model_name=model_name,
            max_iterations=max_iterations,
            artifact_dir=str(artifacts_dir)
        )
        
        patch_duration = (datetime.now() - start_time).total_seconds()
        
        if not patch_result["success"]:
            print(f"❌ Patch generation failed: {patch_result.get('error', 'Unknown error')}")
            return
        
        patch_content = patch_result.get("patch", "")
        if not patch_content:
            print("❌ No patch generated")
            return
            
        print(f"✅ Patch generated successfully!")
        print(f"📄 Patch size: {len(patch_content)} characters")
        print(f"⏱️  Patch generation time: {patch_duration:.2f} seconds")
        print()
        
        # Step 2: Run Docker evaluation
        print("🐳 Step 2: Running Docker evaluation...")
        start_time = datetime.now()
        
        docker_result = await run_local_docker_evaluation(
            task_id=task_id,
            patch_content=patch_content,
            model_name=model_name
        )
        
        docker_duration = (datetime.now() - start_time).total_seconds()
        
        print(f"⏱️  Docker evaluation time: {docker_duration:.2f} seconds")
        
        if "error" in docker_result:
            print(f"❌ Docker evaluation failed: {docker_result['error']}")
            print(f"📋 Docker result: {json.dumps(docker_result, indent=2)}")
        else:
            print(f"✅ Docker evaluation completed!")
            solve_rate = docker_result.get("solve_rate", 0.0)
            num_solved = docker_result.get("num_solved", 0)
            total_instances = docker_result.get("total_instances", 1)
            
            print(f"📊 Results:")
            print(f"   Solve Rate: {solve_rate:.2%}")
            print(f"   Solved: {num_solved}/{total_instances}")
            print(f"   Evaluation Method: {docker_result.get('evaluation_method', 'unknown')}")
        
        print()
        
        # Step 3: Save complete results
        print("💾 Step 3: Saving results...")
        
        complete_result = {
            "task_id": task_id,
            "model_name": model_name,
            "max_iterations": max_iterations,
            "patch_generation": {
                "success": patch_result["success"],
                "duration": patch_duration,
                "patch_size": len(patch_content),
                "patch_content": patch_content
            },
            "docker_evaluation": docker_result,
            "total_duration": patch_duration + docker_duration,
            "timestamp": datetime.now().isoformat(),
            "artifacts_dir": str(artifacts_dir)
        }
        
        results_file = artifacts_dir / "complete_results.json"
        with open(results_file, 'w') as f:
            json.dump(complete_result, f, indent=2)
        
        print(f"✅ Complete results saved to: {results_file}")
        print()
        
        # Step 4: Show summary
        print("🎯 EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Task: {task_id}")
        print(f"Model: {model_name}")
        print(f"Total Duration: {complete_result['total_duration']:.2f} seconds")
        print(f"Patch Generated: {patch_result['success']}")
        print(f"Docker Evaluation: {'Success' if 'error' not in docker_result else 'Failed'}")
        
        if 'error' not in docker_result:
            solve_rate = docker_result.get("solve_rate", 0.0)
            print(f"Solve Rate: {solve_rate:.2%}")
        
        print(f"Results Directory: {artifacts_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Evaluation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main CLI function"""
    if len(sys.argv) < 3:
        print("Usage: python run_evaluation_cli.py <task_id> <model_name> [max_iterations]")
        print()
        print("Examples:")
        print("  python run_evaluation_cli.py django__django-11299 google/gemma-2-9b-it")
        print("  python run_evaluation_cli.py sympy__sympy-11618 google/gemma-2-9b-it 5")
        print()
        print("Available tasks (examples):")
        print("  - django__django-11299")
        print("  - django__django-15987") 
        print("  - sympy__sympy-11618")
        print("  - sympy__sympy-12096")
        print()
        print("Available models:")
        print("  - google/gemma-2-9b-it")
        print("  - microsoft/DialoGPT-medium")
        print("  - huggingface/CodeBERTa-small-v1")
        sys.exit(1)
    
    task_id = sys.argv[1]
    model_name = sys.argv[2]
    max_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    
    # Run the evaluation
    asyncio.run(run_manual_evaluation(task_id, model_name, max_iterations))


if __name__ == "__main__":
    main()
