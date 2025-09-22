#!/usr/bin/env python3
"""
CLI tool to run SWE-bench evaluations manually.
This replicates the same functionality as the web API but runs directly from command line.
"""

import asyncio
import json
import sys
import argparse
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
    """Main CLI function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Run complete SWE-bench evaluation (patch generation + Docker evaluation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --task-id django__django-11299 --model google/gemma-2-9b-it
  %(prog)s --task-id sympy__sympy-11618 --model google/gemma-2-9b-it --max-iterations 5
  %(prog)s --task-id django__django-11299 --model microsoft/DialoGPT-medium --verbose

Available tasks (examples):
  - django__django-11299, django__django-15987
  - sympy__sympy-11618, sympy__sympy-12096
  - astropy__astropy-12907, matplotlib__matplotlib-13989

Available models:
  - google/gemma-2-9b-it (recommended)
  - microsoft/DialoGPT-medium
  - huggingface/CodeBERTa-small-v1
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--task-id", 
        required=True,
        help="SWE-bench task ID (e.g., django__django-11299)"
    )
    
    parser.add_argument(
        "--model", 
        required=True,
        help="Hugging Face model name (e.g., google/gemma-2-9b-it)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum iterations for patch generation (default: 3)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Custom output directory for results (default: ./artifacts)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.max_iterations < 1 or args.max_iterations > 10:
        print("❌ Error: max-iterations must be between 1 and 10")
        sys.exit(1)
    
    if args.verbose:
        print(f"🔧 Configuration:")
        print(f"   Task ID: {args.task_id}")
        print(f"   Model: {args.model}")
        print(f"   Max Iterations: {args.max_iterations}")
        print(f"   Output Directory: {args.output_dir or './artifacts'}")
        print(f"   Dry Run: {args.dry_run}")
        print()
    
    if args.dry_run:
        print("🔍 Dry run mode - would execute:")
        print(f"   Patch generation: {args.task_id} with {args.model}")
        print(f"   Docker evaluation: Real SWE-bench harness")
        print(f"   Max iterations: {args.max_iterations}")
        print("✅ Dry run completed (no actual evaluation performed)")
        return
    
    # Run the evaluation
    asyncio.run(run_manual_evaluation(args.task_id, args.model, args.max_iterations))


if __name__ == "__main__":
    main()
