#!/usr/bin/env python3
"""
Simple CLI tool to run just the Docker evaluation part.
Use this to test Docker evaluation with a known patch.
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path to import lm_eval modules
sys.path.append(str(Path(__file__).parent.parent))

from routes.evaluation import run_local_docker_evaluation


async def run_docker_evaluation_only(task_id: str, model_name: str, patch_content: str = None):
    """
    Run only the Docker evaluation part with a provided patch.
    
    Args:
        task_id: SWE-bench task ID (e.g., "django__django-11299")
        model_name: Model name (e.g., "google/gemma-2-9b-it")
        patch_content: Patch content to evaluate (optional, will use test patch if not provided)
    """
    print(f"🐳 Running Docker evaluation only...")
    print(f"📋 Task ID: {task_id}")
    print(f"🤖 Model: {model_name}")
    print(f"⏰ Started at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Use test patch if none provided
    if not patch_content:
        patch_content = """diff --git a/django/contrib/admin/options.py b/django/contrib/admin/options.py
index 1234567..abcdefg 100644
--- a/django/contrib/admin/options.py
+++ b/django/contrib/admin/options.py
@@ -100,6 +100,7 @@ class ModelAdmin(BaseModelAdmin):
         return self._get_changelist_instance(request)
 
     def changelist_view(self, request, extra_context=None):
+        # Fix for issue 11299
         cl = self.get_changelist(request)
         if not cl.has_view_permission(request):
             raise PermissionDenied"""
        print("📄 Using test patch (no patch provided)")
    else:
        print(f"📄 Using provided patch ({len(patch_content)} characters)")
    
    print()
    
    try:
        # Run Docker evaluation
        print("🐳 Running Docker evaluation...")
        start_time = datetime.now()
        
        docker_result = await run_local_docker_evaluation(
            task_id=task_id,
            patch_content=patch_content,
            model_name=model_name
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"⏱️  Evaluation duration: {duration:.2f} seconds")
        print()
        
        if "error" in docker_result:
            print(f"❌ Docker evaluation failed: {docker_result['error']}")
            print(f"📋 Full result: {json.dumps(docker_result, indent=2)}")
        else:
            print(f"✅ Docker evaluation completed successfully!")
            
            solve_rate = docker_result.get("solve_rate", 0.0)
            num_solved = docker_result.get("num_solved", 0)
            total_instances = docker_result.get("total_instances", 1)
            
            print(f"📊 Results:")
            print(f"   Solve Rate: {solve_rate:.2%}")
            print(f"   Solved: {num_solved}/{total_instances}")
            print(f"   Evaluation Method: {docker_result.get('evaluation_method', 'unknown')}")
            
            # Show where results are stored
            if "results_file" in docker_result:
                print(f"   Results File: {docker_result['results_file']}")
            if "summary_file" in docker_result:
                print(f"   Summary File: {docker_result['summary_file']}")
        
        print()
        print("🎯 DOCKER EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Task: {task_id}")
        print(f"Model: {model_name}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Status: {'Success' if 'error' not in docker_result else 'Failed'}")
        
        if 'error' not in docker_result:
            solve_rate = docker_result.get("solve_rate", 0.0)
            print(f"Solve Rate: {solve_rate:.2%}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Docker evaluation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main CLI function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Run Docker evaluation only (faster for testing Docker setup)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --task-id django__django-11299 --model google/gemma-2-9b-it
  %(prog)s --task-id sympy__sympy-11618 --model google/gemma-2-9b-it --patch-file my_patch.diff
  %(prog)s --task-id django__django-11299 --model microsoft/DialoGPT-medium --verbose
  %(prog)s --task-id django__django-11299 --model google/gemma-2-9b-it --custom-patch "diff --git..."

Available tasks (examples):
  - django__django-11299, django__django-15987
  - sympy__sympy-11618, sympy__sympy-12096
  - astropy__astropy-12907, matplotlib__matplotlib-13989

Available models:
  - google/gemma-2-9b-it (recommended)
  - microsoft/DialoGPT-medium
  - huggingface/CodeBERTa-small-v1

Note: If no patch is provided, a test patch will be used.
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
        help="Model name for evaluation (e.g., google/gemma-2-9b-it)"
    )
    
    # Patch options (mutually exclusive)
    patch_group = parser.add_mutually_exclusive_group()
    patch_group.add_argument(
        "--patch-file",
        help="Path to patch file (.diff or .patch)"
    )
    
    patch_group.add_argument(
        "--custom-patch",
        help="Custom patch content as string"
    )
    
    patch_group.add_argument(
        "--use-test-patch",
        action="store_true",
        help="Use built-in test patch (default if no other patch option)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for Docker evaluation in seconds (default: 300)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.timeout < 30 or args.timeout > 1800:
        print("❌ Error: timeout must be between 30 and 1800 seconds")
        sys.exit(1)
    
    # Determine patch content
    patch_content = None
    
    if args.patch_file:
        try:
            with open(args.patch_file, 'r') as f:
                patch_content = f.read()
            if args.verbose:
                print(f"📄 Loaded patch from file: {args.patch_file} ({len(patch_content)} characters)")
        except FileNotFoundError:
            print(f"❌ Patch file not found: {args.patch_file}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error reading patch file: {e}")
            sys.exit(1)
    
    elif args.custom_patch:
        patch_content = args.custom_patch
        if args.verbose:
            print(f"📄 Using custom patch: {len(patch_content)} characters")
    
    else:
        # Use test patch (default)
        if args.verbose:
            print("📄 Using built-in test patch")
    
    if args.verbose:
        print(f"🔧 Configuration:")
        print(f"   Task ID: {args.task_id}")
        print(f"   Model: {args.model}")
        print(f"   Patch Source: {'File' if args.patch_file else 'Custom' if args.custom_patch else 'Test'}")
        print(f"   Timeout: {args.timeout} seconds")
        print(f"   Dry Run: {args.dry_run}")
        print()
    
    if args.dry_run:
        print("🔍 Dry run mode - would execute:")
        print(f"   Docker evaluation: {args.task_id} with {args.model}")
        print(f"   Patch: {'From file' if args.patch_file else 'Custom' if args.custom_patch else 'Test patch'}")
        print(f"   Timeout: {args.timeout} seconds")
        print("✅ Dry run completed (no actual evaluation performed)")
        return
    
    # Run the Docker evaluation
    asyncio.run(run_docker_evaluation_only(args.task_id, args.model, patch_content))


if __name__ == "__main__":
    main()
