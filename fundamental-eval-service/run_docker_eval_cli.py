#!/usr/bin/env python3
"""
Simple CLI tool to run just the Docker evaluation part.
Use this to test Docker evaluation with a known patch.
"""

import asyncio
import json
import sys
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
    """Main CLI function"""
    if len(sys.argv) < 3:
        print("Usage: python run_docker_eval_cli.py <task_id> <model_name> [patch_file]")
        print()
        print("Examples:")
        print("  python run_docker_eval_cli.py django__django-11299 google/gemma-2-9b-it")
        print("  python run_docker_eval_cli.py django__django-11299 google/gemma-2-9b-it my_patch.diff")
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
    patch_content = None
    
    # Load patch from file if provided
    if len(sys.argv) > 3:
        patch_file = sys.argv[3]
        try:
            with open(patch_file, 'r') as f:
                patch_content = f.read()
            print(f"📄 Loaded patch from file: {patch_file}")
        except FileNotFoundError:
            print(f"❌ Patch file not found: {patch_file}")
            sys.exit(1)
    
    # Run the Docker evaluation
    asyncio.run(run_docker_evaluation_only(task_id, model_name, patch_content))


if __name__ == "__main__":
    main()
