import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _safe_jsonl_write(path: Path, records):
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def run_task(cfg: Dict) -> Dict:
    """
    Run a SWE-bench task with proper evaluation.
    
    cfg keys expected:
      - task_id (e.g., "django__django-11299")
      - model_endpoint (e.g., "http://localhost:8000/v1/chat/completions")
      - model_name (optional string for tagging results)
      - strategy (default: "systematic")
      - truncation_strategy (default: "ast_llm_compaction")
      - max_tokens (default: 16000)
      - max_workers (default: 8)
      - dataset_name (default: "princeton-nlp/SWE-bench_Verified")
      - artifact_dir (default: "./artifacts/swebench_leader")
      - run_id (default: "leader_agent_proto")
      - use_docker (default: False) - Set to True to use Docker-based evaluation
    """
    task_id   = cfg["task_id"]
    endpoint  = cfg["model_endpoint"]
    model_nm  = cfg.get("model_name", "leader-agent@proto")
    strategy  = cfg.get("strategy", "systematic")
    trunc     = cfg.get("truncation_strategy", "ast_llm_compaction")
    max_toks  = int(cfg.get("max_tokens", 16000))
    max_w     = int(cfg.get("max_workers", 8))
    dname     = cfg.get("dataset_name", "princeton-nlp/SWE-bench_Verified")
    run_id    = cfg.get("run_id", "leader_agent_proto")
    use_docker = cfg.get("use_docker", False)  # Default to local evaluation
    art_dir   = Path(cfg.get("artifact_dir", "./artifacts/swebench_leader")).resolve()

    _ensure_dir(art_dir)
    # Per-instance working directory
    inst_dir = art_dir / task_id.replace("/", "_")
    _ensure_dir(inst_dir)

    # 1) Run the appropriate LeaderAgent to generate patch
    # Check if this is a Hugging Face model
    if endpoint.startswith("hf:"):
        model_name = endpoint[3:]  # Remove "hf:" prefix
        runner_path = Path(__file__).parent / "run_hf_swebench.py"
    else:
        runner_path = Path(__file__).parent / "run_clean_swebench.py"
    
    leader_agent_path = Path(__file__).parent
    
    if endpoint.startswith("hf:"):
        # For Hugging Face models, use model-name instead of model-endpoint
        cmd = [
            sys.executable, str(runner_path),
            "--task-id", task_id,
            "--model-name", model_name,
            "--strategy", strategy,
            "--truncation-strategy", trunc,
            "--max-tokens", str(max_toks),
            "--output_dir_name", "harness",
            "--leader-agent-path", str(leader_agent_path)
        ]
    else:
        # For API endpoints, use model-endpoint
        cmd = [
            sys.executable, str(runner_path),
            "--task-id", task_id,
            "--strategy", strategy,
            "--model-endpoint", endpoint,
            "--truncation-strategy", trunc,
            "--max-tokens", str(max_toks),
            "--output_dir_name", "harness",
            "--leader-agent-path", str(leader_agent_path)
        ]
    
    # Run from artifacts dir so outputs land under there
    subprocess.run(cmd, check=True, cwd=art_dir)

    # 2) Collect the patch and create predictions.jsonl
    patch_path = inst_dir / "solution.patch"
    predictions_path = art_dir / "predictions.jsonl"
    
    # Read patch content if it exists
    patch_content = ""
    if patch_path.exists():
        patch_content = patch_path.read_text()
    
    record = {
        "instance_id": task_id,
        "model_name_or_path": model_nm,
        "model_patch": patch_content
    }
    _safe_jsonl_write(predictions_path, [record])

    # 3) Run SWE-bench evaluation
    if use_docker:
        # Use Docker-based evaluation (original SWE-bench harness)
        print("Using Docker-based evaluation...")
        eval_cmd = [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", dname,
            "--predictions_path", str(predictions_path),
            "--max_workers", str(max_w),
            "--run_id", run_id
        ]
        
        try:
            subprocess.run(eval_cmd, check=True, cwd=art_dir)
        except subprocess.CalledProcessError as e:
            print(f"Docker evaluation failed: {e}")
            # Create a minimal results.json for failed evaluation
            results_json = art_dir / "results.json"
            with open(results_json, 'w') as f:
                json.dump({
                    "total_instances": 1,
                    "num_solved": 0,
                    "solve_rate": 0.0,
                    "instances": [{
                        "instance_id": task_id,
                        "status": "failed",
                        "tests_passed": False,
                        "error": str(e),
                        "evaluation_method": "docker_failed"
                    }]
                }, f, indent=2)
    else:
        # Use local evaluation (no Docker required)
        print("Using local evaluation (no Docker required)...")
        try:
            from .local_evaluator import LocalSWEBenchEvaluator
            
            evaluator = LocalSWEBenchEvaluator(str(art_dir))
            results_data = evaluator.run_evaluation(str(predictions_path), dname)
            
            # Save results.json
            results_json = art_dir / "results.json"
            with open(results_json, 'w') as f:
                json.dump(results_data, f, indent=2)
                
            print(f"Local evaluation completed: {results_data['num_solved']}/{results_data['total_instances']} solved")
            
        except Exception as e:
            print(f"Local evaluation failed: {e}")
            # Create a minimal results.json for failed evaluation
            results_json = art_dir / "results.json"
            with open(results_json, 'w') as f:
                json.dump({
                    "total_instances": 1,
                    "num_solved": 0,
                    "solve_rate": 0.0,
                    "instances": [{
                        "instance_id": task_id,
                        "status": "failed",
                        "tests_passed": False,
                        "error": str(e),
                        "evaluation_method": "local_fallback"
                    }]
                }, f, indent=2)

    # 4) Check for results.json and return summary
    results_json = art_dir / "results.json"
    summary = {
        "predictions_path": str(predictions_path),
        "results_json": str(results_json) if results_json.exists() else None,
        "artifacts_dir": str(art_dir),
        "task_id": task_id,
        "model_name": model_nm,
        "run_id": run_id,
        "patch_generated": patch_path.exists(),
        "patch_size": len(patch_content) if patch_content else 0
    }
    
    # If results.json exists, add summary metrics
    if results_json.exists():
        try:
            with open(results_json, 'r') as f:
                results_data = json.load(f)
                summary.update({
                    "total_instances": results_data.get("total_instances", 0),
                    "num_solved": results_data.get("num_solved", 0),
                    "solve_rate": results_data.get("solve_rate", 0.0)
                })
        except Exception as e:
            print(f"Error reading results.json: {e}")
    
    return summary