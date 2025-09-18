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
    cfg keys expected:
      - task_id (e.g., "django__django-11299")
      - model_endpoint (e.g., "http://localhost:8000/v1/chat/completions" or Modal URL)
      - model_name (optional string for tagging results)
      - strategy (default: "systematic")
      - truncation_strategy (default: "ast_llm_compaction")
      - max_tokens (default: 16000)
      - max_workers (default: 8)        # for evaluator
      - dataset_name (default: "princeton-nlp/SWE-bench_Verified")
      - artifact_dir (default: "./artifacts/swebench_leader")
      - run_id (default: "leader_agent_proto")
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
    art_dir   = Path(cfg.get("artifact_dir", "./artifacts/swebench_leader")).resolve()

    _ensure_dir(art_dir)
    # Per-instance working directory (to match runner's behavior)
    inst_dir = art_dir / task_id.replace("/", "_")
    _ensure_dir(inst_dir)

    # 1) Run the generator (our copied script)
    runner_path = Path(__file__).parent / "run_leader_agent_swebench.py"
    leader_agent_path = Path(__file__).parent  # Point to the directory containing leader_agent.py
    
    # Check if we should use pre-built image
    use_prebuilt = cfg.get("use_prebuilt_image", False)
    docker_image = "swebench-leader:latest" if use_prebuilt else None
    
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
    
    # Add custom Docker image if using pre-built
    if docker_image:
        cmd.extend(["--docker-image", docker_image])
    
    # Run from artifacts dir so outputs land under there
    subprocess.run(cmd, check=True, cwd=art_dir)

    # 2) Collect the patch
    patch_path = inst_dir / "solution.patch"
    predictions_path = art_dir / "predictions.jsonl"
    record = {
        "instance_id": task_id,
        "model_name_or_path": model_nm,
        "model_patch": patch_path.read_text() if patch_path.exists() else ""
    }
    _safe_jsonl_write(predictions_path, [record])

    # 3) Invoke SWE-bench harness evaluator
    # Writes results.json + per-instance logs in CWD; copy into artifacts.
    eval_cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dname,
        "--predictions_path", str(predictions_path),
        "--max_workers", str(max_w),
        "--run_id", run_id
    ]
    subprocess.run(eval_cmd, check=True, cwd=art_dir)

    # 4) Persist results.json into artifacts if created in CWD or art_dir
    # (Most installs put results.json in current working dir, which we set to art_dir)
    results_json = art_dir / "results.json"
    summary = {
        "predictions_path": str(predictions_path),
        "results_json": str(results_json) if results_json.exists() else None,
        "artifacts_dir": str(art_dir),
        "task_id": task_id,
        "model_name": model_nm,
        "run_id": run_id
    }
    return summary
