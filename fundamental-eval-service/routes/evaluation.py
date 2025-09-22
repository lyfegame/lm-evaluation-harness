"""
SWE-bench evaluation endpoints
"""
import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

from config import settings

router = APIRouter()


class EvaluationRequest(BaseModel):
    """Request model for SWE-bench evaluation"""
    task_id: str
    model_name: Optional[str] = settings.default_model
    max_iterations: Optional[int] = settings.max_iterations
    max_workers: Optional[int] = settings.max_workers


class EvaluationResponse(BaseModel):
    """Response model for evaluation results"""
    task_id: str
    model_name: str
    status: str
    duration: float
    patch_generated: bool
    patch_size: int
    solve_rate: float
    artifacts_dir: str
    results_json: Optional[str] = None
    error: Optional[str] = None


# In-memory job tracking (for simple deployment)
active_jobs: Dict[str, Dict[str, Any]] = {}
completed_jobs: Dict[str, Dict[str, Any]] = {}


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_swebench_task(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
):
    """
    Run SWE-bench evaluation for a specific task
    """
    job_id = f"{request.task_id}_{request.model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Check if job is already running
    if job_id in active_jobs:
        raise HTTPException(status_code=409, detail="Evaluation already in progress")
    
    try:
        # Start evaluation in background
        background_tasks.add_task(
            run_evaluation_task,
            job_id,
            request.task_id,
            request.model_name,
            request.max_iterations,
            request.max_workers
        )
        
        # Mark job as active
        active_jobs[job_id] = {
            "task_id": request.task_id,
            "model_name": request.model_name,
            "status": "running",
            "started_at": datetime.utcnow().isoformat()
        }
        
        return EvaluationResponse(
            task_id=request.task_id,
            model_name=request.model_name,
            status="started",
            duration=0.0,
            patch_generated=False,
            patch_size=0,
            solve_rate=0.0,
            artifacts_dir=f"Job started with ID: {job_id}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start evaluation: {str(e)}")


@router.get("/evaluate/{task_id}")
async def get_evaluation_status(task_id: str):
    """
    Get the status of a specific evaluation task
    """
    # Look for job in active or completed jobs
    job = None
    for job_id, job_data in {**active_jobs, **completed_jobs}.items():
        if job_data.get("task_id") == task_id:
            job = job_data
            job["job_id"] = job_id
            break
    
    if not job:
        raise HTTPException(status_code=404, detail="Evaluation task not found")
    
    return job


@router.get("/evaluate/{task_id}/results")
async def get_evaluation_results(task_id: str):
    """
    Get detailed results for a completed evaluation
    """
    # Find the job
    job = None
    job_id = None
    for jid, job_data in completed_jobs.items():
        if job_data.get("task_id") == task_id:
            job = job_data
            job_id = jid
            break
    
    if not job:
        raise HTTPException(status_code=404, detail="Evaluation results not found")
    
    if job.get("status") != "completed":
        raise HTTPException(status_code=202, detail="Evaluation still in progress")
    
    # Load results from artifacts directory
    artifacts_dir = job.get("artifacts_dir")
    if not artifacts_dir or not Path(artifacts_dir).exists():
        raise HTTPException(status_code=404, detail="Artifacts directory not found")
    
    results = {}
    
    # Load results.json if it exists
    results_json_path = Path(artifacts_dir) / "results.json"
    if results_json_path.exists():
        with open(results_json_path, 'r') as f:
            results["evaluation_results"] = json.load(f)
    
    # Load predictions.jsonl if it exists
    predictions_path = Path(artifacts_dir) / "predictions.jsonl"
    if predictions_path.exists():
        with open(predictions_path, 'r') as f:
            results["predictions"] = [json.loads(line) for line in f]
    
    # Load solution.patch if it exists
    solution_path = Path(artifacts_dir) / "django__django-11299" / "solution.patch"
    if solution_path.exists():
        with open(solution_path, 'r') as f:
            results["solution_patch"] = f.read()
    
    return {
        "job_id": job_id,
        "task_id": task_id,
        "artifacts_dir": artifacts_dir,
        **results
    }


@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status: running, completed, failed"),
    limit: int = Query(10, description="Maximum number of jobs to return")
):
    """
    List all evaluation jobs
    """
    all_jobs = {**active_jobs, **completed_jobs}
    
    if status:
        filtered_jobs = {
            job_id: job_data for job_id, job_data in all_jobs.items()
            if job_data.get("status") == status
        }
    else:
        filtered_jobs = all_jobs
    
    # Sort by started_at (most recent first)
    sorted_jobs = sorted(
        filtered_jobs.items(),
        key=lambda x: x[1].get("started_at", ""),
        reverse=True
    )
    
    return {
        "jobs": dict(sorted_jobs[:limit]),
        "total": len(filtered_jobs),
        "active": len(active_jobs),
        "completed": len(completed_jobs)
    }


async def run_evaluation_task(
    job_id: str,
    task_id: str,
    model_name: str,
    max_iterations: int,
    max_workers: int
):
    """
    Background task to run SWE-bench evaluation
    """
    try:
        # Import here to avoid circular imports
        import sys
        from pathlib import Path
        
        # Add the parent directory to path to import lm_eval modules
        parent_dir = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(parent_dir))
        
        from lm_eval.tasks.swebench_leader.run_hf_remote_swebench import run_hf_remote_swebench_evaluation
        from lm_eval.tasks.swebench_leader.task import run_task
        
        # Create timestamped artifact directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_short = task_id.replace("__", "_").replace("/", "_")
        artifact_dir = Path(settings.artifacts_dir) / f"{task_short}_{model_name.replace('/', '_')}_{timestamp}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Update job status
        active_jobs[job_id]["status"] = "evaluating"
        active_jobs[job_id]["artifacts_dir"] = str(artifact_dir)
        
        # Generate patch using Hugging Face remote inference
        from lm_eval.tasks.swebench_leader.run_hf_remote_swebench import run_hf_remote_swebench_evaluation
        
        # Step 1: Generate patch (remote inference)
        patch_result = run_hf_remote_swebench_evaluation(
            task_id=task_id,
            model_name=model_name,
            max_iterations=max_iterations,
            artifact_dir=str(artifact_dir)
        )
        
        if not patch_result["success"]:
            raise Exception(f"Patch generation failed: {patch_result.get('error', 'Unknown error')}")
        
        # Step 2: Run local Docker evaluation
        patch_content = patch_result.get("patch", "")
        if not patch_content:
            raise Exception("No patch generated")
        
        # Run local Docker evaluation
        docker_result = await run_local_docker_evaluation(
            task_id=task_id,
            patch_content=patch_content,
            model_name=model_name
        )
        
        result = {
            "success": True,
            "patch": patch_content,
            "duration": patch_result.get("duration", 0.0),
            "docker_result": docker_result
        }
        
        # The run_task function handles the complete evaluation pipeline
        # including patch generation and Docker-based evaluation
        # Load results from the generated results.json
        results_json_path = artifact_dir / "results.json"
        results_data = {}
        if results_json_path.exists():
            with open(results_json_path, 'r') as f:
                results_data = json.load(f)
        
        # Move job to completed
        completed_jobs[job_id] = {
            "task_id": task_id,
            "model_name": model_name,
            "status": "completed",
            "started_at": active_jobs[job_id]["started_at"],
            "artifacts_dir": str(artifact_dir),
            "completed_at": datetime.now().isoformat(),
            "duration": result.get("duration", 0.0),
            "patch_generated": bool(result.get("patch")),
            "patch_size": len(result.get("patch", "")),
            "solve_rate": results_data.get("solve_rate", 0.0),
            "results_json": str(results_json_path) if results_json_path.exists() else None,
            "job_id": job_id,
            "evaluation_method": "docker_real"  # Mark as real evaluation
        }
        
        # Remove from active jobs
        del active_jobs[job_id]
        
    except Exception as e:
        # Mark as failed
        if job_id in active_jobs:
            completed_jobs[job_id] = {
                **active_jobs[job_id],
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "error": str(e),
                "evaluation_method": "docker_failed"
            }
            del active_jobs[job_id]


async def run_local_docker_evaluation(task_id: str, patch_content: str, model_name: str) -> Dict[str, Any]:
    """
    Run lightweight Docker evaluation locally.
    Results are stored externally for analysis.
    """
    try:
        # Create results directory
        results_dir = Path(settings.artifacts_dir) / "local_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create task-specific directory
        task_dir = results_dir / f"{task_id}_{model_name.replace('/', '_')}"
        task_dir.mkdir(exist_ok=True)
        
        # Set environment variables for Docker container
        env_vars = {
            "TASK_ID": task_id,
            "PATCH_CONTENT": patch_content,
            "MODEL_NAME": model_name
        }
        
        # Run Docker container with docker compose
        cmd = [
            "docker", "compose", "run", "--rm",
            "-e", f"TASK_ID={task_id}",
            "-e", f"PATCH_CONTENT={patch_content}",
            "-e", f"MODEL_NAME={model_name}",
            "swebench-evaluator"
        ]
        
        logger.info(f"Running Docker evaluation: {' '.join(cmd)}")
        
        # Run the Docker container
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,  # Run from service root
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        # Check for results
        summary_file = task_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "error": "No summary file generated",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        return {"error": "Docker evaluation timeout after 30 minutes"}
    except Exception as e:
        return {"error": f"Docker evaluation failed: {str(e)}"}
