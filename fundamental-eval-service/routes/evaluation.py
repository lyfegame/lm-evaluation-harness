"""
SWE-bench evaluation endpoints
"""
import asyncio
import json
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
        from lm_eval.tasks.swebench_leader.local_evaluator import LocalSWEBenchEvaluator
        
        # Create timestamped artifact directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_short = task_id.replace("__", "_").replace("/", "_")
        artifact_dir = Path(settings.artifacts_dir) / f"{task_short}_{model_name.replace('/', '_')}_{timestamp}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Update job status
        active_jobs[job_id]["status"] = "evaluating"
        active_jobs[job_id]["artifacts_dir"] = str(artifact_dir)
        
        # Run the evaluation
        result = run_hf_remote_swebench_evaluation(
            task_id=task_id,
            model_name=model_name,
            max_iterations=max_iterations,
            artifact_dir=str(artifact_dir)
        )
        
        if result["success"]:
            # Run local evaluation
            evaluator = LocalSWEBenchEvaluator(str(artifact_dir))
            predictions_path = result["predictions_file"]
            results_data = evaluator.run_evaluation(predictions_path, settings.swebench_dataset)
            
            # Save results.json
            results_json = artifact_dir / "results.json"
            with open(results_json, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            # Move job to completed
            completed_jobs[job_id] = {
                **active_jobs[job_id],
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "duration": result.get("duration", 0),
                "patch_generated": bool(result.get("patch")),
                "patch_size": len(result.get("patch", "")),
                "solve_rate": results_data.get("solve_rate", 0.0),
                "results_json": str(results_json)
            }
        else:
            # Mark as failed
            completed_jobs[job_id] = {
                **active_jobs[job_id],
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error": result.get("error", "Unknown error")
            }
        
        # Remove from active jobs
        del active_jobs[job_id]
        
    except Exception as e:
        # Mark as failed
        if job_id in active_jobs:
            completed_jobs[job_id] = {
                **active_jobs[job_id],
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error": str(e)
            }
            del active_jobs[job_id]
