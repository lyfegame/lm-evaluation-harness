"""
FastAPI HTTP wrapper for LM Evaluation Harness
Deployed on Render.com as a production evaluation service
"""

import os
import json
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Import lm_eval
import lm_eval
from lm_eval import evaluator
from lm_eval.models.openai_completions import OpenAICompletionsAPI

app = FastAPI(
    title="LM Evaluation Harness API",
    description="Production-ready LM Evaluation Harness service for evaluating language models",
    version="0.4.9.1"
)

# Request/Response models
class EvaluationRequest(BaseModel):
    model_endpoint: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API endpoint"
    )
    model_name: str = Field(
        default="gpt-3.5-turbo",
        description="Model name to evaluate"
    )
    model_source: Optional[str] = Field(
        default=None,
        description="Original model source (e.g., 'Qwen/Qwen2.5-Coder-14B-Instruct') for tokenizer mapping"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for model endpoint"
    )
    tasks: List[str] = Field(
        default=["hellaswag"],
        description="List of evaluation tasks to run"
    )
    limit: int = Field(
        default=5,
        description="Number of samples to evaluate per task"
    )
    batch_size: Optional[int] = Field(
        default=1,
        description="Batch size for evaluation"
    )

class EvaluationResponse(BaseModel):
    success: bool
    model_name: str
    model_endpoint: str
    tasks: List[str]
    samples_evaluated: int
    results: Dict[str, Any]
    timestamp: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
    available_tasks: List[str]

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="lm-evaluation-harness",
        version="0.4.9.1",
        timestamp=datetime.utcnow().isoformat(),
        available_tasks=[
            "hellaswag", "arc_easy", "arc_challenge", "truthfulqa", 
            "mmlu", "gsm8k", "winogrande", "piqa", "boolq"
        ]
    )

@app.post("/evaluate", response_model=EvaluationResponse)
def run_evaluation(request: EvaluationRequest):
    """Run evaluation on specified tasks"""
    try:
        # Validate inputs
        if not request.model_endpoint:
            raise HTTPException(status_code=400, detail="model_endpoint is required")
        
        if not request.model_name:
            raise HTTPException(status_code=400, detail="model_name is required")

        # Initialize model
        # For non-OpenAI models, use a default tokenizer to avoid mapping errors
        model_for_api = request.model_name
        
        # Check if this is a known OpenAI model or a custom model
        is_openai_model = any(model_for_api.startswith(prefix) for prefix in [
            'gpt-', 'text-', 'davinci', 'curie', 'babbage', 'ada'
        ])
        
        # For custom models (like Qwen), use gpt-3.5-turbo as the tokenizer fallback
        if not is_openai_model:
            # Use gpt-3.5-turbo tokenizer as a reasonable default for custom models
            model_for_api = "gpt-3.5-turbo"
            print(f"Using gpt-3.5-turbo tokenizer for custom model: {request.model_name}")
        
        model_args = {
            "base_url": request.model_endpoint,
            "model": model_for_api,
            "tokenized_requests": False
        }
        
        # Add API key if provided
        if request.api_key:
            model_args["api_key"] = request.api_key
        elif os.environ.get("OPENAI_API_KEY"):
            model_args["api_key"] = os.environ.get("OPENAI_API_KEY")

        # Create model instance
        model = OpenAICompletionsAPI(**model_args)

        # Run evaluation
        results = evaluator.simple_evaluate(
            model=model,
            tasks=request.tasks,
            limit=request.limit,
            batch_size=request.batch_size or 1,
            log_samples=True,
            write_out=False,
            verbosity="INFO"
        )

        # Calculate total samples evaluated
        total_samples = 0
        for task_name, task_result in results["results"].items():
            if isinstance(task_result, dict) and "samples" in task_result:
                total_samples += len(task_result["samples"])

        return EvaluationResponse(
            success=True,
            model_name=request.model_name,
            model_endpoint=request.model_endpoint,
            tasks=request.tasks,
            samples_evaluated=total_samples or request.limit * len(request.tasks),
            results=results["results"],
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        error_msg = str(e)
        print(f"Evaluation error: {error_msg}")
        
        return EvaluationResponse(
            success=False,
            model_name=request.model_name,
            model_endpoint=request.model_endpoint,
            tasks=request.tasks,
            samples_evaluated=0,
            results={},
            timestamp=datetime.utcnow().isoformat(),
            error=error_msg
        )

@app.post("/hellaswag", response_model=EvaluationResponse)
def run_hellaswag_evaluation(request: EvaluationRequest):
    """Convenience endpoint for HellaSwag evaluation"""
    request.tasks = ["hellaswag"]
    return run_evaluation(request)

@app.post("/mmlu", response_model=EvaluationResponse)
def run_mmlu_evaluation(request: EvaluationRequest):
    """Convenience endpoint for MMLU evaluation"""
    request.tasks = ["mmlu"]
    return run_evaluation(request)

@app.get("/tasks")
def list_available_tasks():
    """List all available evaluation tasks"""
    try:
        # Import task registry
        from lm_eval.tasks import TaskManager
        task_manager = TaskManager()
        available_tasks = list(task_manager.all_tasks.keys())
        
        return {
            "success": True,
            "total_tasks": len(available_tasks),
            "tasks": sorted(available_tasks),
            "popular_tasks": [
                "hellaswag", "arc_easy", "arc_challenge", "truthfulqa",
                "mmlu", "gsm8k", "winogrande", "piqa", "boolq"
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "fallback_tasks": [
                "hellaswag", "arc_easy", "arc_challenge", "truthfulqa",
                "mmlu", "gsm8k", "winogrande", "piqa", "boolq"
            ]
        }

if __name__ == "__main__":
    # Use environment variable for production deployment or default based on environment
    is_production = os.environ.get("RENDER", False) or os.environ.get("PRODUCTION", False)
    default_port = 8000 if is_production else 4000
    port = int(os.environ.get("PORT", default_port))
    
    print(f"Starting LM Evaluation Harness API on port {port}")
    print(f"Environment: {'Production' if is_production else 'Development/Local'}")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=not is_production  # Disable auto-reload in production
    )