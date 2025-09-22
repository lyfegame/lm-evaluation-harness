"""
Configuration settings for the Fundamental Eval Service
"""
import os
from typing import Optional


class Settings:
    # Application settings
    app_name: str = "Fundamental Eval Service"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API settings
    api_prefix: str = "/api/v1"
    cors_origins: list = ["*"]
    
    # SWE-bench settings
    swebench_dataset: str = "princeton-nlp/SWE-bench_Verified"
    default_model: str = "google/gemma-2-9b-it"
    max_iterations: int = 3
    max_workers: int = 1
    
    # Storage settings
    artifacts_dir: str = "/app/artifacts"
    max_artifact_age_days: int = 7
    
    # Job queue settings
    redis_url: Optional[str] = None
    celery_broker_url: Optional[str] = None
    celery_result_backend: Optional[str] = None
    
    # Render-specific settings
    render_external_url: Optional[str] = None
    port: int = 8000
    
    # Monitoring
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    def __init__(self):
        # Render-specific configuration
        if os.getenv("RENDER"):
            # Running on Render
            self.redis_url = os.getenv("REDIS_URL")
            self.celery_broker_url = self.redis_url
            self.celery_result_backend = self.redis_url
            self.render_external_url = os.getenv("RENDER_EXTERNAL_URL")
            self.debug = False
            self.log_level = "INFO"
            self.artifacts_dir = "/app/artifacts"
        else:
            # Local development
            self.debug = True
            self.log_level = "DEBUG"
            self.artifacts_dir = "./artifacts"


# Global settings instance
settings = Settings()