#!/usr/bin/env python3
"""
Run LeaderAgent on SWE-bench Docker tasks

This script provides multiple prompt strategies for fixing bugs in SWE-bench tasks using
the LeaderAgent that maintains its own Jupyter notebook and executes code cells.

Usage:
    python run_leader_agent_swebench.py --task-id django__django-11299 --strategy systematic --model-endpoint http://localhost:8000/chat
    python run_leader_agent_swebench.py --task-id sympy__sympy-20212 --strategy tdd --model-endpoint http://localhost:8000/chat
    python run_leader_agent_swebench.py --task-id all --strategy systematic --model-endpoint dummy --max-workers 4  # Run all tasks in parallel
"""

import os
import json
import docker
import time
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, List
import logging
from datasets import load_dataset
import pwd
import grp
import subprocess
import shutil
import glob
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import fcntl
import signal
import atexit
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] %(message)s')
logger = logging.getLogger(__name__)
TOOL_VERSION=""

# Global variable to track containers for cleanup
_containers_to_cleanup = []

def cleanup_docker_resources():
    """Clean up Docker containers and images to free up disk space"""
    try:
        docker_client = docker.from_env()
        
        # Clean up stopped containers
        stopped_containers = docker_client.containers.list(filters={'status': 'exited'})
        for container in stopped_containers:
            try:
                container.remove()
                logger.info(f"Removed stopped container: {container.id}")
            except Exception as e:
                logger.warning(f"Failed to remove stopped container {container.id}: {e}")
        
        # Clean up dangling images
        dangling_images = docker_client.images.list(filters={'dangling': True})
        for image in dangling_images:
            try:
                docker_client.images.remove(image.id, force=True)
                logger.info(f"Removed dangling image: {image.id}")
            except Exception as e:
                logger.warning(f"Failed to remove dangling image {image.id}: {e}")
        
        # Clean up unused images (older than 1 day)
        import datetime
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=1)
        all_images = docker_client.images.list()
        for image in all_images:
            try:
                # Check if image is older than 1 day and not being used
                if image.attrs.get('Created'):
                    created_time = datetime.datetime.fromisoformat(
                        image.attrs['Created'].replace('Z', '+00:00')
                    )
                    if created_time < cutoff_time:
                        # Check if any containers are using this image
                        containers_using_image = docker_client.containers.list(
                            filters={'ancestor': image.id}
                        )
                        if not containers_using_image:
                            docker_client.images.remove(image.id, force=True)
                            logger.info(f"Removed old unused image: {image.id}")
            except Exception as e:
                logger.warning(f"Failed to remove old image {image.id}: {e}")
                
    except Exception as e:
        logger.error(f"Error during Docker cleanup: {e}")

def signal_handler(signum, frame):
    """Handle interrupt signals to ensure cleanup"""
    logger.info(f"Received signal {signum}, cleaning up...")
    cleanup_docker_resources()
    sys.exit(1)

# Register signal handlers for graceful cleanup
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Register cleanup function to run at exit
atexit.register(cleanup_docker_resources)

def check_disk_space(path="/home/tianhangzhu", min_gb=5):
    """Check available disk space and warn if low"""
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)
        
        logger.info(f"Disk space: {free_gb:.1f}GB free out of {total/(1024**3):.1f}GB total")
        
        if free_gb < min_gb:
            logger.warning(f"Low disk space: {free_gb:.1f}GB free (less than {min_gb}GB)")
            logger.info("Running Docker cleanup to free space...")
            cleanup_docker_resources()
            
            # Check again after cleanup
            total, used, free = shutil.disk_usage(path)
            free_gb = free / (1024**3)
            logger.info(f"After cleanup: {free_gb:.1f}GB free")
            
            if free_gb < min_gb:
                logger.error(f"Still low disk space: {free_gb:.1f}GB free. Consider manual cleanup.")
                return False
        
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Assume OK if we can't check

CLAUDE_CODE_V2_DIR = "/home/tianhangzhu/gcs_view/home/tianhangzhu/verltrain/claude-code_v2"
def extract_code_cells(notebook_path: str) -> List[str]:
    """Extract all code cells from a Jupyter notebook"""
    code_cells = []
    
    try:
        with open(notebook_path, 'r') as f:
            notebook_data = json.load(f)
        
        for cell in notebook_data.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    code = ''.join(source)
                else:
                    code = source
                
                # Skip empty cells
                if code.strip():
                    code_cells.append(code.strip())
        
        logger.info(f"Extracted {len(code_cells)} non-empty code cells from notebook")
        return code_cells
        
    except Exception as e:
        logger.error(f"Error reading notebook: {e}")
        return []

def ensure_ownership(path: Path, user: str = "tianhangzhu", group: str = "tianhangzhu"):
    """Ensure a file or directory is owned by the specified user and group"""
    try:
        # Get user and group IDs
        uid = pwd.getpwnam(user).pw_uid
        gid = grp.getgrnam(group).gr_gid
        
        # Change ownership
        os.chown(path, uid, gid)
        
        # If it's a directory, recursively change ownership of contents
        if path.is_dir():
            for item in path.rglob('*'):
                os.chown(item, uid, gid)
                
        logger.debug(f"Set ownership of {path} to {user}:{group}")
    except Exception as e:
        # If we don't have permissions, try using sudo
        try:
            if path.is_dir():
                subprocess.run(['sudo', 'chown', '-R', f'{user}:{group}', str(path)], check=True, capture_output=True)
            else:
                subprocess.run(['sudo', 'chown', f'{user}:{group}', str(path)], check=True, capture_output=True)
            logger.debug(f"Set ownership of {path} to {user}:{group} using sudo")
        except Exception as sudo_e:
            logger.warning(f"Could not change ownership of {path}: {e}, sudo attempt: {sudo_e}")

class LeaderAgentSWERunner:
    def __init__(self, leader_agent_path: Optional[str] = None, workspace_dir: Optional[str] = None):
        if leader_agent_path is None:
            # Use the current directory as the default
            self.leader_agent_path = Path.cwd()
        else:
            self.leader_agent_path = Path(leader_agent_path)
            
        # Check if leader_agent.py exists
        self.leader_agent_script = self.leader_agent_path / "leader_agent.py"
        if not self.leader_agent_script.exists():
            raise FileNotFoundError(f"LeaderAgent script not found at {self.leader_agent_script}")
            
        self.docker_client = docker.from_env()
        
        # Use provided workspace_dir or default to ./leader_agent_results
        if workspace_dir is None:
            self.workspace_dir = Path("./leader_agent_results").resolve()
        else:
            self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure correct ownership
        ensure_ownership(self.workspace_dir)
    
    def get_swebench_image(self, instance: Dict) -> str:
        """Get the SWE-bench Docker image name using SWE-Agent v2's naming pattern"""
        instance_id = instance['instance_id']
        # Docker doesn't allow double underscore, so replace with _1776_
        id_docker_compatible = instance_id.replace("__", "_1776_")
        # Use the exact pattern from SWE-Agent v2
        image_name = f"swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        return image_name
      
    def get_tool_definitions(self):
        """Extract tool definitions from the first LLM call and convert JS-style values to Python"""
        tool_definitions_path = Path(__file__).parent / "tool_definitions.json"
        with open(tool_definitions_path, 'r') as f:
            tools = json.load(f)
        return self._convert_js_to_python(tools)
    
    def _convert_js_to_python(self, obj):
        """Recursively convert JavaScript-style values to Python equivalents"""
        if isinstance(obj, dict):
            return {key: self._convert_js_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_js_to_python(item) for item in obj]
        else:
            # Handle boolean values - check both actual booleans and string representations
            if obj is True:
                return True
            elif obj is False:
                return False
            elif obj is None:
                return None
            elif isinstance(obj, str):
                obj_lower = obj.lower().strip()
                if obj_lower == "false":
                    return False
                elif obj_lower == "true":
                    return True
                elif obj_lower == "null":
                    return None
            return obj
    
    def _json_dumps_python_style(self, obj):
        """JSON dumps that preserves Python boolean format"""
        json_str = json.dumps(obj, indent=2)
        # Replace JSON booleans with Python booleans
        json_str = json_str.replace('"true"', 'True')
        json_str = json_str.replace('"false"', 'False')
        json_str = json_str.replace('"null"', 'None')
        json_str = json_str.replace('true', 'True')
        json_str = json_str.replace('false', 'False')
        json_str = json_str.replace('null', 'None')
        return json_str

    def get_prompt_strategy(self, strategy_name: str, problem_statement: str) -> str:
        """Get the appropriate prompt based on the selected strategy"""
        
        import json
        tool_def_str = self._json_dumps_python_style(self.get_tool_definitions())
        
        escaped_problem_statement = json.dumps(json.dumps(problem_statement))
        
        first_cell_source = f'''# Problem Statement and Configuration
problem_statement = {escaped_problem_statement}

work_dir = "/testbed"

# Tool Definitions
tool_definitions = {tool_def_str}'''
        # Wrap the code in <code> tags
        wrapped_code = f"<code>{first_cell_source}</code>"
        
        # Return the prompt in the expected format
        return f"<thinking>Please solve a PR request</thinking>{wrapped_code}"
        
    
    def run_leader_agent(self, instance: Dict, model_endpoint: str, prompt_strategy: str = "systematic", truncation_strategy: str = "ast_llm_compaction", max_tokens: int = 13000) -> Dict:
        """Run LeaderAgent in Docker container for SWE-bench task"""
        instance_id = instance['instance_id']
        output_dir = self.workspace_dir / instance_id.replace('/', '_')
        container = None
        
        # Check disk space before starting
        if not check_disk_space():
            return {
                'instance_id': instance_id,
                'status': 'error',
                'error': 'Insufficient disk space',
                'output_dir': str(output_dir)
            }
        
        # Create output directory if it doesn't exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup real filesystem path for Docker (handles GCS FUSE mount)
        real_output_dir = Path(str(output_dir.resolve()).replace('/home/tianhangzhu/gcs_view/home/tianhangzhu/', '/home/tianhangzhu/'))
        real_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get Docker image
        image_name = self.get_swebench_image(instance)
        logger.info(f"Using Docker image: {image_name}")
        logger.info(f"Using prompt strategy: {prompt_strategy}")
        
        # Pull image if needed
        try:
            self.docker_client.images.get(image_name)
            logger.info(f"Image {image_name} already available locally")
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling image {image_name}...")
            try:
                self.docker_client.images.pull(image_name)
            except docker.errors.APIError as e:
                logger.error(f"Failed to pull image {image_name}: {e}")
                return {
                    'instance_id': instance_id,
                    'status': 'error',
                    'error': f'Failed to pull Docker image: {e}'
                }
        
        # Get the appropriate prompt based on strategy
        full_prompt = self.get_prompt_strategy(prompt_strategy, instance['problem_statement'])
        
        # Copy LeaderAgent script to real output directory (for Docker)
        agent_script_copy = real_output_dir / "leader_agent.py"
        shutil.copy2(self.leader_agent_script, agent_script_copy)
        ensure_ownership(agent_script_copy)
        
        # Extract code cells from the notebook if it exists
        # Try both possible locations for notebooks
        notebook_patterns = [
            f"{CLAUDE_CODE_V2_DIR}/noninteractive_results_v2/{instance_id}/ipynbs_fake_replay_real_tool_{TOOL_VERSION}_subleader/*.ipynb",
        ]
        
        notebook_files = []
        for pattern in notebook_patterns:
            files = glob.glob(pattern)
            if files:
                notebook_files.extend(files)
                break  # Use the first pattern that matches
        
        # Look for the main notebook specifically
        main_notebook_path = None
        main_code_cells_json_path = None
        all_code_cells_json_paths = []  # Track all JSON files created
        
        if notebook_files:
            # Find the main notebook
            for notebook_path in notebook_files:
                if "_main_full_live.ipynb" in notebook_path:
                    main_notebook_path = notebook_path
                    logger.info(f"Found main notebook: {main_notebook_path}")
                    break
            
            # Extract code cells from ALL notebooks
            for notebook_path in notebook_files:
                logger.info(f"Extracting code cells from: {notebook_path}")
                
                # Extract code cells
                code_cells = extract_code_cells(notebook_path)
                
                if code_cells:
                    # Save code cells as JSON (replace .ipynb with .json)
                    json_path = notebook_path.replace(".ipynb", ".json")
                    try:
                        with open(json_path, 'w') as f:
                            json.dump(code_cells, f, indent=2)
                        logger.info(f"Saved {len(code_cells)} code cells to {json_path}")
                        
                        # Track all JSON files created
                        all_code_cells_json_paths.append(json_path)
                        
                        # Keep track of the main code cells path
                        if notebook_path == main_notebook_path:
                            main_code_cells_json_path = json_path
                    except Exception as e:
                        logger.error(f"Error saving code cells: {e}")
        else:
            logger.warning(f"No notebook found for task {instance_id}")
        
        # Create starting_prompt.txt file in the real filesystem path for Docker
        starting_prompt_path = real_output_dir / "starting_prompt.txt"
        with open(starting_prompt_path, 'w') as f:
            f.write(full_prompt)
        ensure_ownership(starting_prompt_path)
        logger.info(f"Created starting_prompt.txt with {len(full_prompt)} characters")
        
        # Copy required Python dependencies to output directory
        dependencies = [
            f"{CLAUDE_CODE_V2_DIR}/subleader_solve.py",
            f"{CLAUDE_CODE_V2_DIR}/claude_thinking_python.py",
            f"{CLAUDE_CODE_V2_DIR}/tools.py",
            f"{CLAUDE_CODE_V2_DIR}/notebook_manager.py",
            f"{CLAUDE_CODE_V2_DIR}/compact_truncate.py"
        ]
        
        # Add all code cells JSON files to dependencies
        if all_code_cells_json_paths:
            for json_path in all_code_cells_json_paths:
                if Path(json_path).exists():
                    dependencies.append(str(json_path))
                    logger.info(f"Added notebook JSON to dependencies: {Path(json_path).name}")
        elif main_code_cells_json_path:
            # Fallback to just main if somehow all_code_cells_json_paths is empty
            raise FileNotFoundError(f"Code cells JSON file not found: {main_code_cells_json_path}")
        
        for dep_path in dependencies:
            if Path(dep_path).exists():
                dep_copy = real_output_dir / Path(dep_path).name
                shutil.copy2(dep_path, dep_copy)
                ensure_ownership(dep_copy)
                logger.info(f"Copied dependency: {Path(dep_path).name}")
            else:
                logger.warning(f"Dependency not found: {dep_path}")
        
        # Create script for LeaderAgent execution in container
        script_content = f'''#!/bin/bash
set -e

echo "=== Running LeaderAgent for SWE-bench task ==="
echo "Instance: {instance_id}"
echo "Repository: {instance['repo']}"
echo "Docker Image: {image_name}"
echo "Model Endpoint: {model_endpoint}"
echo ""

# Get the UID and GID from the host user (passed via environment)
HOST_UID=${{HOST_UID:-1000}}
HOST_GID=${{HOST_GID:-1000}}

echo "Using UID: $HOST_UID, GID: $HOST_GID"

        # Check if we're using a pre-built image
        if [ -f "/.prebuilt_image" ]; then
            echo "Using pre-built image - skipping package installation"
            echo "Python location: $(which python)"
            echo "Python version: $(python --version)"
        else
            # Install system dependencies
            echo "Installing system utilities..."
            apt-get update && apt-get install -y git curl wget build-essential python3-dev || echo "Some packages failed to install, continuing..."

            # Install Python packages for the system
            echo "Installing Python packages..."
            pip install dill torch tqdm nbformat requests transformers jupyter nbconvert rpds-py chardet beautifulsoup4 jupyter_client aiohttp psutil || echo "Some packages failed to install, continuing..."

            echo "Python location: $(which python)"
            echo "Python version: $(python --version)"
        fi


# The repository should be at /testbed in SWE-Agent v2 images
cd /testbed || {{ echo "ERROR: Cannot find repository at /testbed"; exit 1; }}

echo "Working directory: $(pwd)"

# Checkout the correct commit
echo "Checking out commit: {instance['base_commit']}"
git checkout {instance['base_commit']} 2>/dev/null || true

# Show current state
echo ""
echo "Git status:"
git status --short

echo ""
echo "Repository structure:"
find . -type f -name "*.py" | head -20

echo ""
echo "=== Running LeaderAgent ==="
echo ""

# Export API key for all processes

# Create a non-root user to run LeaderAgent with the same UID/GID as host user
echo "Creating non-root user for LeaderAgent with UID=$HOST_UID..."
groupadd -g $HOST_GID agentuser || true
useradd -m -u $HOST_UID -g $HOST_GID -s /bin/bash agentuser || true

# Install packages for the new user to ensure access (skip if pre-built image)
if [ -f "/.prebuilt_image" ]; then
    echo "Using pre-built image - skipping user package installation"
else
    echo "Installing packages for agentuser..."
    su - agentuser -c "pip install --user dill torch tqdm nbformat requests transformers jupyter nbconvert rpds-py chardet beautifulsoup4 jupyter_client aiohttp psutil" || true
fi

# Copy LeaderAgent script to user's home directory  
cp /output/leader_agent.py /home/agentuser/leader_agent.py
chown agentuser:agentuser /home/agentuser/leader_agent.py

# Copy Python dependencies to the expected location in container
echo "Copying Python dependencies to expected locations..."
mkdir -p {CLAUDE_CODE_V2_DIR}
cp /output/*.py {CLAUDE_CODE_V2_DIR} 2>/dev/null || true

# Also copy all JSON files
echo "Copying JSON dependencies to expected locations..."
cp /output/*.json {CLAUDE_CODE_V2_DIR} 2>/dev/null || true

# Also copy to home directory for backup
echo "Copying Python dependencies to home directory..."
cp /output/*.py /home/agentuser/ 2>/dev/null || true

# Also copy to home directory for backup
echo "Copying JSON dependencies to home directory..."
cp /output/*.json /home/agentuser/ 2>/dev/null || true

# Set proper permissions
echo "Setting proper permissions..."
chown -R agentuser:agentuser {CLAUDE_CODE_V2_DIR} 2>/dev/null || true
chown agentuser:agentuser /home/agentuser/*.py 2>/dev/null || true
chown agentuser:agentuser /home/agentuser/*.json 2>/dev/null || true

# Give the user ownership of necessary directories
echo "Setting proper permissions for testbed and output directories..."
chown -R agentuser:agentuser /testbed || true
chown -R agentuser:agentuser /output || true

# Set up debug environment variables for more verbose logging
export JUPYTER_ENABLE_UNSAFE_IPV6=1
export JUPYTER_DEBUG=1
export IPYTHON_DEBUG=1
export PYTHONDEVMODE=1

# Run LeaderAgent as non-root user with verbose logging
echo "Running LeaderAgent as non-root user with debug logging..."
'''
        
        # Add the code cells file parameter if model endpoint is "dummy"
        if model_endpoint == "dummy" and main_code_cells_json_path:
            # Get just the filename from the main_code_cells_json_path
            main_code_cells_filename = os.path.basename(main_code_cells_json_path)
            code_cells_param = f" --code-cells-file '/output/{main_code_cells_filename}'"
        else:
            code_cells_param = ""
        
        script_content += f'''timeout 1800 su - agentuser -c "cd /testbed && export PYTHONPATH=/output:/home/agentuser:$PYTHONPATH && export JUPYTER_ENABLE_UNSAFE_IPV6=1 && export JUPYTER_DEBUG=1 && export IPYTHON_DEBUG=1 && export PYTHONDEVMODE=1 && export PYTHONUNBUFFERED=1 && /usr/bin/python /home/agentuser/leader_agent.py --model-endpoint '{model_endpoint}' --prompt-file /output/starting_prompt.txt --working-dir '/testbed' --max-iterations 300 --truncation-strategy '{truncation_strategy}' --max-tokens {max_tokens} --output-conversation '/output/conversation.json'{code_cells_param} 2>&1" | tee /output/leader_agent_output.log || true

# Copy results and ensure proper ownership
cp /home/agentuser/leader_agent_notebook.ipynb /output/ 2>/dev/null || true
chown $HOST_UID:$HOST_GID /output/* 2>/dev/null || true

# Check if any files were modified
echo ""
echo "=== Changes made ==="
su - agentuser -c "cd /testbed && git diff --name-status" || git diff --name-status

# Generate patch if there are changes
if su - agentuser -c "cd /testbed && git diff --quiet" || git diff --quiet; then
    echo "No changes were made"
else
    echo "Generating patch..."
    su - agentuser -c "cd /testbed && git diff > /output/solution.patch" || git diff > /output/solution.patch
    chown $HOST_UID:$HOST_GID /output/solution.patch 2>/dev/null || true
    echo "Patch saved to /output/solution.patch"
    
    # Show the patch
    echo ""
    echo "=== Generated Patch ==="
    cat /output/solution.patch || true
fi

echo ""
echo "=== LeaderAgent Execution Complete ==="
'''
        
        script_path = real_output_dir / "run_leader_agent.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        
        # Save instance info
        with open(real_output_dir / "instance_info.json", 'w') as f:
            json.dump({
                'instance_id': instance_id,
                'repo': instance['repo'],
                'base_commit': instance['base_commit'],
                'docker_image': image_name,
                'model_endpoint': model_endpoint,
                'mode': 'leader-agent',
                'prompt_strategy': prompt_strategy,
                'truncation_strategy': truncation_strategy,
                'max_tokens': max_tokens,
                'problem_statement_length': len(instance['problem_statement'])
            }, f, indent=2)
        
        # Run container
        try:
            logger.info(f"Starting container for {instance_id}...")
            
            # Get current user's UID and GID
            current_uid = os.getuid()
            current_gid = os.getgid()
            
            # Prepare environment variables
            env_vars = {
                'LANG': 'C.UTF-8',
                'LC_ALL': 'C.UTF-8',
                'PYTHONUNBUFFERED': '1',
                'HOST_UID': str(current_uid),
                'HOST_GID': str(current_gid),
            }
            
            # Fix Docker mount path for GCS FUSE
            mount_path = str(output_dir.resolve())
            mount_path = mount_path.replace('/home/tianhangzhu/gcs_view/home/tianhangzhu/', '/home/tianhangzhu/')
            
            container = self.docker_client.containers.run(
                image=image_name,
                command=["/bin/bash", "-c", "cat /output/run_leader_agent.sh | bash"],
                volumes={
                    mount_path: {
                        'bind': '/output',
                        'mode': 'rw'
                    }
                },
                environment=env_vars,
                mem_limit='10g',
                working_dir='/testbed',
                detach=True,
                remove=False
            )
            
            # Monitor execution
            start_time = time.time()
            logger.info("Container started, waiting for completion...")
            
            # Stream logs while running
            for line in container.logs(stream=True):
                print(line.decode('utf-8', errors='replace'), end='')
                sys.stdout.flush()
            
            # Wait for completion
            result = container.wait(timeout=1800)  # 30 minute timeout
            duration = time.time() - start_time
            
            # Get final logs
            logs = container.logs().decode('utf-8', errors='replace')
            
            # Save logs
            with open(real_output_dir / "container.log", 'w') as f:
                f.write(logs)
            
            # Check results
            solution_path = real_output_dir / "solution.patch"
            conversation_path = real_output_dir / "conversation.json"
            
            solution_exists = solution_path.exists()
            conversation_exists = conversation_path.exists()
            
            logger.info(f"Execution completed in {duration:.1f}s")
            logger.info(f"Solution generated: {solution_exists}")
            logger.info(f"Conversation saved: {conversation_exists}")
            
            # Optionally sync results to GCS FUSE mount if different from real path
            if str(output_dir) != str(real_output_dir):
                try:
                    for item in real_output_dir.iterdir():
                        src = item
                        dst = output_dir / item.name
                        if src.is_file():
                            shutil.copy2(src, dst)
                    logger.info(f"Synced results to GCS FUSE mount: {output_dir}")
                except Exception as e:
                    logger.debug(f"Optional sync to GCS: {e}")
            
            return {
                'instance_id': instance_id,
                'status': 'success' if solution_exists else 'no_changes',
                'duration': duration,
                'output_dir': str(output_dir),
                'docker_image': image_name,
                'conversation_exists': conversation_exists
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error(f"Error running container: {e}")
            return {
                'instance_id': instance_id,
                'status': 'error',
                'error': str(e),
                'output_dir': str(output_dir)
            }
        finally:
            # Always clean up the container, regardless of success/failure/interruption
            if container is not None:
                try:
                    logger.info(f"Cleaning up container {container.id} for {instance_id}...")
                    
                    # Stop the container if it's still running
                    try:
                        container.reload()  # Refresh container state
                        if container.status == 'running':
                            logger.info(f"Stopping running container {container.id}...")
                            container.stop(timeout=30)
                    except docker.errors.NotFound:
                        logger.info(f"Container {container.id} already removed")
                    except Exception as stop_error:
                        logger.warning(f"Error stopping container {container.id}: {stop_error}")
                    
                    # Remove the container
                    try:
                        container.remove(force=True)
                        logger.info(f"Successfully removed container {container.id}")
                    except docker.errors.NotFound:
                        logger.info(f"Container {container.id} already removed")
                    except Exception as remove_error:
                        logger.warning(f"Error removing container {container.id}: {remove_error}")
                        
                except Exception as cleanup_error:
                    logger.error(f"Error during container cleanup for {instance_id}: {cleanup_error}")
                    # Try one more time with force removal
                    try:
                        if container:
                            container.remove(force=True)
                    except:
                        pass  # Give up if it still fails

def run_single_instance_worker(args_tuple):
    """Worker function for parallel processing - runs a single instance"""
    instance, model_endpoint, strategy, leader_agent_path, workspace_base, resume, truncation_strategy, max_tokens = args_tuple
    
    instance_id = instance['instance_id']
    workspace_dir = f"{workspace_base}/{instance_id}/leader_agent_results"
    
    # Check if solution already exists - but only skip if resume is True
    solution_path = Path(workspace_dir) / instance_id.replace('/', '_') / "solution.patch"
    if solution_path.exists() and resume:
        logger.info(f"Skipping {instance_id} - solution.patch already exists")
        return {
            'instance_id': instance_id,
            'status': 'skipped',
            'duration': 0,
            'output_dir': workspace_dir
        }
    
    # If not resuming and solution exists, log that we're re-running
    if solution_path.exists() and not resume:
        logger.info(f"Force re-running {instance_id} (existing solution.patch will be overwritten)")
    
    try:
        # Run LeaderAgent
        runner = LeaderAgentSWERunner(
            leader_agent_path=leader_agent_path,
            workspace_dir=workspace_dir
        )
        result = runner.run_leader_agent(
            instance, 
            model_endpoint=model_endpoint,
            prompt_strategy=strategy,
            truncation_strategy=truncation_strategy,
            max_tokens=max_tokens
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {instance_id}: {e}")
        return {
            'instance_id': instance_id,
            'status': 'error',
            'error': str(e),
            'duration': 0,
            'output_dir': workspace_dir
        }

def run_all_instances():
    """Run all SWE-bench tasks with LeaderAgent"""
    
    # Check disk space at startup
    logger.info("Checking disk space before starting...")
    if not check_disk_space():
        logger.error("Insufficient disk space to continue. Please free up space and try again.")
        return 1
    #https://fairies--deploy-checkpoint-65b33b-65b3-modelserver-generate.modal.run
    parser = argparse.ArgumentParser(description='Run LeaderAgent on SWE-bench tasks')
    parser.add_argument('--task-id', type=str, default='sphinx-doc__sphinx-8638',
                        help='SWE-bench task ID to run or "all" to run all tasks (default: django__django-11206)')
    parser.add_argument('--strategy', type=str, default='systematic',
                        choices=['systematic', 'tdd', 'scientific', 'defensive', 'simple'],
                        help='Prompt strategy to use (default: systematic)')
    parser.add_argument('--model-endpoint', type=str, default="https://fairies--deploy-checkpoint-9c7df3-9c7d-modelserver-generate.modal.run",
                        help='Model API endpoint URL (e.g., http://localhost:8000/chat) or "dummy" for dummy model replay')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of instances to process (for testing)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run (skip completed instances)')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of parallel workers (default: CPU count / 2)')
    parser.add_argument('--truncation-strategy', type=str, default='ast_llm_compaction',
                        choices=['ast_llm_compaction', 'none', 'simple'],
                        help='Truncation strategy for model input (default: ast_llm_compaction)')
    parser.add_argument('--max-tokens', type=int, default=16000,
                        help='Maximum tokens for model output (default: 16000)')
    parser.add_argument('--output_dir_name', type=str, default="multiturn900",
                        help='Output directory name (default: test_output)')
    parser.add_argument('--leader-agent-path', type=str, default=None,
                        help='Path to the LeaderAgent script directory')
    
    args = parser.parse_args()
    if args.leader_agent_path is None:
        args.leader_agent_path = f"{CLAUDE_CODE_V2_DIR}"
    # Set max workers default
    if args.max_workers is None:
        args.max_workers = max(1, multiprocessing.cpu_count() - 2)
    
    # Load the dataset
    logger.info(f"Loading SWE-bench Verified dataset...")
    dataset = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    
    # If task-id is not "all", run single task
    if args.task_id != "all":
        # Find the specific instance
        instance = None
        for item in dataset:
            if item.get('instance_id') == args.task_id:
                instance = dict(item)
                break
        
        if not instance:
            logger.error(f"Task {args.task_id} not found in SWE-bench Verified")
            return 1
        
        # Set workspace directory
        args.workspace_dir = f"./noninteractive_results_v2/{args.task_id}/leader_agent_results_{args.output_dir_name}"
        
        # Check if solution already exists
        solution_path = Path(args.workspace_dir) / args.task_id / "solution.patch"
        if solution_path.exists() and args.resume:
            logger.info(f"Skipping {args.task_id} - solution.patch already exists at {solution_path}")
            return 0
        
        # Display task info
        print("\n" + "="*60)
        print(f"Task: {instance['instance_id']}")
        print(f"Repository: {instance['repo']}")
        print(f"Base Commit: {instance['base_commit']}")
        print(f"Problem Statement Length: {len(instance['problem_statement'])} chars")
        print(f"Prompt Strategy: {args.strategy}")
        print(f"Model Endpoint: {args.model_endpoint}")
        print(f"Truncation Strategy: {args.truncation_strategy}")
        print(f"Max Tokens: {args.max_tokens}")
        print("="*60 + "\n")
        
        # Run LeaderAgent
        runner = LeaderAgentSWERunner(
            leader_agent_path=args.leader_agent_path,
            workspace_dir=args.workspace_dir
        )
        result = runner.run_leader_agent(
            instance, 
            model_endpoint=args.model_endpoint,
            prompt_strategy=args.strategy,
            truncation_strategy=args.truncation_strategy,
            max_tokens=args.max_tokens
        )
        
        # Show results
        print("\n" + "="*60)
        print("RESULTS:")
        print(f"Status: {result['status']}")
        print(f"Duration: {result.get('duration', 0):.1f}s")
        print(f"Output Directory: {result['output_dir']}")
        
        if result['status'] == 'success':
            print(f"\n✓ Solution patch generated!")
            print(f"  View patch: {result['output_dir']}/solution.patch")
            print(f"  View logs: {result['output_dir']}/leader_agent_output.log")
            if result.get('notebook_exists'):
                print(f"  View notebook: {result['output_dir']}/leader_agent_notebook.ipynb")
            if result.get('conversation_exists'):
                print(f"  View conversation: {result['output_dir']}/conversation.json")
        elif result['status'] == 'no_changes':
            print(f"\n- No changes were made by LeaderAgent")
        else:
            print(f"\n✗ Error: {result.get('error', 'Unknown error')}")
        
        print("="*60 + "\n")
        
        return 0 if result['status'] in ['success', 'no_changes'] else 1
    
    # Run all tasks in parallel
    else:
        logger.info(f"Running all {len(dataset)} SWE-bench Verified tasks")
        logger.info(f"Using prompt strategy: {args.strategy}")
        logger.info(f"Model endpoint: {args.model_endpoint}")
        logger.info(f"Truncation strategy: {args.truncation_strategy}")
        logger.info(f"Max tokens: {args.max_tokens}")
        logger.info(f"Using {args.max_workers} parallel workers")
        
        # Apply limit if specified
        instances = list(dataset)
        if args.limit:
            instances = instances[:args.limit]
            logger.info(f"Limited to first {args.limit} instances")
        
        # Base workspace directory
        workspace_base = f"./noninteractive_results_v2"
        
        # Filter out already completed instances if resuming
        if args.resume:
            instances_to_run = []
            for instance in instances:
                instance_id = instance['instance_id']
                workspace_dir = f"{workspace_base}/{instance_id}/leader_agent_results"
                solution_path = Path(workspace_dir) / instance_id.replace('/', '_') / "solution.patch"
                if not solution_path.exists():
                    instances_to_run.append(instance)
                else:
                    logger.info(f"Skipping already completed: {instance_id}")
        else:
            instances_to_run = instances
        
        logger.info(f"Found {len(instances_to_run)} instances to process (skipped {len(instances) - len(instances_to_run)} completed)")
        
        # Track results
        completed = []
        skipped = []
        failed = []
        no_changes = []
        all_results = {}
        
        # Prepare arguments for parallel processing
        task_args = [
            (instance, args.model_endpoint, args.strategy, args.leader_agent_path, workspace_base, args.resume, args.truncation_strategy, args.max_tokens)
            for instance in instances_to_run
        ]
        
        # Run in parallel with progress bar
        start_time = time.time()
        
        # Check disk space periodically during execution
        last_disk_check = time.time()
        
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_instance = {
                executor.submit(run_single_instance_worker, task_arg): task_arg[0]
                for task_arg in task_args
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(instances), desc="Processing instances") as pbar:
                # Update progress bar for already skipped instances
                pbar.update(len(instances) - len(instances_to_run))
                
                for future in as_completed(future_to_instance):
                    instance = future_to_instance[future]
                    instance_id = instance['instance_id']
                    
                    try:
                        result = future.result(timeout=1800)  # 30 minute timeout per task
                        all_results[instance_id] = result
                        
                        # Update tracking based on status
                        if result['status'] == 'success':
                            completed.append(instance_id)
                            pbar.set_postfix_str(f"✓ {instance_id}")
                        elif result['status'] == 'skipped':
                            skipped.append(instance_id)
                            pbar.set_postfix_str(f"⏭ {instance_id}")
                        elif result['status'] == 'no_changes':
                            no_changes.append(instance_id)
                            pbar.set_postfix_str(f"- {instance_id}")
                        else:
                            failed.append(instance_id)
                            pbar.set_postfix_str(f"✗ {instance_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to get result for {instance_id}: {e}")
                        failed.append(instance_id)
                        pbar.set_postfix_str(f"✗ {instance_id} - Timeout/Error")
                        all_results[instance_id] = {
                            'instance_id': instance_id,
                            'status': 'error',
                            'error': str(e)
                        }
                    
                    pbar.update(1)
        
        total_duration = time.time() - start_time
        
        # Include pre-skipped instances in the skipped count
        total_skipped = len(skipped) + (len(instances) - len(instances_to_run))
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY:")
        print(f"Total instances: {len(instances)}")
        print(f"Instances processed: {len(instances_to_run)}")
        print(f"Successful patches: {len(completed)} ({len(completed)/len(instances)*100:.1f}%)")
        print(f"No changes made: {len(no_changes)}")
        print(f"Skipped (already done): {total_skipped}")
        print(f"Failed: {len(failed)}")
        print(f"Total time: {total_duration/60:.1f} minutes")
        if instances_to_run:
            print(f"Average time per instance: {total_duration/len(instances_to_run)/60:.1f} minutes")
        print(f"Max workers used: {args.max_workers}")
        print("="*60)
        
        # Save summary with thread-safe file writing
        summary_path = Path(args.leader_agent_path) / "leader_agent_batch_summary.json"
        summary_data = {
            'metadata': {
                'total_instances': len(instances),
                'instances_processed': len(instances_to_run),
                'successful': len(completed),
                'no_changes': len(no_changes),
                'skipped': total_skipped,
                'failed': len(failed),
                'strategy': args.strategy,
                'model_endpoint': args.model_endpoint,
                'truncation_strategy': args.truncation_strategy,
                'max_tokens': args.max_tokens,
                'max_workers': args.max_workers,
                'total_duration_seconds': total_duration,
                'timestamp': datetime.now().isoformat()
            },
            'completed': completed,
            'no_changes': no_changes,
            'skipped': skipped,  # Already contains all skipped instances
            'failed': failed,
            'results': all_results
        }
        
        # Write with file locking for safety
        temp_path = summary_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(summary_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            temp_path.rename(summary_path)
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
            if temp_path.exists():
                temp_path.unlink()
        
        logger.info(f"Summary saved to: {summary_path}")
        
        return 0

def main():
    """Main entry point - now calls run_all_instances"""
    return run_all_instances()

if __name__ == "__main__":
    sys.exit(main()) 