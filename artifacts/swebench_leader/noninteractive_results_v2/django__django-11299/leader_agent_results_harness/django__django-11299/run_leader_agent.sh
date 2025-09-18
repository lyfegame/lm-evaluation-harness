#!/bin/bash
set -e

echo "=== Running LeaderAgent for SWE-bench task ==="
echo "Instance: django__django-11299"
echo "Repository: django/django"
echo "Docker Image: swebench/sweb.eval.x86_64.django_1776_django-11299:latest"
echo "Model Endpoint: https://fairies--deploy-checkpoint-9c7df3-9c7d-modelserver-generate.modal.run"
echo ""

# Get the UID and GID from the host user (passed via environment)
HOST_UID=${HOST_UID:-1000}
HOST_GID=${HOST_GID:-1000}

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
cd /testbed || { echo "ERROR: Cannot find repository at /testbed"; exit 1; }

echo "Working directory: $(pwd)"

# Checkout the correct commit
echo "Checking out commit: 6866c91b638de5368c18713fa851bfe56253ea55"
git checkout 6866c91b638de5368c18713fa851bfe56253ea55 2>/dev/null || true

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
mkdir -p /home/tianhangzhu/gcs_view/home/tianhangzhu/verltrain/claude-code_v2
cp /output/*.py /home/tianhangzhu/gcs_view/home/tianhangzhu/verltrain/claude-code_v2 2>/dev/null || true

# Also copy all JSON files
echo "Copying JSON dependencies to expected locations..."
cp /output/*.json /home/tianhangzhu/gcs_view/home/tianhangzhu/verltrain/claude-code_v2 2>/dev/null || true

# Also copy to home directory for backup
echo "Copying Python dependencies to home directory..."
cp /output/*.py /home/agentuser/ 2>/dev/null || true

# Also copy to home directory for backup
echo "Copying JSON dependencies to home directory..."
cp /output/*.json /home/agentuser/ 2>/dev/null || true

# Set proper permissions
echo "Setting proper permissions..."
chown -R agentuser:agentuser /home/tianhangzhu/gcs_view/home/tianhangzhu/verltrain/claude-code_v2 2>/dev/null || true
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
timeout 1800 su - agentuser -c "cd /testbed && export PYTHONPATH=/output:/home/agentuser:$PYTHONPATH && export JUPYTER_ENABLE_UNSAFE_IPV6=1 && export JUPYTER_DEBUG=1 && export IPYTHON_DEBUG=1 && export PYTHONDEVMODE=1 && export PYTHONUNBUFFERED=1 && /usr/bin/python /home/agentuser/leader_agent.py --model-endpoint 'https://fairies--deploy-checkpoint-9c7df3-9c7d-modelserver-generate.modal.run' --prompt-file /output/starting_prompt.txt --working-dir '/testbed' --max-iterations 300 --truncation-strategy 'ast_llm_compaction' --max-tokens 16000 --output-conversation '/output/conversation.json' 2>&1" | tee /output/leader_agent_output.log || true

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
