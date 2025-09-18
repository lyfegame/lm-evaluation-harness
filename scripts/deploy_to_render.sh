#!/bin/bash
set -euxo pipefail

echo "=== Deploying SWE-bench LeaderAgent to Render ==="
echo ""

# Check if render CLI is installed
if ! command -v render &> /dev/null; then
    echo "❌ Render CLI not found. Please install it first:"
    echo "   npm install -g @render/cli"
    echo "   or visit: https://render.com/docs/cli"
    exit 1
fi

# Check if user is logged in
if ! render auth whoami &> /dev/null; then
    echo "❌ Not logged in to Render. Please run:"
    echo "   render auth login"
    exit 1
fi

echo "✅ Render CLI is ready"
echo ""

# Deploy using render.yaml
echo "🚀 Deploying background worker..."
render services create --file render.yaml

echo ""
echo "✅ Deployment initiated!"
echo ""
echo "Next steps:"
echo "1. Go to your Render dashboard"
echo "2. Set the following environment variables:"
echo "   - TASK_ID: The SWE-bench task ID to evaluate"
echo "   - MODEL_ENDPOINT: Your Modal endpoint URL"
echo "3. Start the worker manually or set up triggers"
echo ""
echo "To run a specific task:"
echo "   render services update swebench-leader-worker --env TASK_ID=django__django-11299"
echo ""
