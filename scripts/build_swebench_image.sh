#!/bin/bash

echo "=== Building Pre-built SWE-bench LeaderAgent Docker Image ==="
echo ""

# Build the custom image
echo "Building Docker image with all packages pre-installed..."
docker build -f Dockerfile.swebench-leader -t swebench-leader:latest .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker image built successfully!"
    echo "Image name: swebench-leader:latest"
    echo ""
    echo "Image size:"
    docker images swebench-leader:latest
    echo ""
    echo "You can now use this image for faster SWE-bench evaluations."
    echo "The image includes all necessary packages and is ready to use."
else
    echo ""
    echo "❌ Docker image build failed!"
    echo "Please check the error messages above."
    exit 1
fi
