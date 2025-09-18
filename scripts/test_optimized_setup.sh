#!/bin/bash

echo "=== Testing Optimized SWE-bench Setup ==="
echo ""

# Check if optimized image exists
if docker images | grep -q "swebench-leader:latest"; then
    echo "✅ Optimized image found: swebench-leader:latest"
    
    # Show image size
    echo ""
    echo "Image details:"
    docker images swebench-leader:latest
    echo ""
    
    # Test the image
    echo "🧪 Testing optimized image..."
    docker run --rm swebench-leader:latest python -c "
import torch
import transformers
import requests
import nbformat
print('✅ All packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'Requests version: {requests.__version__}')
print(f'Nbformat version: {nbformat.__version__}')
"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Optimized image test passed!"
        echo ""
        echo "You can now run SWE-bench evaluations with:"
        echo "python3 scripts/run_swebench_leader.py --task-id django__django-11299"
        echo ""
        echo "This will use the pre-built image and skip package installation."
    else
        echo ""
        echo "❌ Optimized image test failed!"
        echo "Please rebuild the image with: ./scripts/build_swebench_image.sh"
    fi
else
    echo "❌ Optimized image not found!"
    echo ""
    echo "Please build the optimized image first:"
    echo "./scripts/setup_optimized_swebench.sh"
fi
