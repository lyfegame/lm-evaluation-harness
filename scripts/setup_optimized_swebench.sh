#!/bin/bash

echo "=== SWE-bench LeaderAgent Optimization Setup ==="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"

# Check if we have the base SWE-bench image
echo ""
echo "Checking for base SWE-bench image..."
if docker images | grep -q "swebench/sweb.eval.x86_64.django_1776_django-11299"; then
    echo "✅ Base SWE-bench image found"
else
    echo "📥 Pulling base SWE-bench image..."
    docker pull swebench/sweb.eval.x86_64.django_1776_django-11299:latest
    if [ $? -eq 0 ]; then
        echo "✅ Base SWE-bench image pulled successfully"
    else
        echo "❌ Failed to pull base SWE-bench image"
        exit 1
    fi
fi

# Build the optimized image
echo ""
echo "🔨 Building optimized SWE-bench LeaderAgent image..."
echo "This will take several minutes as it installs all packages..."
echo ""

./scripts/build_swebench_image.sh

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup complete!"
    echo ""
    echo "=== Usage Instructions ==="
    echo ""
    echo "1. **Fast Mode (Pre-built Image)**:"
    echo "   python3 scripts/run_swebench_leader.py --task-id django__django-11299"
    echo "   (Uses pre-built image, skips package installation)"
    echo ""
    echo "2. **Development Mode (Original Image)**:"
    echo "   Edit config_default.yaml and set use_prebuilt_image: false"
    echo "   python3 scripts/run_swebench_leader.py --task-id django__django-11299"
    echo "   (Uses original image, installs packages each time)"
    echo ""
    echo "3. **Check Image Status**:"
    echo "   docker images | grep swebench"
    echo ""
    echo "=== Benefits of Pre-built Image ==="
    echo "✅ Faster startup (no package installation)"
    echo "✅ No disk space issues"
    echo "✅ Consistent environment"
    echo "✅ Better for production use"
    echo ""
else
    echo ""
    echo "❌ Setup failed!"
    echo "Please check the error messages above."
    exit 1
fi
