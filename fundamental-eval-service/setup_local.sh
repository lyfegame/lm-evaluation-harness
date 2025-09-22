#!/bin/bash

# Setup script for lightweight local Docker evaluation

echo "🚀 Setting up lightweight SWE-bench evaluation environment..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p results
mkdir -p cache
mkdir -p local_results

# Set permissions
chmod 755 results
chmod 755 cache
chmod 755 local_results

# Build Docker image
echo "🐳 Building Docker image..."
docker-compose build swebench-evaluator

# Test Docker setup
echo "🧪 Testing Docker setup..."
docker-compose run --rm swebench-evaluator python -c "import swebench; print('✅ SWE-bench installed successfully')"

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Start the web service: python app.py"
echo "2. Open browser: http://localhost:8000/docs"
echo "3. Submit evaluation request"
echo "4. Check results in: ./results/"
echo ""
echo "🔍 To analyze results:"
echo "ls -la results/"
echo "cat results/*/summary.json"
