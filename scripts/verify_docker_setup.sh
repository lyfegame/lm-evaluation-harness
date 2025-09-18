#!/bin/bash

echo "=== Docker Sanity Checks for SWE-bench ==="
echo ""

echo "1. Checking Docker version..."
docker --version
echo ""

echo "2. Testing Docker with hello-world..."
docker run --rm hello-world
echo ""

echo "3. Pulling SWE-bench test image..."
docker pull swebench/sweb.eval.x86_64.django_1776_django-11299:latest
echo ""

echo "4. Verifying image was pulled successfully..."
docker images | grep swebench
echo ""

echo "=== Docker setup verification complete ==="
echo "If all steps completed without errors, Docker is ready for SWE-bench evaluation."
