# 🐳 Lightweight Local Docker SWE-bench Evaluation

A complete setup for running real SWE-bench evaluations locally with minimal resource usage and external result storage for analysis.

## 🎯 **What This Setup Provides**

- ✅ **Real Docker-based evaluation** (not simulated)
- ✅ **Minimal resource usage** (no caching, lightweight images)
- ✅ **External result storage** (easy analysis and review)
- ✅ **Render web service** (remote patch generation)
- ✅ **Local Docker evaluation** (real testing)
- ✅ **Built-in analysis tools** (comprehensive result review)

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Render API    │    │   Hugging Face   │    │  Local Docker   │
│   (Web Service) │◄──►│   (Model API)    │◄──►│  (Evaluation)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Job Queue     │    │   Patch Gen      │    │   Results       │
│   (Redis)       │    │   (Remote)       │    │   (External)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Workflow:**
1. **Render API** receives evaluation request
2. **Hugging Face API** generates code patch (remote)
3. **Local Docker** runs real SWE-bench evaluation
4. **Results stored externally** for analysis

## 🚀 **Quick Start**

### **1. One-Command Setup**
```bash
cd fundamental-eval-service
./setup_local.sh
```

### **2. Start the Service**
```bash
source ../venv/bin/activate
python app.py
```

### **3. Access the API**
- **Interactive docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health
- **API base**: http://localhost:8000/api/v1

### **4. Submit Evaluation**
```bash
curl -X POST "http://localhost:8000/api/v1/evaluate" \
     -H "Content-Type: application/json" \
     -d '{
       "task_id": "django__django-11299",
       "model_name": "google/gemma-2-9b-it",
       "max_iterations": 3,
       "max_workers": 1
     }'
```

## 📊 **Results Analysis**

### **Analyze All Results**
```bash
# View comprehensive analysis of all evaluations
python analyze_results.py
```

### **View Individual Result**
```bash
# Show detailed information for a specific evaluation
python analyze_results.py results/django__django-11299_google_gemma-2-9b-it/
```

### **Sample Analysis Output**
```
🔍 Analyzing results in: ./results
============================================================
📊 Found 3 evaluation results

📋 EVALUATION SUMMARY
------------------------------------------------------------
Task ID                   Model               Solve Rate   Duration   
------------------------------------------------------------
django__django-11299      gemma-2-9b-it       100.00%      2.4s       
sympy__sympy-20590        gemma-2-9b-it       0.00%        15.2s      
pandas__pandas-12345      gemma-2-9b-it       50.00%       8.7s       
------------------------------------------------------------
AVERAGE                                           50.00%      8.8s       

📈 DETAILED ANALYSIS
------------------------------------------------------------
🤖 Model Performance:
  gemma-2-9b-it           50.00%   (1/2) avg: 8.8s

🎯 Task Difficulty:
  django__django-11299    100.00%  (1/1)
  sympy__sympy-20590      0.00%    (0/1)
  pandas__pandas-12345    50.00%   (1/2)
```

## 📁 **Results Storage Structure**

```
results/
├── django__django-11299_google_gemma-2-9b-it/
│   ├── summary.json          # Key metrics and results
│   ├── predictions.jsonl     # Generated patch
│   ├── results.json          # Full SWE-bench results
│   └── error.json           # Error details (if failed)
├── sympy__sympy-20590_google_gemma-2-9b-it/
│   └── ...
└── evaluation_summary.csv    # Exported analysis data
```

### **File Descriptions:**

- **`summary.json`**: Key metrics, solve rate, duration, timestamp
- **`predictions.jsonl`**: Generated patch content in SWE-bench format
- **`results.json`**: Complete SWE-bench evaluation results
- **`error.json`**: Detailed error information (if evaluation failed)
- **`evaluation_summary.csv`**: All results in CSV format for spreadsheet analysis

## 🔧 **Configuration**

### **Docker Configuration**
```yaml
# docker-compose.yml
version: '3.8'
services:
  swebench-evaluator:
    build:
      context: .
      dockerfile: Dockerfile.evaluator
    volumes:
      - ./results:/app/results      # External result storage
      - ./cache:/app/cache          # Optional cache (minimal)
    environment:
      - PYTHONUNBUFFERED=1
      - SWEBENCH_CACHE_LEVEL=base   # Minimal caching
    deploy:
      resources:
        limits:
          memory: 4G                # Memory limit
          cpus: '2.0'              # CPU limit
```

### **Environment Variables**
```bash
# Required
TASK_ID=django__django-11299
PATCH_CONTENT="diff --git a/file.py..."
MODEL_NAME=google/gemma-2-9b-it

# Optional
SWEBENCH_CACHE_LEVEL=base          # base, env, instance
MAX_WORKERS=1                      # Parallel evaluation workers
TIMEOUT=1800                       # 30 minute timeout
```

## 💰 **Resource Requirements**

### **Minimum System Requirements:**
- **CPU**: 2+ cores (4+ recommended)
- **Memory**: 4GB+ RAM (8GB+ recommended)
- **Storage**: 20GB+ free space
- **Docker**: Installed and running

### **Resource Usage:**
| Component | CPU | Memory | Storage | Duration |
|-----------|-----|--------|---------|----------|
| **API Service** | Low | Low | 100MB | Always running |
| **Patch Generation** | Low | Low | 1MB | 2-5 seconds |
| **Docker Evaluation** | High | 2-4GB | 5-10GB | 5-30 minutes |
| **Results Storage** | Low | Low | 1-10MB | Instant |

### **Cost Breakdown:**
- **Render API**: $25/month (Standard plan)
- **Hugging Face**: $0.01-0.10 per evaluation
- **Local Docker**: Free (your hardware)
- **Total**: ~$25/month + usage costs

## 🛠️ **Troubleshooting**

### **Common Issues:**

#### **Docker Not Found**
```bash
# Install Docker Desktop
brew install --cask docker
# Or download from: https://www.docker.com/products/docker-desktop/
```

#### **Permission Denied**
```bash
# Fix script permissions
chmod +x setup_local.sh
chmod +x analyze_results.py
```

#### **Docker Build Failed**
```bash
# Clean Docker cache
docker system prune -a
# Rebuild
docker-compose build --no-cache
```

#### **Evaluation Timeout**
```bash
# Increase timeout in evaluate_local.py
timeout=3600  # 60 minutes instead of 30
```

#### **Out of Memory**
```bash
# Reduce memory limit in docker-compose.yml
memory: 2G  # Instead of 4G
```

### **Debug Commands:**
```bash
# Check Docker status
docker --version
docker-compose --version

# Test Docker setup
docker-compose run --rm swebench-evaluator python -c "import swebench; print('OK')"

# View Docker logs
docker-compose logs swebench-evaluator

# Check results
ls -la results/
cat results/*/summary.json
```

## 📈 **Performance Optimization**

### **Faster Evaluations:**
1. **Use SSD storage** for Docker volumes
2. **Increase memory limit** to 8GB
3. **Use multiple CPU cores** (max_workers=2-4)
4. **Enable Docker build cache** (subsequent builds)

### **Storage Optimization:**
```bash
# Clean up old results
rm -rf results/*/

# Clean Docker cache
docker system prune -a

# Compress old results
tar -czf results_backup.tar.gz results/
```

## 🔄 **Workflow Examples**

### **Single Evaluation**
```bash
# 1. Submit evaluation
curl -X POST "http://localhost:8000/api/v1/evaluate" \
     -H "Content-Type: application/json" \
     -d '{"task_id": "django__django-11299", "model_name": "google/gemma-2-9b-it"}'

# 2. Check status
curl "http://localhost:8000/api/v1/evaluate/django__django-11299"

# 3. Analyze results
python analyze_results.py
```

### **Batch Evaluation**
```bash
# Evaluate multiple tasks
for task in django__django-11299 sympy__sympy-20590 pandas__pandas-12345; do
    curl -X POST "http://localhost:8000/api/v1/evaluate" \
         -H "Content-Type: application/json" \
         -d "{\"task_id\": \"$task\", \"model_name\": \"google/gemma-2-9b-it\"}"
    sleep 30  # Wait between requests
done

# Analyze all results
python analyze_results.py
```

### **Model Comparison**
```bash
# Test different models on same task
models=("google/gemma-2-9b-it" "meta-llama/Llama-2-7b-chat-hf" "microsoft/DialoGPT-medium")
task="django__django-11299"

for model in "${models[@]}"; do
    curl -X POST "http://localhost:8000/api/v1/evaluate" \
         -H "Content-Type: application/json" \
         -d "{\"task_id\": \"$task\", \"model_name\": \"$model\"}"
    sleep 60  # Wait between evaluations
done

# Compare results
python analyze_results.py
```

## 📚 **API Reference**

### **Submit Evaluation**
```http
POST /api/v1/evaluate
Content-Type: application/json

{
  "task_id": "django__django-11299",
  "model_name": "google/gemma-2-9b-it",
  "max_iterations": 3,
  "max_workers": 1
}
```

### **Check Status**
```http
GET /api/v1/evaluate/{task_id}
```

### **List All Jobs**
```http
GET /api/v1/jobs?status=completed&limit=10
```

### **Get Results**
```http
GET /api/v1/results/{job_id}
```

## 🎯 **Best Practices**

### **Evaluation Strategy:**
1. **Start small** - Test with simple tasks first
2. **Monitor resources** - Watch CPU/memory usage
3. **Batch processing** - Group similar evaluations
4. **Regular cleanup** - Remove old results and Docker cache

### **Analysis Workflow:**
1. **Run evaluations** - Submit multiple tasks/models
2. **Wait for completion** - Monitor job status
3. **Analyze results** - Use `analyze_results.py`
4. **Export data** - Use generated CSV files
5. **Clean up** - Remove old results

### **Development Tips:**
1. **Use interactive docs** - http://localhost:8000/docs
2. **Check logs** - Monitor console output
3. **Test locally** - Verify setup before deployment
4. **Version control** - Commit results for tracking

## 🚀 **Deployment to Render**

### **1. Update render.yaml**
```yaml
services:
  - type: web_service
    name: swebench-local-eval
    env: python
    plan: standard
    buildCommand: pip install -r requirements.txt
    startCommand: python app.py
    envVars:
      - key: HF_TOKEN
        sync: false
```

### **2. Deploy to Render**
```bash
# Push to your repository
git add .
git commit -m "Add lightweight Docker evaluation"
git push origin main

# Render will auto-deploy from your repository
```

### **3. Test Remote API**
```bash
# Replace with your Render URL
curl -X POST "https://swebench-local-eval.onrender.com/api/v1/evaluate" \
     -H "Content-Type: application/json" \
     -d '{"task_id": "django__django-11299", "model_name": "google/gemma-2-9b-it"}'
```

## 📞 **Support**

### **Getting Help:**
1. **Check logs** - Look for error messages
2. **Verify setup** - Run `./setup_local.sh` again
3. **Test components** - Verify Docker, API, and analysis tools
4. **Review documentation** - Check this README and main README

### **Common Commands:**
```bash
# Full reset
docker system prune -a
rm -rf results/
./setup_local.sh

# Quick test
docker-compose run --rm swebench-evaluator python -c "print('Docker OK')"
python -c "import requests; print('API OK')" 
python analyze_results.py
```

---

**🎉 You now have a complete, lightweight, real SWE-bench evaluation system!**

This setup provides real Docker-based evaluation with minimal resource usage and external result storage for easy analysis. Perfect for research, development, and production use! 🚀
