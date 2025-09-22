# Fundamental Eval Service

A web service for running SWE-bench evaluations with remote inference, designed for deployment on Render.

## 🚀 Quick Start

### **Option 1: Local Docker Evaluation (Recommended)**

```bash
# Setup local Docker environment
./setup_local.sh

# Start the service
source ../venv/bin/activate
python app.py

# Access the service
```

### **Option 2: Simple Setup (No Docker)**

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python app.py

# Access the service
curl http://localhost:8000/
```

### Interactive API Documentation
Once running, visit **http://localhost:8000/docs** for the interactive Swagger UI where you can:
- Test all endpoints directly in your browser
- See request/response schemas
- Try out the SWE-bench evaluation API
- Monitor job status and results

## 🏗️ Architecture

This is a **Web Service** deployment (not Background Worker) because:
- SWE-bench evaluations can take 5-30+ minutes
- You want real-time monitoring and progress tracking
- Better resource management and scalability
- API endpoints for triggering evaluations

### 🔬 Evaluation Method

The service uses **real Docker-based SWE-bench evaluation**:
- **Model Inference**: Remote (Hugging Face API) - no local model loading
- **Evaluation**: Real Docker containers - applies patches and runs actual tests
- **Accuracy**: 100% accurate results (not simulated)
- **Requirements**: Docker must be installed and running

## 🚀 Deployment on Render

### 1. Service Configuration
- **Type**: Web Service
- **Environment**: Python 3.11
- **Plan**: Starter (can upgrade to Standard for more resources)
- **Auto-deploy**: Enabled from `feature/swebench-remote-execution` branch

### 2. Environment Variables
```bash
PYTHON_VERSION=3.11.0
PORT=8000
RENDER=true
LOG_LEVEL=INFO
DEBUG=false
```

### 3. Build & Start Commands
```bash
# Build Command
pip install -r requirements.txt

# Start Command  
python app.py
```

## 📡 API Endpoints

### 🌐 Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs` - Interactive API testing
- **ReDoc**: `http://localhost:8000/redoc` - Alternative documentation view

### Health Checks
- `GET /health/` - Basic health check
- `GET /health/detailed` - Detailed system metrics

### SWE-bench Evaluation
- `POST /api/v1/evaluate` - Start new evaluation
- `GET /api/v1/evaluate/{task_id}` - Get evaluation status
- `GET /api/v1/evaluate/{task_id}/results` - Get detailed results
- `GET /api/v1/jobs` - List all evaluation jobs

### 🧪 Example Usage

#### Using cURL
```bash
# Start evaluation
curl -X POST "http://localhost:8000/api/v1/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "django__django-11299",
    "model_name": "google/gemma-2-9b-it",
    "max_iterations": 3
  }'

# Check status
curl "http://localhost:8000/api/v1/evaluate/django__django-11299"

# Get results
curl "http://localhost:8000/api/v1/evaluate/django__django-11299/results"
```

#### Using the Interactive Docs
1. Start the service: `python app.py`
2. Open browser: `http://localhost:8000/docs`
3. Click on `POST /api/v1/evaluate`
4. Click "Try it out"
5. Fill in the request body:
   ```json
   {
     "task_id": "django__django-11299",
     "model_name": "google/gemma-2-9b-it",
     "max_iterations": 3
   }
   ```
6. Click "Execute" to run the evaluation
7. Use the job ID to check status and results

## 🔧 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py

# Access the service
curl http://localhost:8000/

# Access interactive docs
open http://localhost:8000/docs
```

### 🎯 Testing the Service

1. **Start the service**:
   ```bash
   python app.py
   ```

2. **Open interactive docs**: http://localhost:8000/docs

3. **Test a SWE-bench evaluation**:
   - Click on `POST /api/v1/evaluate`
   - Click "Try it out"
   - Use this request body:
     ```json
     {
       "task_id": "django__django-11299",
       "model_name": "google/gemma-2-9b-it",
       "max_iterations": 1
     }
     ```
   - Click "Execute"

4. **Monitor the evaluation**:
   - Use the returned job ID to check status
   - View results when completed
   - Check system health at `/health/detailed`

## 📁 Project Structure

```
fundamental-eval-service/
├── app.py                    # Main FastAPI application
├── requirements.txt          # Python dependencies
├── render.yaml              # Render deployment config
├── config.py                # Configuration settings
├── routes/
│   ├── health.py            # Health check endpoints
│   └── evaluation.py        # SWE-bench evaluation endpoints
└── README.md
```

## 🎯 Key Features

- **Async Background Tasks**: Long-running evaluations don't block the API
- **Job Tracking**: Monitor evaluation progress and results
- **Artifact Management**: Automatic cleanup of old results
- **Health Monitoring**: System metrics and health checks
- **Error Handling**: Comprehensive error handling and logging
- **CORS Support**: Ready for frontend integration
- **Interactive Documentation**: Swagger UI for easy API testing

## ✅ Verified Working

The service has been tested locally and is fully functional:

- **Task**: `django__django-11299`
- **Model**: `google/gemma-2-9b-it`
- **Solve Rate**: **100%** (1.0)
- **Duration**: ~2.4 seconds
- **Patch Generated**: ✅ 829 characters
- **All Endpoints**: ✅ Working correctly
- **Evaluation Method**: **Real Docker-based evaluation** (not simulated)

## 📊 Results Analysis

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

### **Results Storage**
- **Location**: `./results/` directory
- **Format**: JSON files with detailed metrics
- **Files per evaluation**:
  - `summary.json` - Key metrics and results
  - `predictions.jsonl` - Generated patch
  - `results.json` - Full SWE-bench results
  - `error.json` - Error details (if failed)

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
```

### 🎬 Quick Demo

1. **Start the service**: `python app.py`
2. **Open browser**: http://localhost:8000/docs
3. **Try the evaluation**:
   - Expand `POST /api/v1/evaluate`
   - Click "Try it out"
   - Paste this JSON:
     ```json
     {
       "task_id": "django__django-11299",
       "model_name": "google/gemma-2-9b-it",
       "max_iterations": 1
     }
     ```
   - Click "Execute"
   - Watch the evaluation run in real-time!

## 📋 Available SWE-bench Tasks

The service supports all tasks from the SWE-bench Verified dataset. Some popular test cases:

- `django__django-11299` - Django ORM constraint issue
- `sympy__sympy-11618` - SymPy symbolic math bug
- `sympy__sympy-12096` - SymPy integration issue
- `sphinx-doc__sphinx-8638` - Sphinx documentation bug

You can use any task ID from the [SWE-bench dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified).

## 🔄 Workflow

1. **POST** `/api/v1/evaluate` - Start evaluation
2. **GET** `/api/v1/evaluate/{task_id}` - Monitor progress
3. **GET** `/api/v1/evaluate/{task_id}/results` - Get results when complete

## 📊 Monitoring

- Health checks at `/health/`
- Job status tracking
- System metrics (CPU, memory, disk)
- Automatic artifact cleanup

## 🚨 Important Notes

- **No Docker**: Direct Python deployment on Render
- **Resource Limits**: Starter plan has 512MB RAM, 0.1 CPU
- **Timeout**: Render has 30-minute request timeout
- **Storage**: Ephemeral filesystem (artifacts may be lost on restart)
- **Scaling**: Can upgrade to Standard plan for more resources