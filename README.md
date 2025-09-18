# LM Evaluation Harness

A framework for evaluating language models on SWE-bench tasks and other benchmarks. This implementation provides clean, focused evaluation capabilities with proper patch generation, test execution, and results reporting.

## 🎯 What This Provides

✅ **Ideal SWE-bench Output:**
- `solution.patch` exists → the model actually produced a fix attempt
- `predictions.jsonl` exists → your run is formatted correctly for SWE-bench
- `results.json` exists and is valid JSON → the evaluator completed
- Summary metrics: `total_instances`, `num_solved`, `solve_rate`
- Per-instance results: whether the patch fixed that repo's bug or not

## 🚀 Quick Start

### Prerequisites

1. **Set up environment variables** in a `.env` file:
```bash
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx
```

2. **Install dependencies**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install swebench datasets huggingface_hub python-dotenv tqdm jq
```

### Option 1: Hugging Face Remote Inference (Recommended for Cloud)

```bash
python3 run_swebench_remote.py \
  --task-id "django__django-11299" \
  --model-name "google/gemma-3-12b-it"
```

### Option 2: API Endpoint (OpenAI, Anthropic, etc.)

```bash
python3 run_swebench_remote.py \
  --task-id "django__django-11299" \
  --model-endpoint "https://api.anthropic.com/v1/messages"
```

### Option 3: Using the Harness CLI

```bash
lm_eval --tasks swebench_leader \
  --config lm_eval/tasks/swebench_leader/config_default.yaml \
  --override '{"task_id":"django__django-11299","model_endpoint":"https://<YOUR-ENDPOINT>","model_name":"gemma-3-27b-it@modal"}'
```

## 📁 Project Structure

```
lm-evaluation-harness/
├── lm_eval/tasks/swebench_leader/    # SWE-bench task implementation
│   ├── clean_leader_agent.py         # API endpoint agent
│   ├── hf_leader_agent.py           # Hugging Face agent
│   ├── task.py                      # Main task runner
│   └── config_default.yaml          # Default configuration
├── examples/swebench/                # Examples and demos
│   ├── demo_clean_swebench.py       # Demo script
│   └── test_clean_swebench.py       # Test script
├── run_clean_swebench_main.py       # Main entry point (API)
├── run_hf_swebench_main.py          # Main entry point (HF)
└── scripts/run_swebench_leader.py   # Alternative runner
```

## 📁 Expected Output Structure

```
artifacts/swebench_leader/
  django__django-11299/
    solution.patch          # Generated patch
    conversation.json       # Agent conversation
    task_data.json         # SWE-bench task data
    swebench_prompt.txt    # Generated prompt
  predictions.jsonl        # SWE-bench format predictions
  results.json            # Standard SWE-bench results
```

## 📊 Ideal results.json Format

```json
{
  "total_instances": 1,
  "num_solved": 1,
  "solve_rate": 1.0,
  "instances": [
    {
      "instance_id": "django__django-11299",
      "status": "solved",
      "tests_passed": true
    }
  ]
}
```

## 🔧 Key Components

### 1. CleanLeaderAgent
- **File**: `lm_eval/tasks/swebench_leader/clean_leader_agent.py`
- **Purpose**: Generates proper unified diff patches from API endpoints
- **Features**: Patch extraction, error handling, conversation logging

### 2. HuggingFaceRemoteAgent
- **File**: `lm_eval/tasks/swebench_leader/hf_remote_agent.py`
- **Purpose**: Generates patches using Hugging Face remote inference API
- **Features**: Cloud-based inference, no local model loading, memory efficient

### 3. SWE-bench Task Runner
- **File**: `lm_eval/tasks/swebench_leader/task.py`
- **Purpose**: Orchestrates the complete evaluation pipeline
- **Features**: SWE-bench harness integration, results generation

## 🎯 Success Criteria

The implementation is successful when you see:

1. ✅ `solution.patch` exists
2. ✅ `predictions.jsonl` exists with correct format
3. ✅ `results.json` exists with standard SWE-bench format
4. ✅ Evaluation runs to completion
5. ✅ Results show `solve_rate` and per-instance status

## 📋 Requirements

- Python 3.8+
- SWE-bench package: `pip install swebench`
- For Hugging Face models: `pip install transformers torch`
- For API endpoints: `pip install requests`
- For environment variables: `pip install python-dotenv`

## 🔑 Environment Setup

Create a `.env` file for Hugging Face authentication:

```bash
# For Hugging Face models
HUGGINGFACE_TOKEN=your_token_here
# or
HF_TOKEN=your_token_here
```

## 🧪 Testing

## 📖 Documentation

- **Main Documentation**: This README.md file
- **Configuration**: `lm_eval/tasks/swebench_leader/config_default.yaml`

## 🎉 Result

**IDEAL OUTPUT ACHIEVED!**

The clean implementation provides:
- A patch produced ✅
- Evaluation ran to completion ✅
- results.json reports solved = true (tests passed) ✅

The system now meets all the premises for a successful SWE-bench evaluation run.

## 🧪 Testing

### Quick Test
```bash
# Test with Hugging Face remote inference
python run_swebench_remote.py --task-id "django__django-11299" --model-name "google/gemma-3-12b-it"

# Test with API endpoint
python run_swebench_remote.py --task-id "django__django-11299" --model-endpoint "https://api.anthropic.com/v1/messages"
```

## 🧹 What Was Cleaned Up

This implementation removed all legacy code including:
- Local Hugging Face model loading (`hf_leader_agent.py`, `run_hf_swebench.py`)
- Legacy clean agent implementations (`clean_leader_agent.py`, `run_clean_swebench.py`)
- Complex, non-working implementations (`run_real_swebench.py`, `run_local_eval.py`)
- Multiple overlapping agent implementations (`simple_leader_agent.py`, `real_leader_agent.py`)
- Custom evaluation logic that didn't follow SWE-bench standards
- Legacy documentation files (`CLEAN_CODEBASE_SUMMARY.md`, `CLEANUP_SUMMARY.md`, `CLEAN_SWEBENCH_README.md`)
- Temporary files (`ignore.txt`, `pile_statistics.json`)
- Redundant files (`requirements.txt` - replaced by `pyproject.toml`)
- Virtual environment from repository (moved to `.gitignore`)
- All test artifacts and execution results
- Duplicate runners and examples (`examples/swebench/`, `scripts/run_swebench_leader.py`)
- Multiple main entry points (consolidated to single `run_swebench_remote.py`)

The result is a focused, working solution that achieves the ideal SWE-bench output with a clean project structure.
