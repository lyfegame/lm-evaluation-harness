# Deploying SWE-bench LeaderAgent to Render

This guide explains how to deploy the SWE-bench LeaderAgent as a background worker on Render.

## 🚀 Quick Start

### 1. Prerequisites

- Render account (free tier available)
- Render CLI installed: `npm install -g @render-cli`
- Your Modal endpoint URL

### 2. Deploy

```bash
# Login to Render
render auth login

# Deploy the worker
./scripts/deploy_to_render.sh
```

### 3. Configure Environment Variables

In your Render dashboard, set these environment variables:

**Required:**
- `TASK_ID`: SWE-bench task ID (e.g., `django__django-11299`)
- `MODEL_ENDPOINT`: Your Modal endpoint URL

**Optional:**
- `MAX_ITERATIONS`: Number of iterations (default: 10)
- `MODEL_NAME`: Model name (default: `gemma-3-27b-it@modal`)
- `STRATEGY`: Prompt strategy (default: `systematic`)

## 📋 Manual Deployment

If you prefer manual setup:

1. **Create a new Background Worker** in Render dashboard
2. **Connect your GitHub repository**
3. **Set the following:**
   - **Build Command**: `echo "Using Dockerfile"`
   - **Start Command**: `python scripts/render_worker.py`
   - **Dockerfile Path**: `./Dockerfile.render`
   - **Plan**: Starter (or higher for more resources)

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TASK_ID` | ✅ | - | SWE-bench task ID to evaluate |
| `MODEL_ENDPOINT` | ✅ | - | Modal endpoint URL |
| `MAX_ITERATIONS` | ❌ | 10 | Number of LeaderAgent iterations |
| `MODEL_NAME` | ❌ | gemma-3-27b-it@modal | Model identifier |
| `STRATEGY` | ❌ | systematic | Prompt strategy |
| `TRUNCATION_STRATEGY` | ❌ | ast_llm_compaction | Code truncation method |
| `MAX_TOKENS` | ❌ | 16000 | Maximum tokens per request |
| `ARTIFACT_DIR` | ❌ | /app/artifacts/swebench_leader | Output directory |
| `RUN_ID` | ❌ | render_worker | Run identifier |

### Resource Requirements

- **Minimum Plan**: Starter (512MB RAM, 0.1 CPU)
- **Recommended Plan**: Standard (1GB RAM, 0.5 CPU) or higher
- **Storage**: 2GB+ for Docker images and artifacts

## 🎯 Running Evaluations

### Single Task

```bash
# Update environment variable and restart
render services update swebench-leader-worker --env TASK_ID=django__django-11299
```

### Multiple Tasks

For multiple tasks, you can:

1. **Sequential**: Update `TASK_ID` after each completion
2. **Parallel**: Deploy multiple workers with different `TASK_ID` values
3. **Batch**: Use a script to iterate through task IDs

## 📊 Monitoring

### Logs

View logs in Render dashboard or via CLI:

```bash
render logs swebench-leader-worker --follow
```

### Artifacts

Results are stored in `/app/artifacts/swebench_leader/`:
- `predictions.jsonl`: SWE-bench format predictions
- `results.json`: Evaluation results
- Container logs and intermediate files

## 🔍 Troubleshooting

### Common Issues

1. **Docker space issues**: Upgrade to a higher plan with more storage
2. **Timeout**: Increase timeout in Render settings
3. **Memory issues**: Upgrade to Standard plan or higher
4. **Network issues**: Check Modal endpoint connectivity

### Debug Mode

Enable debug logging by setting:
```bash
render services update swebench-leader-worker --env PYTHONUNBUFFERED=1
```

## 💰 Cost Optimization

- **Use Starter plan** for testing (free tier available)
- **Scale to 0** when not in use
- **Monitor usage** in Render dashboard
- **Use spot instances** if available

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
name: Deploy to Render
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Render
        run: |
          npm install -g @render-cli
          render auth login --token ${{ secrets.RENDER_TOKEN }}
          ./scripts/deploy_to_render.sh
```

## 📈 Scaling

For high-volume evaluations:

1. **Multiple Workers**: Deploy several workers with different task IDs
2. **Queue System**: Implement a task queue (Redis, etc.)
3. **Load Balancing**: Use Render's auto-scaling features
4. **Resource Optimization**: Monitor and adjust plan sizes

## 🆘 Support

- **Render Docs**: https://render.com/docs
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
