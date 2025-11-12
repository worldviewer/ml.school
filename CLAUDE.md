# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an end-to-end machine learning system for training, deploying, and monitoring models. The project demonstrates a complete ML workflow using the Palmer Penguins dataset with a classification model deployed both locally and on AWS SageMaker.

## Package Management

**CRITICAL: Use `uv` exclusively for all Python package management.**

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`
- Run Python scripts: `uv run <script-name>.py`
- Run Python tools: `uv run pytest`, `uv run ruff`
- Launch Python REPL: `uv run python`

For scripts with PEP 723 inline metadata:
- `uv run script.py`
- `uv add package-name --script script.py`
- `uv remove package-name --script script.py`

## Task Automation with Just

The project uses `just` as its task runner (see [justfile](justfile)). All common tasks are defined as just recipes:

### Testing & Linting
- `just test` - Run project unit tests using pytest
- Individual tests: `uv run pytest tests/path/to/test_file.py::test_function_name`

### Local Development
- `just mlflow` - Start MLflow server at http://127.0.0.1:5000
- `just train` - Run training pipeline locally
- `just serve` - Serve the latest registered model locally on port 8080
- `just sample` - Send sample request to local model
- `just sqlite` - Display sample statistics from local database

### Monitoring
- `just traffic` - Generate fake traffic to local model
- `just labels` - Generate fake ground truth labels
- `just monitor` - Run monitoring pipeline

### AWS/SageMaker Deployment
- `just aws-setup <user> [region]` - Set up AWS environment
- `just sagemaker-deploy` - Deploy model to SageMaker
- `just sagemaker-sample` - Invoke SageMaker endpoint
- `just sagemaker-traffic` - Generate traffic to SageMaker
- `just sagemaker-monitor` - Run monitoring on SageMaker
- `just aws-train` - Run training pipeline on AWS Batch
- `just aws-deploy` - Deploy to SageMaker using AWS Batch

## Architecture

### Pipeline System (Metaflow-based)

The project uses **Metaflow** for orchestrating ML workflows. All pipelines inherit from a base `Pipeline` class ([src/common/pipeline.py](src/common/pipeline.py)) that provides:

- **Config-based project settings**: YAML configuration files in [config/](config/) define backend, MLflow tracking URI, and deployment targets
- **Custom decorators** applied automatically to all steps:
  - `@dataset` - Loads and preprocesses data (handles missing values, shuffling)
  - `@backend` - Instantiates backend implementation (Local or SageMaker)
  - `@logging` - Configures logging per step
  - `@mlflow` - Sets up MLflow tracking

Key pipelines in [src/pipelines/](src/pipelines/):
- `training.py` - Train model with cross-validation, evaluation, and registration
- `deployment.py` - Deploy latest model from registry to target platform
- `monitoring.py` - Monitor deployed model using Evidently for data/prediction drift
- `traffic.py` - Generate synthetic traffic and labels for testing
- `rag.py` - RAG agent pipeline using Google ADK
- `indexing.py` - Build FAISS vector index for RAG

### Backend Abstraction

The [src/inference/backend.py](src/inference/backend.py) defines an abstract `Backend` interface with two implementations:

1. **Local**: Uses SQLite database and local HTTP endpoint
2. **Sagemaker**: AWS SageMaker endpoints with S3 for data capture

Both backends support:
- `load()` - Load production data
- `save()` - Store predictions
- `label()` - Generate ground truth labels
- `invoke()` - Make predictions
- `deploy()` - Deploy model

Configuration is environment-specific:
- [config/local.yml](config/local.yml) - Local development
- [config/sagemaker.yml](config/sagemaker.yml) - AWS deployment (uses env vars: `$ENDPOINT_NAME`, `$BUCKET`, `$AWS_REGION`)

### Model Implementation

The custom MLflow model ([src/inference/model.py](src/inference/model.py)) wraps:
- Scikit-learn preprocessing pipeline (feature transformations)
- Keras neural network (classification model)
- Custom `predict()` method returning class labels and probabilities

### Agent System

Two agent implementations in [src/agents/](src/agents/):

1. **RAG Agent** ([agents/rag/](src/agents/rag/)) - Google ADK-based agent with FAISS retrieval
2. **Tic-Tac-Toe Agent** ([agents/tic_tac_toe/](src/agents/tic_tac_toe/)) - Multi-agent system with game logic, player, and commentator sub-agents

## Environment Variables

Key environment variables (defined in `.env`):
- `KERAS_BACKEND` - Backend for Keras (default: "tensorflow")
- `MLFLOW_TRACKING_URI` - MLflow server URL
- `ENDPOINT_NAME` - SageMaker endpoint name
- `BUCKET` - S3 bucket for SageMaker artifacts
- `AWS_REGION` - AWS region
- `AWS_ROLE` - IAM role ARN for SageMaker

## Running Pipelines

Metaflow pipelines can be run with decorators:
```bash
# Local execution
uv run src/pipelines/training.py run

# With retry decorator
uv run src/pipelines/training.py --with retry run

# AWS Batch execution with production profile
METAFLOW_PROFILE=production uv run src/pipelines/training.py run --with batch --with retry

# With custom config
uv run src/pipelines/deployment.py --config project config/sagemaker.yml run --backend backend.Sagemaker
```

## Testing Strategy

Tests are organized in [tests/](tests/) mirroring the source structure:
- `tests/pipelines/` - Pipeline step tests with fixtures
- `tests/inference/` - Model and backend tests
- `tests/common/` - Shared utilities and pipeline base tests

Pytest configuration in [pyproject.toml](pyproject.toml:68-74):
- Integration tests marked with `@pytest.mark.integration`
- Skip integration tests: `uv run pytest -m "not integration"`
- Fail fast after 2 failures: `--maxfail=2`

## Code Quality

- **Linter/Formatter**: Ruff (configured in [pyproject.toml](pyproject.toml:37-58))
- **Type Checking**: Pyright (type checking mode: off)
- Format code: `uv run ruff format`
- Lint code: `uv run ruff check`
- Auto-fix: `uv run ruff check --fix`

## Project Structure

```
src/
├── agents/           # AI agents (RAG, Tic-Tac-Toe)
├── common/           # Shared utilities (pipeline base, embeddings)
├── inference/        # Model and backend implementations
├── pipelines/        # Metaflow workflows
└── scripts/          # Helper scripts (AWS setup)
```

## AWS Infrastructure

CloudFormation templates in [cloud-formation/](cloud-formation/):
- `mlflow-cfn.yaml` - MLflow server on EC2

Setup commands:
- `just aws-setup <username>` - Creates IAM user, S3 bucket, configures Metaflow
- `just aws-mlflow` - Deploy MLflow stack
- `just aws-ssh` - SSH to MLflow server
- `just aws-teardown` - Clean up all resources

## MLflow Integration

MLflow is used for:
- Experiment tracking (autologging enabled)
- Model registry (models registered as "penguins")
- Model serving (both local and via SageMaker)

Access MLflow UI at http://127.0.0.1:5000 after running `just mlflow`.

## Data Flow

1. **Training**: [data/penguins.csv](data/penguins.csv) → Training Pipeline → MLflow Model Registry
2. **Deployment**: Model Registry → Backend (Local/SageMaker)
3. **Inference**: Client → Backend → Model → Backend (saves predictions)
4. **Monitoring**: Backend datastore → Monitoring Pipeline → Evidently Reports
