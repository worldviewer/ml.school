---
name: metaflow
description: Write Metaflow pipeline code following the project's patterns including custom decorators, Pipeline base class, cross-validation, MLflow tracking, and backend abstraction. Use this when creating or modifying Metaflow flows.
---

# Metaflow Pipeline Development

This skill helps you write Metaflow pipelines that follow the established patterns in this codebase.

## Core Architecture

### Base Pipeline Class

All pipelines inherit from the `Pipeline` base class defined in [src/common/pipeline.py](../../../src/common/pipeline.py):

```python
from common.pipeline import Pipeline, dataset

class MyPipeline(Pipeline):
    """Description of what this pipeline does."""
    pass
```

The `Pipeline` base class provides:

- **Config management**: YAML-based project configuration via `Config` parameter
- **Backend abstraction**: Automatic instantiation of Local or SageMaker backends
- **MLflow integration**: Automatic tracking URI configuration
- **Dataset loading**: Automatic data loading and preprocessing
- **Logging**: Preconfigured logger available in every step

### Custom Decorators

The project provides several custom decorators that are automatically applied:

1. **`@dataset`** - Loads and preprocesses data (handles missing values, shuffling)
   - Creates `self.data` artifact with pandas DataFrame
   - Applies to steps that need dataset access
   - Replaces extraneous values with NaN
   - Shuffles data (fixed seed in dev, random in production)

2. **`@logging`** - Configures logging per step
   - Creates `self.logger` artifact
   - Automatically applied to all steps

3. **`@mlflow`** - Sets up MLflow tracking
   - Configures MLflow tracking URI
   - Automatically applied to all steps

4. **`@backend`** - Instantiates backend implementation
   - Creates `self.backend_impl` artifact
   - Supports Local and SageMaker backends
   - Manually applied to steps that need backend access

### Parameters

Pipelines typically define these parameters:

```python
from metaflow import Parameter, Config, config_expr, project

@project(name=config_expr("project.project"))
class MyPipeline(Pipeline):
    # Project configuration (inherited from Pipeline)
    project = Config(
        "project",
        help="Project configuration settings.",
        default="config/local.yml",
        parser=parse_project_configuration,
    )

    # Backend module (inherited from Pipeline)
    backend = Parameter(
        "backend",
        help="Backend module implementation.",
        default=project.backend["module"],
    )

    # Dataset path (inherited from Pipeline)
    dataset = Parameter(
        "dataset",
        help="Project dataset that will be used.",
        default="data/penguins.csv",
    )

    # MLflow tracking URI (inherited from Pipeline)
    mlflow_tracking_uri = Parameter(
        "mlflow-tracking-uri",
        help="MLflow tracking URI.",
        default=project.mlflow_tracking_uri,
    )

    # Custom parameters for your pipeline
    custom_param = Parameter(
        "custom-param",
        help="Description of custom parameter.",
        default=42,
    )
```

## Common Patterns

### 1. Start Step with MLflow

```python
@dataset
@card
@step
def start(self):
    """Start the pipeline."""
    import mlflow

    self.logger.info("MLflow tracking server: %s", self.mlflow_tracking_uri)

    self.mode = "production" if current.is_production else "development"
    self.logger.info("Running flow in %s mode.", self.mode)

    # Start MLflow run
    run = mlflow.start_run(run_name=current.run_id)
    self.mlflow_run_id = run.info.run_id

    # Continue to next step(s)
    self.next(self.next_step)
```

### 2. Parallel Branching

```python
@step
def start(self):
    """Start with parallel branches."""
    # Run multiple independent steps in parallel
    self.next(self.branch_a, self.branch_b)

@step
def branch_a(self):
    """First parallel branch."""
    # Do work
    self.next(self.join)

@step
def branch_b(self):
    """Second parallel branch."""
    # Do work
    self.next(self.join)

@step
def join(self, inputs):
    """Join parallel branches."""
    # Merge artifacts from all inputs
    self.merge_artifacts(inputs)
    self.next(self.end)
```

### 3. Foreach Pattern (Cross-Validation)

```python
@step
def prepare_folds(self):
    """Generate fold indices."""
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=5, shuffle=True)
    self.folds = list(enumerate(kfold.split(self.data)))

    # Run each fold in parallel
    self.next(self.process_fold, foreach="folds")

@step
def process_fold(self):
    """Process one fold."""
    self.fold, (self.train_indices, self.test_indices) = self.input
    self.logger.info("Processing fold %d...", self.fold)

    # Do work for this fold
    self.next(self.aggregate)

@step
def aggregate(self, inputs):
    """Aggregate results from all folds."""
    import numpy as np

    # Merge specific artifacts
    self.merge_artifacts(inputs, include=["mlflow_run_id"])

    # Calculate mean across folds
    metrics = [[i.metric_a, i.metric_b] for i in inputs]
    self.metric_a_mean, self.metric_b_mean = np.mean(metrics, axis=0)

    self.next(self.end)
```

### 4. Environment Variables for Keras/TensorFlow

```python
import os

# Define at module level
environment_variables = {
    "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "tensorflow"),
}

class MyPipeline(Pipeline):
    @environment(vars=environment_variables)
    @step
    def train(self):
        """Train model with proper environment."""
        from keras import models, layers
        # Build and train model
        pass
```

### 5. Nested MLflow Runs

```python
@step
def train_fold(self):
    """Train model for one fold."""
    import mlflow

    # Create nested run under parent run
    with (
        mlflow.start_run(run_id=self.mlflow_run_id),
        mlflow.start_run(
            run_name=f"fold-{self.fold}",
            nested=True,
        ) as run,
    ):
        self.fold_run_id = run.info.run_id

        # Disable model logging for individual folds
        mlflow.autolog(log_models=False)

        # Train model
        self.model = build_model()
        history = self.model.fit(self.x_train, self.y_train)

    self.next(self.evaluate_fold)
```

### 6. Model Registration

```python
@step
def register(self, inputs):
    """Register model in MLflow."""
    import tempfile
    import mlflow
    from pathlib import Path

    self.merge_artifacts(inputs)

    with (
        mlflow.start_run(run_id=self.mlflow_run_id),
        tempfile.TemporaryDirectory() as directory,
    ):
        # Prepare artifacts
        artifacts = {
            "model": save_model(directory),
            "preprocessor": save_preprocessor(directory),
        }

        # Point to source code
        root = Path(__file__).parent.parent
        code_paths = [(root / "inference" / "model.py").as_posix()]

        # Register model
        mlflow.pyfunc.log_model(
            name="model",
            python_model=root / "inference" / "model.py",
            registered_model_name="model-name",
            code_paths=code_paths,
            artifacts=artifacts,
            pip_requirements=["package==1.0.0"],
        )

    self.next(self.end)
```

### 7. Using Backend

```python
from common.pipeline import Pipeline, backend

class MyPipeline(Pipeline):
    @backend
    @step
    def invoke_model(self):
        """Make predictions using backend."""
        # backend_impl is available via @backend decorator
        predictions = self.backend_impl.invoke(data)

        # Save predictions
        self.backend_impl.save(predictions)

        self.next(self.end)
```

### 8. End Step

```python
@step
def end(self):
    """End the pipeline."""
    self.logger.info("The pipeline finished successfully.")
```

## Running Pipelines

### Local Execution

```bash
uv run src/pipelines/my_pipeline.py run
```

### With Decorators

```bash
# With retry
uv run src/pipelines/my_pipeline.py --with retry run

# With custom config
uv run src/pipelines/my_pipeline.py --config project config/sagemaker.yml run
```

### AWS Batch Execution

```bash
METAFLOW_PROFILE=production uv run src/pipelines/my_pipeline.py run --with batch --with retry
```

## Pipeline Structure Template

```python
import os
from pathlib import Path

from metaflow import (
    Parameter,
    card,
    current,
    environment,
    step,
)

from common.pipeline import Pipeline, dataset, backend

# Environment variables (if needed for Keras/TensorFlow)
environment_variables = {
    "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "tensorflow"),
}


class MyPipeline(Pipeline):
    """Description of what this pipeline does.

    This pipeline should:
    - List main responsibilities
    - Describe inputs and outputs
    - Mention key decisions or thresholds
    """

    # Custom parameters
    my_param = Parameter(
        "my-param",
        help="Description of parameter.",
        default="default-value",
    )

    @dataset
    @card
    @step
    def start(self):
        """Start the pipeline."""
        import mlflow

        self.logger.info("MLflow tracking server: %s", self.mlflow_tracking_uri)
        self.mode = "production" if current.is_production else "development"

        run = mlflow.start_run(run_name=current.run_id)
        self.mlflow_run_id = run.info.run_id

        self.next(self.next_step)

    @step
    def next_step(self):
        """Do some work."""
        # Implementation
        self.next(self.end)

    @step
    def end(self):
        """End the pipeline."""
        self.logger.info("The pipeline finished successfully.")


if __name__ == "__main__":
    MyPipeline()
```

## Best Practices

1. **Always use the Pipeline base class** - Inherit from `common.pipeline.Pipeline`
2. **Import decorators explicitly** - Import custom decorators you need (`dataset`, `backend`)
3. **Document your pipeline** - Use clear docstrings for the class and each step
4. **Log important information** - Use `self.logger` for progress and debug info
5. **Use cards for visualization** - Add `@card` to important steps
6. **Handle production vs development** - Use `current.is_production` for mode detection
7. **Merge artifacts in join steps** - Always call `self.merge_artifacts(inputs)` in join steps
8. **Track with MLflow** - Start runs with meaningful names tied to `current.run_id`
9. **Use environment decorator for ML** - Wrap training/inference steps with `@environment`
10. **Follow parameter naming** - Use kebab-case for parameter names, snake_case for attributes

## Configuration Files

Pipelines use YAML configuration files in the `config/` directory:

- `config/local.yml` - Local development configuration
- `config/sagemaker.yml` - AWS SageMaker configuration

Configuration structure:

```yaml
project: project-name
mlflow_tracking_uri: http://localhost:5000
backend:
  module: backend.Local  # or backend.Sagemaker
  # Backend-specific settings
```

## Common Imports

```python
# Metaflow imports
from metaflow import (
    Config,
    Parameter,
    card,
    current,
    environment,
    step,
    project,
    config_expr,
)

# Project imports
from common.pipeline import Pipeline, dataset, backend

# ML imports (as needed)
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
```

## Troubleshooting

1. **Missing artifacts in join steps** - Use `self.merge_artifacts(inputs, include=["artifact_name"])`
2. **MLflow connection errors** - Check `self.mlflow_tracking_uri` and ensure server is running
3. **Backend instantiation errors** - Verify config file and backend module path
4. **Environment variable issues** - Use `@environment(vars=environment_variables)` decorator
5. **Dataset not loading** - Ensure file exists and `@dataset` decorator is applied

## Examples

See these pipelines for reference:

- [src/pipelines/training.py](../../../src/pipelines/training.py) - Full training pipeline with cross-validation
- [src/pipelines/deployment.py](../../../src/pipelines/deployment.py) - Model deployment pipeline
- [src/pipelines/monitoring.py](../../../src/pipelines/monitoring.py) - Model monitoring pipeline
- [src/pipelines/traffic.py](../../../src/pipelines/traffic.py) - Traffic generation pipeline
