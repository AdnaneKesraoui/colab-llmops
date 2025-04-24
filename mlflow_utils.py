# oas_pipeline/mlflow_utils.py
import mlflow

def start_experiment(name: str):
    mlflow.set_experiment(name)

def log_metrics(metrics: dict, step: int=None):
    for k,v in metrics.items():
        mlflow.log_metric(k, v, step=step)

def log_params(params: dict):
    for k,v in params.items():
        mlflow.log_param(k, v)

def log_artifact(path: str):
    mlflow.log_artifact(path)
