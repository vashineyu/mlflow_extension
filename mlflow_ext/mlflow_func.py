import mlflow


def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)


def log_params(params: dict):
    mlflow.log_params(params)


def log_tags(tags: dict):
    mlflow.set_tags(tags)


def log_artifact(filepath: str):
    mlflow.log_artifact(filepath)
