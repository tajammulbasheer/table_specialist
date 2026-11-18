"""Local experiment tracking with MLflow. Everything stays in mlruns/ on this machine; telemetry is off.

Run IDs are lowercase words joined by underscores and always name the stage, then what ran:
  smoke_<test>_<model>                    smoke_lora_base
  final_<benchmark>_<model>               final_spider_base
  gate_<benchmark>_<model>_<setting>      gate_spider_base_raw
  train_<arm>_<model>_seed<k>_round<r>    train_tablespecialist_base_seed1_round2
Earlier runs named baseline_* predate this scheme and are kept unchanged.
"""
import os
import re

from ts.repo import ROOT

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # plain files; a database backend would need more packages
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

RUN_ID = re.compile(r"^(smoke|final|gate|train|baseline)(_[a-z0-9]+)+$")
DEFAULT_STORE = ROOT / "mlruns"


def check_run_id(run_id):
    if not RUN_ID.match(run_id):
        raise ValueError(f"Run ID does not follow the scheme in ts/tracking.py: {run_id}")
    return run_id


def _flat_numbers(values, prefix=""):
    """Keep numeric fields, flattening nested ones; lists and text are not metrics."""
    out = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat_numbers(value, f"{name}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = value
    return out


def log_run(run_id, experiment, params, metrics, artifact=None, store=DEFAULT_STORE):
    """Record one finished run. Metric names use "." and letters only, so MLflow accepts them."""
    import mlflow  # imported here so that scoring code does not need it

    check_run_id(run_id)
    mlflow.set_tracking_uri(f"file:{store}")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({key: str(value)[:500] for key, value in params.items()})
        mlflow.log_metrics(_flat_numbers(metrics))
        if artifact is not None:
            mlflow.log_artifact(str(artifact))
