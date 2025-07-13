import mlflow
from mlflow.tracking import MlflowClient
import sys

def get_latest_run_id(experiment_name="Default"):
    client = MlflowClient()
    
    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Get latest run by start time
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("No runs found in the experiment")
    
    return runs[0].info.run_id

def register_model_from_latest_run(experiment_name="Default", model_name="MyModel"):
    run_id = get_latest_run_id(experiment_name)
    model_uri = f"runs:/{run_id}/model"

    # Register the model
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    print(f"✅ Model registered:")
    print(f"   Name: {result.name}")
    print(f"   Version: {result.version}")
    print(f"   Source Run ID: {run_id}")

if __name__ == "__main__":
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else "Default"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "MyModel"

    register_model_from_latest_run(experiment_name, model_name)
