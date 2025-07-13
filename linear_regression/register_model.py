import sys
import mlflow

def register_model(run_id, model_name="MyModel"):
    # Path to model artifact within the run
    model_uri = f"runs:/{run_id}/model"
    
    # Register the model to the MLflow Model Registry
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    print(f"Model registered: {result.name}, version: {result.version}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python register_model.py <run_id> [model_name]")
        sys.exit(1)

    run_id = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "MyModel"

    register_model(run_id, model_name)
