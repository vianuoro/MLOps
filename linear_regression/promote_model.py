import sys
from mlflow.tracking import MlflowClient
import mlflow

def promote_model(model_name, version, tag_name="deployment_stage", tag_value="Production"):
    client = MlflowClient()

    # --- DEPRECATED: transition_model_version_stage (optional fallback) ---
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=tag_value
        )
        print(f"✅ Deprecated stage set: {model_name} v{version} → {tag_value}")
    except Exception as e:
        print(f"⚠️  Skipping deprecated stage API: {e}")

    # --- ✅ Recommended: Use tags instead ---
    try:
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key=tag_name,
            value=tag_value
        )
        print(f"✅ Tag '{tag_name}' set to '{tag_value}' for {model_name} v{version}")
    except Exception as e:
        print(f"❌ Failed to tag model version: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python promote_model.py <model_name> <version> <stage>")
        sys.exit(1)

    model_name = sys.argv[1]
    version = int(sys.argv[2])
    stage = sys.argv[3]

    promote_model(model_name, version, tag_value=stage)
