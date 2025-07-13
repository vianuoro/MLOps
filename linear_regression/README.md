## Train model
python3 train_model.py

## Launch MLFlow UI and list the runs
mlflow ui &
ls mlruns/0/

## Serve the model trained
mlflow models serve -m runs:/<RUN-ID-1>/model -p 1234 &
curl -X POST http://127.0.0.1:1234/invocations   -H "Content-Type: application/json"   -d '{
        "dataframe_split": {
          "columns": ["feature_0"],
          "data": [[0.5], [0.2]]
        }
      }'

## Add to MLFlow Model Registry and Serve its 3rd version
python register_model.py <RUN-ID-2> LinearRegressionModel
python register_model.py <RUN-ID-3> LinearRegressionModel
python register_model.py <RUN-ID-4> LinearRegressionModel

mlflow models serve -m "models:/LinearRegressionModel/3" -p 5001 &

## Ask for a prediction on a specific version of a specific model