python3 train_model.py
mlflow ui &
ls mlruns/0/
mlflow models serve -m runs:/<RUN-ID>/model -p 1234 &
curl -X POST http://127.0.0.1:1234/invocations   -H "Content-Type: application/json"   -d '{
        "dataframe_split": {
          "columns": ["feature_0"],
          "data": [[0.5], [0.2]]
        }
      }'

python register_model.py <RUN-ID> LinearRegressionModel
