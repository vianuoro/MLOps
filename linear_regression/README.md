## Train the model and start mlflow ui
python3 train_model.py
mlflow ui &
## list the trained model and serve it
ls mlruns/0/
mlflow models serve -m runs:/<RUN-ID-1>/model -p 1234 &

## USE API to make a prediction from two values
curl -X POST http://127.0.0.1:1234/invocations   -H "Content-Type: application/json"   -d '{
        "dataframe_split": {
          "columns": ["feature_0"],
          "data": [[0.5], [0.2]]
        }
      }'

## Make three more models and register them in mlflow registry
python3 train_model.py
ls mlruns/0/
python register_model.py <RUN-ID-2> LinearRegressionModel
python3 train_model.py
ls mlruns/0/
python register_model.py <RUN-ID-3> LinearRegressionModel
python3 train_model.py
ls mlruns/0/
python register_model.py <RUN-ID-4> LinearRegressionModel
Visit http://127.0.0.1:5000/#/models/LinearRegressionModel

## Serve the fourth version of the model
mlflow models serve -m "models:/LinearRegressionModel/4" -p 1235 &

## USE API to make a prediction from two values
curl -X POST http://127.0.0.1:1235/invocations   -H "Content-Type: application/json"   -d '{
        "dataframe_split": {
          "columns": ["feature_0"],
          "data": [[0.5], [0.2]]
        }
      }'

## Promote version 3 to production
python3 promote_model.py LinearRegressionModel 4 Production
