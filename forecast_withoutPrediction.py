import os
import ast
import pandas as pd
import numpy as np
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.helpers import ContainerLocation
from autoai_ts_libs.utils.ts_pipeline import TSPipeline
from autoai_ts_libs.transforms.imputers import linear
from autoai_ts_libs.srom.estimators.time_series.models.MT2RForecaster import MT2RForecaster
from autoai_ts_libs.utils.metrics import get_scorer

# Setup CPU info
CPU_NUMBER = 3
if 'RUNTIME_HARDWARE_SPEC' in os.environ:
    CPU_NUMBER = int(ast.literal_eval(os.environ['RUNTIME_HARDWARE_SPEC'])['num_cpu'])

# ✅ Your credentials
api_key = 'Xi6hhu-CK2M2L03CI811GAmRZbiXQ2GTzKSjVjA1GjxD'
deployment_url = 'https://us-south.ml.cloud.ibm.com'
project_id = '53c9bbac-1ea0-4f25-960e-b5c2e2bb6129'

# Setup API client
credentials = Credentials(api_key=api_key, url=deployment_url)
client = APIClient(credentials)
client.set.default_project(project_id)

# Define training data references
training_data_references = [
    DataConnection(data_asset_id='df369b9d-e98c-4aaa-aa43-f79c04ab6455')
]
training_data_references[0].set_client(client)

# Define result connection (dummy since we’re just loading)
training_result_reference = DataConnection(
    location=ContainerLocation(
        path='auto_ml/4ffd1419-06c9-4cc1-8846-c7f263e4bacd/wml_data/57c76016-7440-44ad-b9d7-7b51b1dc7629/data/autoai-ts',
        model_location='auto_ml/4ffd1419-06c9-4cc1-8846-c7f263e4bacd/wml_data/57c76016-7440-44ad-b9d7-7b51b1dc7629/data/autoai-ts/model.zip',
        training_status='auto_ml/4ffd1419-06c9-4cc1-8846-c7f263e4bacd/wml_data/57c76016-7440-44ad-b9d7-7b51b1dc7629/training-status.json'
    )
)

# Metadata required for reading data
experiment_metadata = dict(
    prediction_type='timeseries',
    prediction_columns=['Temperature', 'Humidity', 'Rainfall'],
    csv_separator=',',
    holdout_size=20,
    training_data_references=training_data_references,
    training_result_reference=training_result_reference,
    timestamp_column_name=-1,
    backtest_num=4,
    pipeline_type='customized',
    customized_pipelines=['MT2RForecaster'],
    lookback_window=60,  # ✅ Reduced for performance
    forecast_window=1,
    max_num_daub_ensembles=3,
    feature_columns=['Temperature', 'Humidity', 'Rainfall', 'WindSpeed_kph', 'Pressure_mb', 'FeelsLike_c'],
    future_exogenous_available=True,
    gap_len=0,
    deployment_url=deployment_url,
    project_id=project_id,
    numerical_imputation_strategy=['FlattenIterative', 'Linear', 'Cubic', 'Previous']
)

# Load training and holdout data
X_train, X_test, y_train, y_test = training_data_references[0].read(
    experiment_metadata=experiment_metadata,
    with_holdout_split=True,
    use_flight=True
)
print(f"Data read successfully. Training features shape: {X_train.shape}, Training targets shape: {y_train.shape}")
print(f"Test features shape: {X_test.shape}, Test targets shape: {y_test.shape}")
# ✅ Fix pandas fragmentation warning
X_train = X_train.copy()
y_train = y_train.copy()
X_test = X_test.copy()
y_test = y_test.copy()

# Add mocked district information
districts = ['Nagpur', 'Mumbai', 'Pune']
district_column_train = np.random.choice(districts, size=len(X_train))
district_column_test = np.random.choice(districts, size=len(X_test))

# Add the district column to your datasets
X_train['District'] = district_column_train
X_test['District'] = district_column_test
# Display first few rows of X_train and X_test to see the district mapping
print("\n🚦 X_train with District Column (first 10 rows):")
print(X_train.head(10))

print("\n🚦 X_test with District Column (first 10 rows):")
print(X_test.head(10))

# User Input: Ask the user to select a district
print("Please enter your district (e.g., Nagpur, Mumbai, Pune):")
user_district = input().strip()

# Validate user input
if user_district not in districts:
    print(f"Invalid district! Please choose from {', '.join(districts)}.")
    exit()

# Filter the data for the selected district
X_train_district = X_train[X_train['District'] == user_district]
X_test_district = X_test[X_test['District'] == user_district]
y_train_district = y_train[X_train['District'] == user_district]
y_test_district = y_test[X_test['District'] == user_district]

# Ensure we have data for the selected district
if X_train_district.empty or X_test_district.empty:
    print(f"No data available for the selected district: {user_district}.")
    exit()


# Setup forecasting pipeline
linear_imputer = linear(missing_val_identifier=float("nan"))
mt2_r_forecaster = MT2RForecaster(
    feature_columns=[0, 1, 2],
    lookback_win=60,
    n_jobs=1,
    prediction_win=1,
    target_columns=[0, 1, 2],
)
pipeline = TSPipeline(
    steps=[
        ("linear_imputer", linear_imputer),
        ("MT2RForecaster", mt2_r_forecaster),
    ],
    feature_columns=[0, 1, 2],  # Feature columns for training
    target_columns=[0, 1, 2],   # Target columns for forecasting
)

# Fit the pipeline using the district-specific data
pipeline.fit(X_train_district[['Temperature', 'Humidity', 'Rainfall', 'WindSpeed_kph', 'Pressure_mb', 'FeelsLike_c']].values, 
             y_train_district.values)

# Evaluate the model
scorer = get_scorer("neg_avg_symmetric_mean_absolute_percentage_error")
score = scorer(pipeline, X_test_district[['Temperature', 'Humidity', 'Rainfall', 'WindSpeed_kph', 'Pressure_mb', 'FeelsLike_c']].values, y_test_district.values)
print(f"\n✅ Test Score for {user_district}: {score}\n")

# Make predictions for the selected district
predictions = pipeline.predict(X_test_district[['Temperature', 'Humidity', 'Rainfall', 'WindSpeed_kph', 'Pressure_mb', 'FeelsLike_c']].values)

# Convert predictions into a DataFrame with proper column headers
predictions_df = pd.DataFrame(predictions, columns=['Predicted_Temperature', 'Predicted_Humidity', 'Predicted_Rainfall'])

# Print predictions in readable format
print(f"\n📈 Predictions for {user_district}:\n{predictions_df.head()}")

# Forecast next future step (no X provided, will continue from last point)
forecast = pipeline.predict()
forecast_df = pd.DataFrame(forecast, columns=['Predicted_Temperature', 'Predicted_Humidity', 'Predicted_Rainfall'])
print(f"\n📊 Forecast for the next time step for {user_district}:\n{forecast_df}")
