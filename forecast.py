import os
import ast
import pandas as pd
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

# ✅ Add district column to training and test data
district_column = ['Nagpur', 'Pune', 'Mumbai', 'Nagpur', 'Mumbai', 'Pune', 'Nagpur', 'Mumbai', 'Pune', 'Nagpur'] * 30000
X_train['District'] = district_column[:len(X_train)]
X_test['District'] = district_column[:len(X_test)]

# Display a portion of the data to verify
print("\nSample of X_train with District column added:")
print(X_train.head())

# Function to train a model for a specific district
def train_model_for_district(district_data, target_data, district_name):
    print(f"\nTraining model for {district_name}...")

    # Setup imputer and forecaster
    linear_imputer = linear(missing_val_identifier=float("nan"))
    mt2_r_forecaster = MT2RForecaster(
        feature_columns=[0, 1, 2],  # Adjust as necessary
        lookback_win=60,
        n_jobs=1,
        prediction_win=1,
        target_columns=[0, 1, 2],  # Adjust as necessary
    )
    
    # Create a pipeline
    pipeline = TSPipeline(
        steps=[("linear_imputer", linear_imputer), ("MT2RForecaster", mt2_r_forecaster)],
        feature_columns=[0, 1, 2],  # Adjust as necessary
        target_columns=[0, 1, 2],   # Adjust as necessary
    )
    
    # Fit the model for the district
    pipeline.fit(district_data.values, target_data.values)
    
    print(f"Model for {district_name} trained successfully.")
    
    return pipeline

# Ask user for district input
district_input = input("Enter a district (Nagpur, Mumbai, Pune): ").strip()

# Check if input is valid
valid_districts = ['Nagpur', 'Mumbai', 'Pune']
if district_input not in valid_districts:
    print(f"Invalid district. Please choose from {valid_districts}.")
else:
    # Train separate models for each district
    print("\nTraining models for each district...")

    pune_data = X_train[X_train['District'] == 'Pune']
    mumbai_data = X_train[X_train['District'] == 'Mumbai']
    nagpur_data = X_train[X_train['District'] == 'Nagpur']

    pune_model = train_model_for_district(pune_data, y_train[X_train['District'] == 'Pune'], 'Pune')
    mumbai_model = train_model_for_district(mumbai_data, y_train[X_train['District'] == 'Mumbai'], 'Mumbai')
    nagpur_model = train_model_for_district(nagpur_data, y_train[X_train['District'] == 'Nagpur'], 'Nagpur')

    # Make predictions for the selected district
    if district_input == 'Pune':
        predictions = pune_model.predict(X_test[X_test['District'] == 'Pune'].values)
        print("\nPune Predictions:")
    elif district_input == 'Mumbai':
        predictions = mumbai_model.predict(X_test[X_test['District'] == 'Mumbai'].values)
        print("\nMumbai Predictions:")
    else:
        predictions = nagpur_model.predict(X_test[X_test['District'] == 'Nagpur'].values)
        print("\nNagpur Predictions:")

    # Print predictions in readable format
    print(pd.DataFrame(predictions, columns=['Temperature', 'Humidity', 'Rainfall']))

    # Evaluate the model performance on selected district
    scorer = get_scorer("neg_avg_symmetric_mean_absolute_percentage_error")

    if district_input == 'Pune':
        score = scorer(pune_model, X_test[X_test['District'] == 'Pune'].values, y_test[X_test['District'] == 'Pune'].values)
    elif district_input == 'Mumbai':
        score = scorer(mumbai_model, X_test[X_test['District'] == 'Mumbai'].values, y_test[X_test['District'] == 'Mumbai'].values)
    else:
        score = scorer(nagpur_model, X_test[X_test['District'] == 'Nagpur'].values, y_test[X_test['District'] == 'Nagpur'].values)

    print(f"\n{district_input} Test Score: {score}")
