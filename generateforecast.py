import pandas as pd
import numpy as np
import os
import ast
import warnings
from datetime import datetime # Import datetime

try:
    from ibm_watsonx_ai import Credentials, APIClient
    from ibm_watsonx_ai.helpers import DataConnection, ContainerLocation
    from autoai_ts_libs.utils.ts_pipeline import TSPipeline
    from autoai_ts_libs.transforms.imputers import linear
    from autoai_ts_libs.srom.estimators.time_series.models.srom_estimators import FlattenAutoEnsembler
    from autoai_ts_libs.srom.estimators.regression.auto_ensemble_regressor import EnsembleRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.linear_model import LinearRegression, SGDRegressor
    from xgboost import XGBRegressor
    from autoai_ts_libs.srom.joint_optimizers.auto.auto_regression import AutoRegression
    from sklearn.pipeline import make_pipeline
    import autoai_ts_libs.srom.joint_optimizers.cv.time_series_splits
    import autoai_ts_libs.srom.joint_optimizers.pipeline.srom_param_grid
    import sklearn.metrics
    import autoai_ts_libs.srom.joint_optimizers.utils.no_op
    from autoai_ts_libs.utils.metrics import get_scorer
    from sklearn.base import BaseEstimator, MetaEstimatorMixin
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure ibm-watsonx-ai, scikit-learn, xgboost, autoai-ts-libs, and pandas are installed.")
    # raise # Consider raising error


# ==============================================================================
# Define the Wrapper Class (Unchanged from previous version)
# ==============================================================================
class ConstrainedPipelineWrapper(BaseEstimator, MetaEstimatorMixin):
    """
    A wrapper for a pipeline that applies constraints to prediction outputs.
    Specifically, forces the 'Rainfall' column to be non-negative.
    """
    def __init__(self, pipeline_to_wrap, target_columns_list):
        self.pipeline_to_wrap = pipeline_to_wrap
        self.target_columns = target_columns_list
        self._rainfall_col_index = -1 # Default to -1 (not found)
        try:
            self._rainfall_col_index = self.target_columns.index('Rainfall')
        except ValueError:
            print("[Wrapper] WARNING: 'Rainfall' not found in target_columns. Non-negative constraint cannot be applied.")
        except Exception as e:
            print(f"[Wrapper] Error finding Rainfall index: {e}")

    def fit(self, X, y=None, **kwargs):
        self.pipeline_to_wrap.fit(X, y, **kwargs)
        if hasattr(self.pipeline_to_wrap, 'classes_'): self.classes_ = self.pipeline_to_wrap.classes_
        if hasattr(self.pipeline_to_wrap, 'n_features_in_'): self.n_features_in_ = self.pipeline_to_wrap.n_features_in_
        self._is_fitted = True
        return self

    def predict(self, X=None, supporting_features=None, **kwargs):
        # Note: Even if supporting_features is None here, the underlying call might change
        # based on whether future exogenous features were expected during config/training.
        if supporting_features is not None:
             # This path might still be needed if predict is called with test data (X)
             # but the original pipeline was built expecting supporting_features.
             # However, for forecasting without future exogenous, supporting_features should be None.
             y_pred_original = self.pipeline_to_wrap.predict(supporting_features=supporting_features, **kwargs)
        elif X is not None:
             # Used when predicting on test data (X_test_f)
             y_pred_original = self.pipeline_to_wrap.predict(X=X, **kwargs)
        else:
             # Used for forecasting when no future exogenous features are provided
             y_pred_original = self.pipeline_to_wrap.predict(**kwargs)

        if self._rainfall_col_index != -1:
            y_pred_constrained = y_pred_original
            y_pred_constrained[:, self._rainfall_col_index] = np.maximum(y_pred_constrained[:, self._rainfall_col_index], 0)
            return y_pred_constrained
        else:
            return y_pred_original

    def score(self, X, y=None, sample_weight=None, **kwargs):
         if hasattr(self.pipeline_to_wrap, 'score'):
              score_kwargs = {}
              if sample_weight is not None: score_kwargs['sample_weight'] = sample_weight
              return self.pipeline_to_wrap.score(X, y, **score_kwargs)
         else:
              raise AttributeError("The underlying pipeline does not have a score method.")

    def __getattr__(self, name):
        if name in ['pipeline_to_wrap', '_rainfall_col_index', 'target_columns', '_is_fitted'] or name.startswith('_'):
             raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try: return getattr(self.pipeline_to_wrap, name)
        except AttributeError: raise AttributeError(f"'{type(self).__name__}' object (and its wrapped pipeline) has no attribute '{name}'")

    def get_params(self, deep=True):
        params = {'pipeline_to_wrap': self.pipeline_to_wrap, 'target_columns_list': self.target_columns}
        if deep and hasattr(self.pipeline_to_wrap, 'get_params'):
            for key, value in self.pipeline_to_wrap.get_params(deep=True).items(): params[f'pipeline_to_wrap__{key}'] = value
        return params

    def set_params(self, **params):
        pipeline_params = {}
        for key, value in params.items():
            if key == 'pipeline_to_wrap': self.pipeline_to_wrap = value
            elif key == 'target_columns_list':
                self.target_columns = value
                try: self._rainfall_col_index = self.target_columns.index('Rainfall')
                except ValueError: self._rainfall_col_index = -1
            elif key.startswith('pipeline_to_wrap__'): pipeline_params[key.split('__', 1)[1]] = value
            else: setattr(self, key, value)
        if pipeline_params and hasattr(self.pipeline_to_wrap, 'set_params'): self.pipeline_to_wrap.set_params(**pipeline_params)
        elif pipeline_params:
             for key, value in pipeline_params.items(): setattr(self.pipeline_to_wrap, key, value)
        return self


# ==============================================================================
# Main Forecasting Function (Modified)
# ==============================================================================
def generate_district_forecast(
    district_name: str,
    wml_api_key: str,
    # future_windspeed_data parameter removed
    data_asset_id: str = 'be8ecda5-314b-4a8f-9e3e-5cfc5c89542c',
    project_id: str = '53c9bbac-1ea0-4f25-960e-b5c2e2bb6129',
    deployment_url: str = 'https://us-south.ml.cloud.ibm.com',
    holdout_size: int = 20,
    datetime_format: str = '%d-%m-%Y %H:%M',
    timestamp_col: str = 'Timestamp',
    district_col: str = 'District',
    condition_col: str = 'Condition'
    ) -> pd.DataFrame | None:
    """
    Generates a weather forecast for a specific district using an AutoAI pipeline.
    This version does NOT require or use future exogenous features (e.g., wind speed).

    Args:
        district_name: The name of the district to filter data for and predict.
        wml_api_key: Your IBM Cloud API Key for Watson Machine Learning.
        data_asset_id: The ID of the WML data asset containing the training data.
        project_id: The ID of your WML project.
        deployment_url: The deployment URL for your WML instance.
        holdout_size: Number of latest records per district to use for testing (optional eval).
        datetime_format: The format string for parsing the timestamp column.
        timestamp_col: Name of the timestamp column in the data asset.
        district_col: Name of the district column in the data asset.
        condition_col: Name of the condition column (will be dropped).

    Returns:
        A pandas DataFrame containing the forecast (Temperature, Humidity, Rainfall)
        with a daily timestamp index starting from the current date, or None if an error occurs.
        Rainfall values are constrained to be non-negative.
    """
    print(f"--- Starting forecast generation for district: {district_name} (No Future Exogenous Features) ---")
    main_start_time = datetime.now()

    # --- Configuration & Metadata ---
    timestamp_col_name = timestamp_col
    district_col_name = district_col
    condition_col_name = condition_col
    api_key = wml_api_key

    training_data_references = [DataConnection(data_asset_id=data_asset_id)]
    training_result_reference = DataConnection(
        location=ContainerLocation(
            path=f'auto_ml/forecast_{district_name}_noexog/wml_data', # Example path
            model_location=f'auto_ml/forecast_{district_name}_noexog/wml_data/model.zip',
            training_status=f'auto_ml/forecast_{district_name}_noexog/training-status.json'
        )
    )

    experiment_metadata = dict(
        prediction_type='timeseries',
        prediction_columns=['Temperature', 'Humidity', 'Rainfall'],
        csv_separator=',',
        training_data_references=training_data_references,
        training_result_reference=training_result_reference,
        timestamp_column_name=timestamp_col_name,
        backtest_num=4,
        pipeline_type='customized',
         # Keep customized pipelines, AutoAI/libs might handle lack of exogenous data
        customized_pipelines=['MT2RForecaster', 'ExogenousMT2RForecaster', 'LocalizedFlattenEnsembler', 'DifferenceFlattenEnsembler', 'FlattenEnsembler', 'ExogenousLocalizedFlattenEnsembler', 'ExogenousDifferenceFlattenEnsembler', 'ExogenousFlattenEnsembler', 'ARIMA', 'ARIMAX', 'ARIMAX_RSAR', 'ARIMAX_PALR', 'ARIMAX_RAR', 'ARIMAX_DMLR', 'HoltWinterAdditive', 'HoltWinterMultiplicative', 'RandomForestRegressor', 'ExogenousRandomForestRegressor', 'SVM', 'ExogenousSVM'],
        lookback_window=31,
        forecast_window=14,
        max_num_daub_ensembles=3,
        # Keep WindSpeed_kph here, assuming it MIGHT be used as a lagged feature
        feature_columns=['Temperature', 'Humidity', 'Rainfall', 'WindSpeed_kph'],
        # *** KEY CHANGE HERE ***
        future_exogenous_available=False, # Set to False
        # **********************
        gap_len=0,
        deployment_url=deployment_url,
        project_id=project_id,
        numerical_imputation_strategy=['FlattenIterative', 'Linear', 'Cubic', 'Previous'],
        holdout_size=holdout_size
    )
    forecast_window_len = experiment_metadata['forecast_window'] # Still needed for output shape info

    # --- CPU Count ---
    CPU_NUMBER = 1
    if 'RUNTIME_HARDWARE_SPEC' in os.environ:
        try: CPU_NUMBER = int(ast.literal_eval(os.environ['RUNTIME_HARDWARE_SPEC'])['num_cpu'])
        except (KeyError, TypeError, ValueError): pass
    else:
        cpu_count_os = os.cpu_count()
        if cpu_count_os is not None: CPU_NUMBER = cpu_count_os
    print(f"Using CPU_NUMBER: {CPU_NUMBER}")

    # --- Initialize WML Client ---
    try:
        print("Initializing WML Client...")
        if not api_key or api_key == 'PUT_YOUR_APIKEY_HERE': warnings.warn("API key missing or using placeholder.")
        credentials = Credentials(api_key=api_key, url=experiment_metadata['deployment_url'])
        client = APIClient(credentials)
        client.set.default_project(experiment_metadata['project_id'])
        training_data_references[0].set_client(client)
        print("WML client configured successfully.")
    except Exception as e:
        print(f"Error initializing WML client: {e}")
        return None

    # --- Read, Filter, Split Data ---
    try:
        print("Reading and processing training data...")
        read_start = datetime.now()
        all_data_df = training_data_references[0].read(with_holdout_split=False, use_flight=True)
        print(f"Full data read successfully in {(datetime.now() - read_start).total_seconds():.2f} seconds.")

        print(f"Preprocessing and filtering for district: '{district_name}'...")
        filtered_df = all_data_df.copy()
        filtered_df[timestamp_col_name] = pd.to_datetime(filtered_df[timestamp_col_name], format=datetime_format)

        if district_col_name not in filtered_df.columns: raise ValueError(f"District column '{district_col_name}' not found.")
        filtered_df = filtered_df[filtered_df[district_col_name] == district_name].copy()

        if filtered_df.empty:
            print(f"ERROR: No data found for district '{district_name}'.")
            return None

        filtered_df = filtered_df.set_index(timestamp_col_name).sort_index()

        if filtered_df.index.duplicated().any():
            duplicates = filtered_df.index[filtered_df.index.duplicated()].unique()
            warnings.warn(f"Duplicate timestamps found for district '{district_name}' AFTER filtering: {duplicates}. Keeping first occurrence.")
            filtered_df = filtered_df[~filtered_df.index.duplicated(keep='first')]

        target_cols = experiment_metadata['prediction_columns']
        feature_cols = experiment_metadata['feature_columns'] # Still includes WindSpeed
        all_needed_cols = list(set(target_cols + feature_cols))

        missing_cols = [col for col in all_needed_cols if col not in filtered_df.columns]
        if missing_cols: raise ValueError(f"Missing required columns after filtering: {missing_cols}")
        filtered_df = filtered_df[all_needed_cols] # Keep WindSpeed column if present

        print(f"Manually splitting data (holdout_size = {holdout_size})...")
        if len(filtered_df) <= holdout_size:
             print(f"ERROR: Not enough data ({len(filtered_df)} rows) for district '{district_name}' to create test set size {holdout_size}.")
             return None

        test_df_f = filtered_df.iloc[-holdout_size:]
        train_df_f = filtered_df.iloc[:-holdout_size]

        X_train_f = train_df_f[feature_cols]
        y_train_f = train_df_f[target_cols]
        X_test_f = test_df_f[feature_cols]
        y_test_f = test_df_f[target_cols]

        print("Data filtered and split successfully.")
        print(f"Train shape: X={X_train_f.shape}, y={y_train_f.shape}")

    except Exception as e:
        print(f"Error reading or processing data: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- Define and Wrap Pipeline ---
    try:
        print("Defining pipeline structure...")
        define_start = datetime.now()
        # (Pipeline definition - ensure indices match feature_cols if it still includes WindSpeed)
        linear_imputer=linear(missing_val_identifier=float("nan"))
        linear_regression_ens=LinearRegression(n_jobs=CPU_NUMBER)
        multi_output_regressor_ens_lr=MultiOutputRegressor(estimator=linear_regression_ens, n_jobs=CPU_NUMBER)
        sgd_regressor_ens=SGDRegressor(early_stopping=True, random_state=0, tol=0.001)
        multi_output_regressor_ens_sgd=MultiOutputRegressor(estimator=sgd_regressor_ens, n_jobs=CPU_NUMBER)
        xgb_regressor_ens=XGBRegressor(objective="reg:squarederror", base_score=0.5, booster="gbtree", colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1, gamma=0, importance_type=None, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1, missing=float("nan"), reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1, n_jobs=CPU_NUMBER)
        multi_output_regressor_ens_xgb=MultiOutputRegressor(estimator=xgb_regressor_ens, n_jobs=CPU_NUMBER)
        linear_regression_auto=LinearRegression(n_jobs=CPU_NUMBER)
        multi_output_regressor_auto_lr=MultiOutputRegressor(estimator=linear_regression_auto, n_jobs=CPU_NUMBER)
        sgd_regressor_auto=SGDRegressor(early_stopping=True, random_state=0, tol=0.001)
        multi_output_regressor_auto_sgd=MultiOutputRegressor(estimator=sgd_regressor_auto, n_jobs=CPU_NUMBER)
        xgb_regressor_auto=XGBRegressor(objective="reg:squarederror", base_score=0.5, booster="gbtree", colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1, gamma=0, importance_type=None, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1, missing=float("nan"), reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1, n_jobs=CPU_NUMBER)
        multi_output_regressor_auto_xgb=MultiOutputRegressor(estimator=xgb_regressor_auto, n_jobs=CPU_NUMBER)
        linear_regression_best=LinearRegression(n_jobs=CPU_NUMBER)
        multi_output_regressor_best=MultiOutputRegressor(estimator=linear_regression_best, n_jobs=CPU_NUMBER)
        pipeline_best=make_pipeline(multi_output_regressor_best)
        auto_regression=AutoRegression(cv=autoai_ts_libs.srom.joint_optimizers.cv.time_series_splits.TimeSeriesTrainTestSplit(n_splits=1, n_test_size=582, overlap_len=0), execution_platform="single_node_complete_search", execution_round=1, execution_time_per_pipeline=3, level="default", num_option_per_pipeline_for_intelligent_search=30, num_options_per_pipeline_for_random_search=10, param_grid=autoai_ts_libs.srom.joint_optimizers.pipeline.srom_param_grid.SROMParamGrid(), save_prefix="auto_regression_output_", scoring=sklearn.metrics.make_scorer(sklearn.metrics.mean_absolute_error, greater_is_better=False), stages=[ [ ("skipscaling", autoai_ts_libs.srom.joint_optimizers.utils.no_op.NoOp()), ("minmaxscaler", MinMaxScaler()), ], [ ("molinearregression", multi_output_regressor_auto_lr), ("mosgdregressor", multi_output_regressor_auto_sgd), ("moxgbregressor", multi_output_regressor_auto_xgb), ], ], total_execution_time=3, best_estimator_so_far=pipeline_best)
        ensemble_regressor=EnsembleRegressor(aggr_type_for_pred_interval="median", bootstrap_for_pred_interval=True, cv=5, ensemble_type="voting", execution_platform="single_node_complete_search", execution_time_per_pipeline=3, level="default", max_samples_for_pred_interval=1.0, n_estimators_for_pred_interval=1, n_leaders_for_ensemble=1, num_option_per_pipeline_for_intelligent_search=30, num_options_per_pipeline_for_random_search=10, param_grid=None, prediction_percentile=95, save_prefix="auto_regression_output_", scoring=None, stages=[ [ ("skipscaling", autoai_ts_libs.srom.joint_optimizers.utils.no_op.NoOp()), ("minmaxscaler", MinMaxScaler()), ], [ ("molinearregression", multi_output_regressor_ens_lr), ("mosgdregressor", multi_output_regressor_ens_sgd), ("moxgbregressor", multi_output_regressor_ens_xgb), ], ], total_execution_time=3, auto_regression=auto_regression)

        # Indices based on feature_columns = ['Temperature', 'Humidity', 'Rainfall', 'WindSpeed_kph']
        fae_feature_indices = [0, 1, 2] # Indices of targets within feature_cols
        fae_target_indices = [0, 1, 2]  # Indices of targets within feature_cols
        # fae_exog_indices = [3] # Index of WindSpeed within feature_cols (no longer used as look_ahead)
        flatten_auto_ensembler=FlattenAutoEnsembler(
            feature_columns=fae_feature_indices, target_columns=fae_target_indices,
            lookback_win=experiment_metadata['lookback_window'],
            pred_win=experiment_metadata['forecast_window'],
            dag_granularity="multioutput_flat", data_transformation_scheme="log",
            execution_platform="single_node_complete_search", execution_time_per_pipeline=3,
            init_time_optimization=True,
            # *** KEY CHANGE HERE ***
            look_ahead_fcolumns=None, # No future exogenous features expected
            # **********************
            multistep_prediction_strategy="multioutput",
            multistep_prediction_win=experiment_metadata['forecast_window'],
            n_estimators_for_pred_interval=1, n_jobs=CPU_NUMBER, n_leaders_for_ensemble=1,
            store_lookback_history=True, total_execution_time=3, estimator=ensemble_regressor
        )

        tsp_feature_indices = list(range(len(feature_cols))) # [0, 1, 2, 3]
        tsp_target_indices = [feature_cols.index(col) for col in target_cols] # [0, 1, 2]

        pipeline_original = TSPipeline(
            steps=[ ("linear_imputer", linear_imputer),
                    ( "<class 'autoai_ts_libs.srom.estimators.time_series.models.srom_estimators.FlattenAutoEnsembler'>", flatten_auto_ensembler, ), ],
            # *** KEY CHANGE HERE ***
            exogenous_state_=None, # Set to None as no exogenous features are used this way
            # **********************
            feature_columns=tsp_feature_indices, # Still [0, 1, 2, 3] as X_train includes WindSpeed
            prediction_horizon=experiment_metadata['forecast_window'],
            target_columns=tsp_target_indices # Still [0, 1, 2]
        )

        print(f"Pipeline defined in {(datetime.now() - define_start).total_seconds():.2f} seconds.")

        # --- Wrap the pipeline ---
        print("Wrapping pipeline for constraint enforcement...")
        pipeline = ConstrainedPipelineWrapper(pipeline_original, target_cols)

    except Exception as e:
        print(f"Error defining or wrapping pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- Train Pipeline ---
    try:
        print(f"Fitting pipeline model for district: '{district_name}'...")
        fit_start = datetime.now()
        pipeline.fit(X_train_f.values, y_train_f.values)
        print(f"Pipeline fitted successfully in {(datetime.now() - fit_start).total_seconds():.2f} seconds.")
    except Exception as e:
        print(f"Error fitting pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- Generate Forecast ---
    try:
        print("\nGenerating future forecast...")
        forecast_start = datetime.now()
        # *** KEY CHANGE HERE ***
        # Call predict() without supporting_features when future_exogenous_available=False
        y_forecast = pipeline.predict()
        # **********************
        print(f"Generated future forecast (shape: {y_forecast.shape}) in {(datetime.now() - forecast_start).total_seconds():.2f} seconds.")

        # --- Format Forecast Output ---
        print("Formatting forecast output...")
        current_date = pd.Timestamp.now().normalize()
        forecast_index = pd.date_range(start=current_date,
                                       periods=y_forecast.shape[0],
                                       freq='D') # Daily frequency

        y_forecast_df = pd.DataFrame(y_forecast,
                                     columns=target_cols,
                                     index=forecast_index)

        # Constraint is handled by the wrapper during predict()

        total_time = (datetime.now() - main_start_time).total_seconds()
        print(f"--- Forecast generation for {district_name} completed successfully in {total_time:.2f} seconds ---")
        return y_forecast_df

    except Exception as e:
        print(f"Error during forecasting or formatting: {e}")
        import traceback
        traceback.print_exc()
        return None

