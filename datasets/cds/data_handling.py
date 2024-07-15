import pandas as pd
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from utils.dataframe_utils import convert_to_datetime, split_date_and_hour, factorize_column, drop_columns, check_and_handle_missing_values, convert_columns_to_string

def preprocess_cds_df(cds_df: pd.DataFrame,  time_column: str = 'time') -> pd.DataFrame: 
    """
    Preprocess the CDS DataFrame by converting to datetime, handling missing values,
    splitting date and hour, factorizing the date, and dropping unnecessary columns.

    Parameters:
    cds_df (pd.DataFrame): The input DataFrame to preprocess.
    time_column (str): The name of the time column. Default is 'time'.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    cds_df = convert_to_datetime(cds_df, column=time_column)
    cds_df = check_and_handle_missing_values(cds_df, drop=True)
    cds_df = split_date_and_hour(cds_df, time_column, new_hour_col_name="hour_id")
    cds_df = factorize_column(cds_df, 'date', 'date_id')
    cds_df = drop_columns(cds_df, [time_column, 'date'])
    cds_df = convert_columns_to_string(cds_df, ["latitude", "longitude"])
    print(f"[INFO] Preprocess completed:\n{cds_df}")

    return cds_df

def create_cds_time_series_dataset(df: pd.DataFrame, training_cutoff: int, max_encoder_length: int, max_prediction_length: int, targets: list = ["tcc", "hcc", "mcc", "lcc", "tciw", "tclw"]) -> TimeSeriesDataSet:
    """
    Create a TimeSeriesDataSet for training.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    training_cutoff (int): The cutoff date_id for training.
    max_encoder_length (int): The maximum length of the encoder.
    max_prediction_length (int): The maximum length of the prediction.
    targets (list): A list of target variables to predict.

    Requirements:
    The data source must be from `cds`.
    The DataFrame must have 'date_id' as the time identifier column and 'hour_id' as its covariate.
    
    Returns:
    TimeSeriesDataSet: The created TimeSeriesDataSet.
    """
    return TimeSeriesDataSet(
        df[lambda x: x.date_id <= training_cutoff],
        time_idx="date_id",
        target=targets,  # Targets to predict
        group_ids=["latitude", "longitude"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["latitude", "longitude"],
        time_varying_known_reals=["date_id", "hour_id"],  # Known covariates
        time_varying_unknown_reals=targets,
        target_normalizer=MultiNormalizer([GroupNormalizer(groups=["latitude", "longitude"])] * len(targets)),
        allow_missing_timesteps=True,  # Allow missing timesteps
    )