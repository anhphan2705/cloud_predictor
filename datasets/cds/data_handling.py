import pandas as pd
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting import TimeSeriesDataSet
from utils.dataframe_utils import convert_to_datetime, factorize_column, drop_columns, check_and_handle_missing_values, convert_columns_to_string

def preprocess_cds_df(cds_df: pd.DataFrame, time_column: str = 'time') -> pd.DataFrame: 
    """
    Preprocess the CDS DataFrame by converting to datetime, handling missing values,
    creating a combined time index, and dropping unnecessary columns.

    Parameters:
    cds_df (pd.DataFrame): The input DataFrame to preprocess.
    time_column (str): The name of the time column. Default is 'time'.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    cds_df = convert_to_datetime(cds_df, column=time_column)
    cds_df = check_and_handle_missing_values(cds_df, drop=True)
    cds_df = factorize_column(cds_df, column=time_column, new_column='time_idx')
    cds_df = drop_columns(cds_df, [time_column])
    cds_df = convert_columns_to_string(cds_df, ["latitude", "longitude"])
    print(f"[INFO] Preprocess completed:\n{cds_df}")

    return cds_df

def create_cds_time_series_datasets(df: pd.DataFrame, min_encoder_length: int, max_encoder_length: int, min_prediction_length: int, max_prediction_length: int, targets: list, mode: str = 'train'):
    """
    Create TimeSeriesDataSet for both training and validation or evaluation.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    max_encoder_length (int): The maximum length of the encoder.
    max_prediction_length (int): The maximum length of the prediction.
    targets (list): A list of target variables to predict.
    min_prediction_length (int): The minimum length of the prediction.
    mode (str): Mode of operation - 'train' or 'eval'.

    Returns:
    tuple: A tuple containing the training and validation TimeSeriesDataSets, or a single evaluation dataset.

    Example Usage:
    train_dataset, val_dataset = create_cds_time_series_datasets(df, max_encoder_length=365, max_prediction_length=365, targets=['tcc', 'hcc'])
    eval_dataset = create_cds_time_series_datasets(df, max_encoder_length=365, max_prediction_length=365, targets=['tcc', 'hcc'], mode='eval')
    """
    print(f'[INFO] Creating TimeSeriesDataSet for {mode} mode...')

    common_params = {
        'time_idx': "time_idx",
        'target': targets,  # Targets to predict
        'group_ids': ["latitude", "longitude"],
        'min_encoder_length': min_encoder_length,
        'max_encoder_length': max_encoder_length,
        'min_prediction_length': min_prediction_length,
        'max_prediction_length': max_prediction_length,
        'static_categoricals': ["latitude", "longitude"],
        'time_varying_known_reals': ["time_idx"],  # Known covariates
        'time_varying_unknown_reals': targets,
        'target_normalizer': MultiNormalizer([GroupNormalizer(groups=["latitude", "longitude"])] * len(targets)),
        'allow_missing_timesteps': False,  # Allow missing timesteps
    }

    if mode == 'train':
        print(f'[DEBUG] time_idx range: ({df["time_idx"].min()}, {df["time_idx"].max()})')
        training_cutoff = df["time_idx"].max() - max_prediction_length
        print(f'[DEBUG] Training cutoff at time_idx: {training_cutoff}')

        training_df = df[df["time_idx"] <= training_cutoff]
        validation_df = df[df["time_idx"] > training_cutoff]
        print(f'[DEBUG] training_df size: {len(training_df)}')
        print(f'[DEBUG] validation_df size: {len(validation_df)}')

        training_dataset = TimeSeriesDataSet(training_df, **common_params)
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            validation_df,
            predict=True,
            stop_randomization=True
        )

        return training_dataset, validation_dataset

    elif mode == 'eval':
        eval_dataset = TimeSeriesDataSet(df, **common_params)

        return eval_dataset

    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose either 'train' or 'eval'.")