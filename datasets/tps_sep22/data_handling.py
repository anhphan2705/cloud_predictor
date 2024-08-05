import pandas as pd
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting import TimeSeriesDataSet
from utils.dataframe_utils import convert_to_datetime, factorize_column, drop_columns, check_and_handle_missing_values, consistency_check, convert_columns_to_string, add_cyclical_calendar_features, add_weekend_feature, add_holidays_feature, add_end_of_year_holidays, convert_columns_to_float

def preprocess_tpssep22_df(df: pd.DataFrame, calendar_cycle: dict, target_columns: list, time_column: str = 'date') -> pd.DataFrame:
    """
    Preprocess the TPS SEP22 DataFrame by converting to datetime, handling missing values,
    creating a combined time index, adding cyclical calendar features, and dropping unnecessary columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame to preprocess.
    calendar_cycle (dict): Dictionary containing the cyclical calendar features.
    target_columns (list): List of target column names to convert to float.
    time_column (str): The name of the time column. Default is 'time'.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    df = convert_to_datetime(df, column=time_column)
    df = check_and_handle_missing_values(df, drop=True)
    df = add_cyclical_calendar_features(df, calendar_cycle, time_column)
    df = add_weekend_feature(df, time_column, count_friday=True)
    df = add_holidays_feature(df, time_column, 'country')
    df = add_end_of_year_holidays(df, time_column)
    df = convert_columns_to_float(df, target_columns)
    df = factorize_column(df, column=time_column, new_column='time_idx')
    df = drop_columns(df, [time_column, 'row_id'])
    consistency_check(df)

    print(f"[INFO] Preprocess completed:\n{df}")

    return df

def create_tpssep22_time_series_datasets(df: pd.DataFrame, time_series_config: dict,  mode: str = 'train'):
    """
    Create TimeSeriesDataSet for both training and validation or evaluation.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    time_series_config (dict): Dictionary containing time series configuration parameters.
    mode (str): Mode of operation - 'train' or 'eval'.

    Returns:
    tuple: A tuple containing the training and validation TimeSeriesDataSets, or a single evaluation dataset.

    Example Usage:
    train_dataset, val_dataset = create_cds_time_series_datasets(df, max_encoder_length=365, max_prediction_length=365, targets=['tcc', 'hcc'])
    eval_dataset = create_cds_time_series_datasets(df, max_encoder_length=365, max_prediction_length=365, targets=['tcc', 'hcc'], mode='eval')
    """
    print(f'[INFO] Creating TimeSeriesDataSet for {mode} mode...')

    max_encoder_length = time_series_config['max_encoder_length']
    max_prediction_length = time_series_config['max_prediction_length']
    min_prediction_length = time_series_config['min_prediction_length']
    min_encoder_length = time_series_config['min_encoder_length']
    targets = time_series_config['target_vars']
    groups = time_series_config['groups']
    static_categoricals = time_series_config['static_categoricals']
    time_varying_known_reals = time_series_config['time_varying_known_reals']
    lags = time_series_config['lags']
    allow_missing_timesteps = time_series_config['allow_missing_timesteps']
    add_relative_time_idx = time_series_config['add_relative_time_idx']
    add_target_scales = time_series_config['add_target_scales']
    add_encoder_length = time_series_config['add_encoder_length']

    common_params = {
        'time_idx': "time_idx",
        'target': targets[0] if len(targets) == 1 else targets,
        'group_ids': groups,
        'min_encoder_length': min_encoder_length,
        'max_encoder_length': max_encoder_length,
        'min_prediction_length': min_prediction_length,
        'max_prediction_length': max_prediction_length,
        'static_categoricals': static_categoricals,
        'time_varying_known_reals': time_varying_known_reals,
        'time_varying_unknown_reals': targets,
        'lags': lags,
        'target_normalizer': GroupNormalizer(groups=groups, transformation="softplus") if len(targets) == 1 else MultiNormalizer([GroupNormalizer(groups=groups)] * len(targets)),
        'allow_missing_timesteps': allow_missing_timesteps,
        'add_relative_time_idx': add_relative_time_idx,
        'add_target_scales': add_target_scales,
        'add_encoder_length': add_encoder_length,
    }

    print(f"[DEBUG] TSD Params:\n{common_params}")

    if mode == 'train':
        training_cutoff = df["time_idx"].max() - max_prediction_length
        print(f'[DEBUG] Training cutoff at time_idx: {training_cutoff}')

        training_df = df[df["time_idx"] <= training_cutoff]
        print(f'[DEBUG] training_df size: {len(training_df)}')
        
        print("[ADVICE] You should go get a coffee...")
        training_dataset = TimeSeriesDataSet(training_df, **common_params)
        print(f'[INFO] Training dataset created.')
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            df,
            predict=True,
            stop_randomization=True
        )
        print(f'[INFO] Validation dataset created.')

        return training_dataset, validation_dataset

    elif mode == 'eval':
        eval_dataset = TimeSeriesDataSet(df, **common_params, predict_mode=True, stop_randomization=True)
        print(f'[INFO] Evaluation dataset created.')

        return None, eval_dataset

    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose either 'train' or 'eval'.")