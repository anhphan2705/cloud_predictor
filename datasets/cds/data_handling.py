import pandas as pd
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting import TimeSeriesDataSet
from utils.dataframe_utils import convert_to_datetime, factorize_column, drop_columns, check_and_handle_missing_values, consistency_check, convert_columns_to_string, add_cyclical_calendar_features

def filter_dataframe(
    df: pd.DataFrame,
    lat_range: list = None, 
    long_range: list = None,
    time_range: tuple = None,
) -> pd.DataFrame:
    """
    Filter the DataFrame based on multiple criteria including latitude ranges, longitude ranges, and time range.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    lat_range (list[float, float], optional): Latitude range to include as a tuple (min_lat, max_lat). Default is None.
    long_range (list[float, float], optional): Longitude range to include as a tuple (min_long, max_long). Default is None.
    time_range (list[str, str], optional): The time range to include in format ('YYYY-MM-DD', 'YYYY-MM-DD'). Default is None.
    
    Returns:
    pd.DataFrame: The filtered DataFrame containing only the rows that meet all specified criteria.
    """
    # Convert 'time' column to datetime if it exists in the DataFrame
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    # Apply latitude range filter
    if lat_range:
        df = df[(df['latitude'] >= lat_range[0]) & (df['latitude'] <= lat_range[1])]
    
    # Apply longitude range filter
    if long_range:
        df = df[(df['longitude'] >= long_range[0]) & (df['longitude'] <= long_range[1])]
    
    # Apply time range filter
    if time_range:
        start_time, end_time = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
        df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    
    return df.reset_index(drop=True)

def preprocess_cds_df(cds_df: pd.DataFrame, latitude_range: list, longtitude_range: list, time_range:list, calendar_cycle: dict, time_column: str = 'time') -> pd.DataFrame: 
    """
    Preprocess the CDS DataFrame by converting to datetime, handling missing values,
    creating a combined time index, and dropping unnecessary columns.

    Parameters:
    cds_df (pd.DataFrame): The input DataFrame to preprocess.
    latitude_range (list): The latitude range to filter the data.
    longtitude_range (list): The longitude range to filter the data.
    time_range (list): The time range to filter the data.
    calendar_cycle (dict): Dictionary containing the cyclical calendar features.
    time_column (str): The name of the time column. Default is 'time'.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    cds_df = convert_to_datetime(cds_df, column=time_column)
    cds_df = filter_dataframe(cds_df, lat_range=latitude_range, long_range=longtitude_range, time_range=time_range)
    cds_df = check_and_handle_missing_values(cds_df, drop=True)
    cds_df = add_cyclical_calendar_features(cds_df, calendar_cycle, time_column)
    cds_df = factorize_column(cds_df, column=time_column, new_column='time_idx')
    cds_df = drop_columns(cds_df, [time_column])
    cds_df = convert_columns_to_string(cds_df, ["latitude", "longitude"])
    consistency_check(cds_df)

    print(f"[INFO] Preprocess completed:\n{cds_df}")

    return cds_df

def create_cds_time_series_datasets(df: pd.DataFrame, time_series_config: dict,  mode: str = 'train'):
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