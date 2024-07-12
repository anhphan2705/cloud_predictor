import pandas as pd

from utils.dataset_handling import get_combined_dataset
from utils.dataframe_handling import convert_to_dataframe, convert_to_datetime, split_date_and_hour, factorize_column, drop_columns, check_and_handle_missing_values

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
    print(f"[INFO] Preprocess completed:\n{cds_df}")

    return cds_df

def data_pipeline(data_root: str, data_source: str = 'cds', target_vars: list = [], time_column: str = 'time', save_dir: str = ''):
    """
    Execute the data pipeline by loading, preprocessing, and saving the data.

    Parameters:
    data_root (str): The directory pattern to search for files (e.g., 'data/*.nc').
    data_source (str): The source of the data. Default is 'cds'.
    target_vars (list): The list of target variables to include in the DataFrame. Default is empty list.
    time_column (str): The name of the time column. Default is 'time'.
    save_dir (str): The directory to save the preprocessed data. Default is empty string.
    """
    # Load the data
    ds = get_combined_dataset(data_root)
    df = convert_to_dataframe(ds, variables=target_vars)
    print("[INFO] Data loaded successfully.")
    
    # Preprocess the data
    if data_source == 'cds':
        df = preprocess_cds_df(df, time_column)
    else:
        raise ValueError(f"[INFO] Data source {data_source} is not supported.")
    
    # Save the preprocessed data
    if save_dir:
        save_to_csv(df, save_dir)