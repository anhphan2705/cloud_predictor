import pandas as pd
import xarray as xr
import numpy as np

def convert_to_dataframe(datasets: xr.Dataset, variables: list = None) -> pd.DataFrame:
    """
    Converts a combined xarray Dataset to a pandas DataFrame, including specified variables.

    Parameters:
    datasets (xr.Dataset): The combined xarray Dataset.
    variables (list, optional): A list of variable names to include in the DataFrame. 
                                If None, include all variables in the dataset.

    Usage:
    df = convert_to_dataframe(combined_ds, variables=['tcc', 'hcc'])

    Returns:
    pd.DataFrame: A DataFrame containing the specified variables, with the index reset.
    """
    if not variables:
        variables = list(datasets.data_vars)
    
    df = datasets[variables].to_dataframe().reset_index()
    print(f"[INFO] Converted dataset to DataFrame. Showing in format [column title] [row counts]:\n{df.count()}")
    return df

def convert_to_datetime(df: pd.DataFrame, column: str = 'time', format: str = None) -> pd.DataFrame:
    """
    Converts a specified column in a DataFrame to datetime format.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to convert.
    column (str): Optional. The name of the column to convert to datetime.
    format (str): Optional. The format to use for conversion. If None, it will infer the format.

    Usage:
    df = convert_to_datetime(df, column='time', format='%Y-%m-%d %H:%M:%S')

    Returns:
    pd.DataFrame: The DataFrame with the specified column converted to datetime.
    """
    try:
        if format:
            df[column] = pd.to_datetime(df[column], format=format, errors='coerce')
        else:
            df[column] = pd.to_datetime(df[column], errors='coerce')
        
        successful_conversions = df[column].notna().sum()
        failed_conversions = df[column].isna().sum()
        
        print(f"[INFO] Successfully converted {successful_conversions} entries to datetime format.")
        if failed_conversions > 0:
            print(f"[WARNING] Failed to convert {failed_conversions} entries to datetime format.")
        
    except Exception as e:
        print(f"[ERROR] An error occurred while converting column '{column}' to datetime: {e}")
    
    return df

def split_date_and_hour(df: pd.DataFrame, time_column: str, new_hour_col_name: str = 'hour', new_date_col_name: str = 'date') -> pd.DataFrame:
    """
    Split and add the 'date' and 'hour' from 'time' column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    time_column (str): The name of the time column to extract 'date' and 'hour' from.

    Usage:
    df = split_date_and_hour(df, time_column='time')

    Returns:
    pd.DataFrame: The DataFrame with 'date' and 'hour' columns added.
    """
    df = df.copy()
    df.loc[:, new_date_col_name] = df[time_column].dt.date
    df.loc[:, new_hour_col_name] = df[time_column].dt.hour

    print(f"[INFO] Extracted 'date' and 'hour' from '{time_column}'.")
    return df

def factorize_column(df: pd.DataFrame, column: str, new_column: str) -> pd.DataFrame:
    """
    Factorizes a specified column and creates a new column with incrementing count.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    column (str): The name of the column to factorize.
    new_column (str): The name of the new column to store the incrementing count.

    Usage:
    df = factorize_column(df, column='date', new_column='date_id')

    Returns:
    pd.DataFrame: The DataFrame with the new factorized column added.
    """
    df = df.copy()
    df.loc[:, new_column] = pd.factorize(df[column])[0]
    print(f"[INFO] Factorized column '{column}' into '{new_column}' with incrementing count.")
    return df

def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drops the specified column(s) from the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    columns (list): A list of column names to drop.

    Usage:
    df = drop_columns(df, columns=['date', 'hour'])

    Returns:
    pd.DataFrame: The DataFrame with the specified column(s) dropped.
    """
    df = df.drop(columns, axis=1)
    print(f"[INFO] Dropped columns: {columns}")
    return df

def drop_duplicates_and_reset_index(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Drops duplicate rows based on specified columns and resets the index.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    subset (list): Optional. A list of column names to consider for identifying duplicates.
                   If None, considers all columns.

    Usage:
    df = drop_duplicates_and_reset_index(df, subset=['time'])

    Returns:
    pd.DataFrame: The DataFrame with duplicates dropped and index reset.
    """
    if subset:
        df = df.drop_duplicates(subset=subset, ignore_index=True)
    else:
        df = df.drop_duplicates(ignore_index=True)
    
    print(f"[INFO] Dropped duplicates if there was any. Remaining rows: {len(df)}")
    return df

def check_and_handle_missing_values(df: pd.DataFrame, drop: bool = False) -> pd.DataFrame:
    """
    Checks for missing values in a DataFrame and optionally drops rows with missing values.

    Parameters:
    df (pd.DataFrame): The DataFrame to check for missing values.
    drop (bool): If True, drops all rows with missing values. Default is False.

    Usage:
    df = check_and_handle_missing_values(df, drop=True)

    Returns:
    pd.DataFrame: The DataFrame after handling missing values.
    """
    missing_info = df.isnull().sum()
    total_missing = missing_info.sum()
    
    if total_missing > 0:
        print(f"[INFO] DataFrame has {total_missing} missing values.")
        print(missing_info[missing_info > 0])
        
        if drop:
            df = df.dropna()
            print("[INFO] Dropped all rows with missing values.")
        else:
            print("[INFO] Missing values were not dropped.")
    else:
        print("[INFO] No missing values in the DataFrame.")
    
    return df

def consistency_check(df: pd.DataFrame):
    """
    Checks for any missing combinations of 'date_id' and 'hour_id' in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to check for consistency.

    Usage:
    consistency_check(df)
    """
    expected_hour_ids = set(np.arange(0, 24, 2))
    unique_dates = df['date_id'].unique()
    grouped_df = df.groupby('date_id')['hour_id'].apply(set)
    
    for date in unique_dates:
        actual_hour_ids = grouped_df.get(date, set())
        missing_hour_ids = expected_hour_ids - actual_hour_ids
        if missing_hour_ids:
            print(f"Missing hour_id(s) for date_id {date}: {missing_hour_ids}")

def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, on: str, how: str = 'inner') -> pd.DataFrame:
    """
    Merges two DataFrames on a specified column.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    on (str): The column name to merge on.
    how (str): How to perform the merge. Default is 'inner'. Options include 'left', 'right', 'outer', 'inner'.

    Usage:
    merged_df = merge_dataframes(df1, df2, on='time', how='inner')

    Returns:
    pd.DataFrame: The merged DataFrame.
    """
    merged_df = pd.merge(df1, df2, on=on, how=how)
    print(f"[INFO] Merged DataFrames on column '{on}' using '{how}' method.")
    return merged_df