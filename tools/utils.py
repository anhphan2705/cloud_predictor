import glob
import xarray as xr
import pandas as pd

def get_files(dir):
    """
    Retrieves a list of files matching the specified directory pattern.
    
    Parameters:
    dir (str): The directory pattern to search for files (e.g., 'data/*.nc').
    
    Returns:
    list: A list of file paths that match the specified directory pattern.
    """
    files = glob.glob(dir)
    print(f"[INFO] Found {len(files)} files in {dir}")
    return files

def df_visualizer(df):
    """
    Prints the first few rows of a DataFrame for visualization purposes.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to visualize.
    """
    print(f"[INFO] Visualizing dataframe...\n{df.head()}")

def get_col_count(df):
    """
    Prints and returns the count of non-missing values in each column of a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame for which to count non-missing values.
    
    Returns:
    pd.Series: A Series containing the count of non-missing values for each column.
    """
    print(f"[INFO] Count of non-missing values in each column:\n{df.count()}")
    return df.count()

def get_dataset(data_files, years, save_dir=''):
    """
    Loads datasets from yearly files, concatenates them along the time dimension, 
    and optionally saves the combined dataset to a specified directory.
    
    Parameters:
    data_files (list): A list of file paths to the yearly data files.
    years (list): A list of years corresponding to the data files.
    save_dir (str, optional): The directory path where the combined dataset should be saved. 
                              If empty, the dataset is not saved. Default is ''.
    
    Returns:
    xr.Dataset: The combined dataset concatenated along the time dimension.
    """
    datasets = []
    
    for file, year in zip(data_files, years):
        ds = xr.open_dataset(file)
        datasets.append(ds)

    combined_ds = xr.concat(datasets, dim='time')

    if save_dir:
        print(f"[INFO] Saving combined dataset to {save_dir}")
        combined_ds.to_netcdf(f'{save_dir}')
    
    return combined_ds

def convert_to_dataframe(datasets, variables):
    """
    Converts a combined xarray Dataset to a pandas DataFrame, including specified variables.
    
    Parameters:
    datasets (xr.Dataset): The combined xarray Dataset.
    variables (list): A list of variable names to include in the DataFrame.
    
    Returns:
    pd.DataFrame: A DataFrame containing the specified variables, with the index reset.
    """
    df = datasets[variables].to_dataframe().reset_index()
    df['time'] = pd.to_datetime(df['time'])
    print(f"[INFO] Converted dataset to DataFrame with shape: {df.shape}")
    return df

def save_to_csv(df, save_dir):
    """
    Saves a DataFrame to a CSV file.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    save_dir (str): The directory path where the CSV file should be saved.
    """
    df.to_csv(f'{save_dir}', index=False)
    print(f"[INFO] Saved DataFrame to {save_dir}")