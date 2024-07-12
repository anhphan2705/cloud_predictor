import glob
import xarray as xr
import pandas as pd
import os

def get_file_paths(dir):
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

def load_datasets(data_paths):
    """
    Loads datasets from yearly files.
    
    Parameters:
    data_paths (list): A list of file paths to the yearly data files.
    
    Returns:
    list: A list of xarray datasets.
    """
    if not data_paths:
        raise ValueError("[INFO] The data file path is empty.")
    
    datasets = []
    for path in data_paths:
        ds = xr.open_dataset(path)
        datasets.append(ds)
        print(f"[INFO] Loaded dataset time range: {ds.time.min().values} to {ds.time.max().values}")
    
    return datasets

def concatenate_datasets(datasets, dim='time'):
    """
    Concatenates a list of datasets along the specified dimension.
    
    Parameters:
    datasets (list): A list of xarray datasets to concatenate.
    dim (str): The dimension along which to concatenate the datasets.
    
    Returns:
    xr.Dataset: The concatenated dataset.
    """
    combined_ds = xr.concat(datasets, dim=dim)
    return combined_ds

def save_dataset(dataset, save_dir, filename='combined_dataset.nc'):
    """
    Saves the combined dataset to the specified directory.
    
    Parameters:
    dataset (xr.Dataset): The combined dataset to save.
    save_dir (str): The directory path where the combined dataset should be saved.
    filename (str): The name of the file to save the dataset as. Default is 'combined_dataset.nc'.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, filename)
    dataset.to_netcdf(file_path)
    print(f"[INFO] Dataset saved to {file_path}")

def get_combined_dataset(data_root: str, save_dir: str = '') -> xr.Dataset:
    """
    Loads datasets from yearly files, concatenates them along the time dimension, 
    and optionally saves the combined dataset to a specified directory.
    
    Parameters:
    data_root (str): The root directory path where the yearly data files are stored.
    save_dir (str, optional): The directory path where the combined dataset should be saved. 
                              If empty, the dataset is not saved. Default is ''.
    
    Returns:
    xr.Dataset: The combined dataset concatenated along the time dimension.
    """
    data_paths = get_file_paths(data_root)
    datasets = load_datasets(data_paths)
    
    if len(datasets) > 1:
        datasets = concatenate_datasets(datasets, dim='time')
    
    if save_dir:
        save_dataset(datasets, save_dir)

    return datasets

def convert_to_dataframe(datasets: xr.Dataset, variables: list = None) -> pd.DataFrame:
    """
    Converts a combined xarray Dataset to a pandas DataFrame, including specified variables.
    
    Parameters:
    datasets (xr.Dataset): The combined xarray Dataset.
    variables (list, optional): A list of variable names to include in the DataFrame. 
                                If None, include all variables in the dataset.
    
    Returns:
    pd.DataFrame: A DataFrame containing the specified variables, with the index reset.
    """
    if not variables:
        variables = list(datasets.data_vars)
    
    df = datasets[variables].to_dataframe().reset_index()
    df['time'] = pd.to_datetime(df['time'])
    print(f"[INFO] Converted dataset to DataFrame:\n{df.count()}")
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