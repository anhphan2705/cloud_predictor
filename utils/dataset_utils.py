import xarray as xr
import pandas as pd
import os
from utils.file_utils import get_file_paths
from typing import List, Union

def load_datasets(data_paths: List[str]) -> List[Union[xr.Dataset, pd.DataFrame]]:
    """
    Loads datasets from the provided file paths.

    Parameters:
    data_paths (List[str]): A list of file paths to the data files.

    Usage:
    datasets = load_datasets(['data/2001.nc', 'data/2002.nc', 'data/data.csv'])

    Returns:
    List[Union[xr.Dataset, pd.DataFrame]]: A list of xarray datasets or pandas dataframes.
    """
    if not data_paths:
        raise ValueError("[INFO] The data file path is empty.")
    
    datasets = []
    for path in data_paths:
        if path.endswith('.nc'):
            ds = xr.open_dataset(path)
            datasets.append(ds)
            print(f"[INFO] Loaded dataset time range: {ds.time.min().values} to {ds.time.max().values}")
        elif path.endswith('.csv'):
            df = pd.read_csv(path)
            datasets.append(df)
            print(f"[INFO] Loaded CSV file: {path}")
        else:
            print(f"[WARNING] Unsupported file format for path: {path}")
    
    return datasets

def concatenate_datasets(datasets: List[Union[xr.Dataset, pd.DataFrame]], dim: str = 'time') -> Union[xr.Dataset, pd.DataFrame]:
    """
    Concatenates a list of datasets along the specified dimension.

    Parameters:
    datasets (List[Union[xr.Dataset, pd.DataFrame]]): A list of datasets to concatenate.
    dim (str): The dimension along which to concatenate the datasets.

    Usage:
    combined_ds = concatenate_datasets(datasets, dim='time')

    Returns:
    Union[xr.Dataset, pd.DataFrame]: The concatenated dataset.
    """
    if all(isinstance(ds, xr.Dataset) for ds in datasets):
        combined_ds = xr.concat(datasets, dim=dim)
    elif all(isinstance(ds, pd.DataFrame) for ds in datasets):
        combined_ds = pd.concat(datasets, ignore_index=True)
    else:
        raise ValueError("[ERROR] Mixed types of datasets cannot be concatenated together.")
    
    return combined_ds

def save_dataset(dataset: Union[xr.Dataset, pd.DataFrame], save_dir: str, filename: str = 'combined_dataset.nc') -> None:
    """
    Saves the combined dataset to the specified directory.

    Parameters:
    dataset (Union[xr.Dataset, pd.DataFrame]): The combined dataset to save.
    save_dir (str): The directory path where the combined dataset should be saved.
    filename (str): The name of the file to save the dataset as. Default is 'combined_dataset.nc'.

    Usage:
    save_dataset(combined_ds, 'data/combined', 'combined_dataset.nc')
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, filename)
    if isinstance(dataset, xr.Dataset):
        dataset.to_netcdf(file_path)
    elif isinstance(dataset, pd.DataFrame):
        dataset.to_csv(file_path, index=False)
    else:
        raise ValueError("[ERROR] Unsupported dataset type for saving.")
    
    print(f"[INFO] Dataset saved to {file_path}")

def get_combined_dataset(data_root: str, dim: str = 'time', save_dir: str = '', file_type: str = 'nc') -> Union[xr.Dataset, pd.DataFrame]:
    """
    Loads datasets from files, concatenates them along the specified dimension, 
    and optionally saves the combined dataset to a specified directory.

    Parameters:
    data_root (str): The root directory path where the data files are stored.
    save_dir (str, optional): The directory path where the combined dataset should be saved. 
                              If empty, the dataset is not saved. Default is ''.
    dim (str, optional): The dimension along which to concatenate the datasets. Default is 'time'.
    file_type (str, optional): The type of files to load ('nc' for NetCDF, 'csv' for CSV). Default is 'nc'.

    Usage:
    combined_ds = get_combined_dataset('data/yearly', save_dir='data/combined', file_type='nc')

    Returns:
    Union[xr.Dataset, pd.DataFrame]: The combined dataset concatenated along the specified dimension.
    """
    data_paths = get_file_paths(data_root)
    datasets = load_datasets(data_paths)
    
    if not datasets:
        raise ValueError("[ERROR] No datasets were loaded. Please check your data paths.")
    
    combined_ds = concatenate_datasets(datasets, dim)
    
    if save_dir:
        if file_type == 'nc':
            save_dataset(combined_ds, save_dir, filename='combined_dataset.nc')
        elif file_type == 'csv':
            save_dataset(combined_ds, save_dir, filename='combined_dataset.csv')
        else:
            raise ValueError("[ERROR] Unsupported file type for saving.")
    
    return combined_ds