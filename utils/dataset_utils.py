import xarray as xr
import os
from utils.file_utils import get_file_paths

def load_datasets(data_paths: list) -> list:
    """
    Loads datasets from yearly files.

    Parameters:
    data_paths (list): A list of file paths to the yearly data files.

    Usage:
    datasets = load_datasets(['data/2001.nc', 'data/2002.nc'])

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

def concatenate_datasets(datasets: list, dim: str = 'time') -> xr.Dataset:
    """
    Concatenates a list of datasets along the specified dimension.

    Parameters:
    datasets (list): A list of xarray datasets to concatenate.
    dim (str): The dimension along which to concatenate the datasets.

    Usage:
    combined_ds = concatenate_datasets(datasets, dim='time')

    Returns:
    xr.Dataset: The concatenated dataset.
    """
    combined_ds = xr.concat(datasets, dim=dim)
    return combined_ds

def save_dataset(dataset: xr.Dataset, save_dir: str, filename: str = 'combined_dataset.nc'):
    """
    Saves the combined dataset to the specified directory.

    Parameters:
    dataset (xr.Dataset): The combined dataset to save.
    save_dir (str): The directory path where the combined dataset should be saved.
    filename (str): The name of the file to save the dataset as. Default is 'combined_dataset.nc'.

    Usage:
    save_dataset(combined_ds, 'data/combined', 'combined_dataset.nc')
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, filename)
    dataset.to_netcdf(file_path)
    print(f"[INFO] Dataset saved to {file_path}")

def get_combined_dataset(data_root: str, dim: str = 'time', save_dir: str = '') -> xr.Dataset:
    """
    Loads datasets from files, concatenates them along the time dimension, 
    and optionally saves the combined dataset to a specified directory.

    Parameters:
    data_root (str): The root directory path where the yearly data files are stored.
    save_dir (str, optional): The directory path where the combined dataset should be saved. 
                              If empty, the dataset is not saved. Default is ''.
    dim (str, optional): The dimension along which to concatenate the datasets. Default is 'time'.

    Usage:
    combined_ds = get_combined_dataset('data/yearly', save_dir='data/combined')

    Returns:
    xr.Dataset: The combined dataset concatenated along the time dimension.
    """
    data_paths = get_file_paths(data_root)
    datasets = load_datasets(data_paths)
    datasets = concatenate_datasets(datasets, dim)
    
    if save_dir:
        save_dataset(datasets, save_dir)

    return datasets