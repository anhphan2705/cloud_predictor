import xarray as xr
import pandas as pd

from utils import get_combined_dataset, convert_to_dataframe, save_to_csv

def preprocess_data(df: pd.DataFrame, target_vars: str) -> pd.DataFrame:    
    return None

def main(data_paths: list, target_vars: list, save_dir: str = ''):
    # Load the data
    ds = get_combined_dataset(data_paths)
    
    # Convert the data to a DataFrame
    df = convert_to_dataframe(ds, variables=target_vars)

    # Preprocess the data
    preprocessed_data = preprocess_data(df, target_vars)
    
    # Save the preprocessed data
    if save_dir:
        save_to_csv(preprocessed_data, save_dir)