import glob
import xarray as xr
import pandas as pd

def get_files(dir):
    files = glob.glob(dir)
    print(f"[INFO] Found {len(files)} files in {dir}")
    return files

def df_visualizer(df):
    # Visualize data in the DataFrame
    print(f"[INFO] Visualizing dataframe...\n{df.head()}")
 
def get_col_count(df):
   # Print the count of non-missing values in each column
    print(f"[INFO] Count of non-missing values in each column:\n{df.count()}")
    return df.count()
    
def get_dataset(data_files, years, save_dir=''):
    # Load datasets from yearly files and return a list of datasets
    # Save the combined dataset if save dir initialized
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
    # Convert the combined dataset to a DataFrame
    df = datasets[variables].to_dataframe().reset_index()
    # Process the DataFrame to handle missing values, etc.
    # df = df.dropna()
    # Convert timestamps to a datetime format
    df['time'] = pd.to_datetime(df['time'])
    print(f"[INFO] Converted dataset to DataFrame with shape: {df.shape}")
    return df

def save_to_csv(df, save_dir):
    # Save the DataFrame to a CSV file
    df.to_csv(f'{save_dir}', index=False)
    print(f"[INFO] Saved DataFrame to {save_dir}")