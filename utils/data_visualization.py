import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def df_visualizer(df: pd.DataFrame):
    """
    Prints the first few rows of a DataFrame for visualization purposes.

    Parameters:
    df (pd.DataFrame): The DataFrame to visualize.

    Usage:
    df_visualizer(df)
    """
    print(f"[INFO] Visualizing dataframe...\n{df.head()}")

def get_col_count(df: pd.DataFrame) -> pd.Series:
    """
    Prints and returns the count of non-missing values in each column of a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame for which to count non-missing values.

    Usage:
    col_counts = get_col_count(df)

    Returns:
    pd.Series: A Series containing the count of non-missing values for each column.
    """
    print(f"[INFO] Count of non-missing values in each column:\n{df.count()}")
    return df.count()

def save_to_csv(df: pd.DataFrame, save_dir: str):
    """
    Saves a DataFrame to a CSV file.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    save_dir (str): The directory path where the CSV file should be saved.

    Usage:
    save_to_csv(df, 'output/data.csv')
    """
    df.to_csv(f'{save_dir}', index=False)
    print(f"[INFO] Saved DataFrame to {save_dir}")

def plot_missing_data(df: pd.DataFrame):
    """
    Plots the missing data in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to plot missing data.

    Usage:
    plot_missing_data(df)
    """
    missing_info = df.isna().sum().sort_values(ascending=False).to_frame().reset_index()
    missing_info.columns = ["Field Name", "NaN Counts"]
    plt.figure(figsize=(10, 6))
    plt.bar(missing_info["Field Name"], missing_info["NaN Counts"])
    plt.xticks(rotation=90)
    plt.xlabel("Columns")
    plt.ylabel("NaN counts")
    plt.title("Missing Values by Column")
    plt.show()

def plot_time_series(df: pd.DataFrame, latitude: float, longitude: float, variables: list, time_column: str = 'time'):
    """
    Plots the time series for specified variables at a specific latitude and longitude.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to plot.
    latitude (float): The latitude to filter the data.
    longitude (float): The longitude to filter the data.
    variables (list): A list of variables to plot.
    time_column (str): The name of the time column. Default is 'time'.

    Usage:
    plot_time_series(df, latitude=23.0, longitude=102.0, variables=['tciw'])
    """
    # Filter the DataFrame for the specific latitude and longitude
    filtered_df = df[(df['latitude'] == latitude) & (df['longitude'] == longitude)]

    # Plot the time series for the variables of interest
    plt.figure(figsize=(23, 5))
    for var in variables:
        sns.lineplot(data=filtered_df, x=time_column, y=var, label=var)

    plt.title(f'Time Series at Latitude {latitude} and Longitude {longitude}')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()
