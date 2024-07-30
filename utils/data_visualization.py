import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_forecasting import TemporalFusionTransformer

def df_visualizer(df: pd.DataFrame) -> None:
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

def plot_missing_data(df: pd.DataFrame) -> None:
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

def plot_time_series(df: pd.DataFrame, latitude: float, longitude: float, variables: list, time_column: str = 'time') -> None:
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

def convert_to_time_idx(years: int = 0, days: int = 0, hours: int = 0, step: int = 2) -> int:
    """
    Convert years, days, and hours to time index counts.

    Parameters:
    years (int): Number of years to convert. Default is 0.
    days (int): Number of days to convert. Default is 0.
    hours (int): Number of hours to convert. Default is 0.
    step (int): Time step in hours. Default is 2 (for 2-hour intervals).

    Returns:
    int: Equivalent time index count.

    Example Usage:
    time_idx = convert_to_time_idx(years=1, days=0, hours=0, step=2)
    """
    total_days = years * 365 + days  # Assuming no leap years for simplicity
    total_hours = total_days * 24 + hours
    return total_hours // step

def plot_predictions(predictions: dict, model: TemporalFusionTransformer, save_dir: str, show_future_observed: bool = False, add_loss_to_title:bool = False, show: bool = True, title: str = 'Model Predictions vs Actual Data') -> None:
    """
    Plot the actual data, trained model predictions, and baseline model predictions.

    Parameters:
    predictions (dict): A dictionary containing the actual data, trained model predictions, and baseline model predictions.
    model (TemporalFusionTransformer): The trained Temporal Fusion Transformer model.
    save_dir (str): Directory to save the plot.
    show_future_observed (bool): Whether to show future observed data. Default is False.
    add_loss_to_title (bool): Whether to add loss to the title. Default is False.
    show (bool): Whether to show the plot. Default is True.
    title (str): Title of the plot.
    """
    print("[INFO] Plotting result...")

    print(f"[DEBUG] Prediction keys: {predictions.keys()}")
    
    for idx in range(1):
        fig, ax = plt.subplots(figsize=(23, 5))
        model.plot_prediction(
            predictions.x,
            predictions.output,
            idx=idx,
            show_future_observed=show_future_observed,
            add_loss_to_title=add_loss_to_title,
            ax=ax,
        )
        if not add_loss_to_title:
            plt.title(title)
        plt.savefig(os.path.join(save_dir, f'model_predictions_{idx}.png'))
        plt.show()

    print(f"[INFO] Plots saved to {save_dir}")

def generate_exclude_features(lags: dict) -> set:
    """
    Generate a set of exclude features based on the provided lags.

    Parameters:
    lags (dict): Dictionary containing variable names as keys and lists of lag values as values.

    Returns:
    set: A set of exclude feature names formatted as 'variable_lagged_by_lag'.
    """
    exclude_features = set()
    for variable, lag_list in lags.items():
        for lag in lag_list:
            exclude_features.add(f'{variable}_lagged_by_{lag}')
    return exclude_features

def interpret_model_predictions(model: TemporalFusionTransformer, prediction: dict, save_dir: str, model_name: str, lags: dict, show: bool = False) -> None:
    """
    Interpret model predictions by plotting the actual values against predicted values for each feature.
    
    This function uses the Temporal Fusion Transformer model to make predictions on the validation dataloader,
    then calculates the prediction vs actual values for each variable and plots them. The resulting plots are saved
    in the specified directory.

    Parameters:
    model (TemporalFusionTransformer): The trained Temporal Fusion Transformer model.
    prediction (dict, optional): A dictionary containing the model predictions. If not provided, the model will make predictions.
    save_dir (str): The directory to save interpretation plots.
    model_name (str): The name of the model for naming the plot files.
    lags (dict): Dictionary containing variable names as keys and lists of lag values as values.
    show (bool, optional): If True, displays the plots. Default is False.

    Usage:
    interpret_model_predictions(trained_model, val_dataloader, './interpretation_plots', model_name="tft", show=True)
    """
    print("[INFO] Interpreting model predictions...")

    print(f"[DEBUG] Prediction keys: {prediction.keys()}")

    # Calculate predictions vs actuals
    predictions_vs_actuals = model.calculate_prediction_actual_by_variable(prediction.x, prediction.output)

    # Generate exclude features based on lags
    exclude_features = generate_exclude_features(lags)
    print("[DEBUG] Excluded plots: ", exclude_features)

    # Get feature names
    features = list(set(predictions_vs_actuals['support'].keys()) - exclude_features)

    # Plot and save interpretation for each feature
    for feature in features:
        model.plot_prediction_actual_by_variable(predictions_vs_actuals, name=feature)
        plt.savefig(os.path.join(save_dir, f'{model_name}_{feature}_interpretation.png'))
        if show:
            plt.show()

    print(f"[INFO] Interpretation plots saved to {save_dir}")