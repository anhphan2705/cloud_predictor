import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
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

def plot_missing_data(df: pd.DataFrame, save: bool = False) -> None:
    """
    Plots the missing data in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to plot missing data.
    save (bool): Whether to save the plot. Default is False.

    Usage:
    plot_missing_data(df)
    """
    missing_info = df.isna().sum().sort_values(ascending=False).to_frame().reset_index()
    missing_info.columns = ["Field Name", "NaN Counts"]
    plt.figure(figsize=(23, 6))
    plt.bar(missing_info["Field Name"], missing_info["NaN Counts"])
    plt.xticks(rotation=90)
    plt.xlabel("Columns")
    plt.ylabel("NaN counts")
    plt.title("Missing Values by Column")
    plt.show()
    if save:
        plt.savefig(os.path.join('missing_data_plot.png'))

def plot_target(df: pd.DataFrame, time_column: str, targets: list, group_by: list = None, save: bool = False, combine_groups: bool = True) -> None:
    """
    Plot the time series data for specified target columns, optionally grouped by specified columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    time_column (str): The name of the time column in the DataFrame.
    targets (list): A list of target columns to plot.
    group_by (list, optional): A list of columns to group by (e.g., ['latitude', 'longitude']). Default is None.
    save (bool): Whether to save the plot. Default is False.
    combine_groups (bool): Whether to combine all groups of the same target into the same graph. Default is True.

    Example Usage:
    plot_target(cds_df, 'time', ['tcc'], group_by=['latitude', 'longitude'], combine_groups=True)
    """
    # Ensure targets is a list
    if isinstance(targets, str):
        targets = [targets]

    # Set the time column as the index
    df = df.set_index(time_column)

    def save_plot(title, filename):
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel(target)
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(os.path.join(filename))
        plt.show()

    if group_by:
        grouped = df.groupby(group_by)
        for target in targets:
            if combine_groups:
                plt.figure(figsize=(23, 6))
                for group_values, group in grouped:
                    plt.plot(group.index, group[target], label=f"{', '.join([f'{col} {val}' for col, val in zip(group_by, group_values)])}")
                save_plot(f'{target} over Time for all groups', f'{target}_all_groups_plot.png')
            else:
                for group_values, group in grouped:
                    plt.figure(figsize=(23, 6))
                    plt.plot(group.index, group[target], label=f"{', '.join([f'{col} {val}' for col, val in zip(group_by, group_values)])}", color='orange')
                    save_plot(f'{target} over Time for {", ".join([f"{col} {val}" for col, val in zip(group_by, group_values)])}', f'{target}_in_{group_values}_plot.png')
    else:
        for target in targets:
            plt.figure(figsize=(23, 6))
            df[target].plot(title=f'{target} over Time', color='orange')
            save_plot(f'{target} over Time', f'{target}_plot.png')

def plot_target_by_year(df: pd.DataFrame, time_column: str, target_column: str, same_graph: bool = True, save: bool = False) -> None:
    """
    Plot the target column by year.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    time_column (str): The name of the time column in the DataFrame.
    target_column (str): The name of the target column to plot.
    same_graph (bool): If True, plot all years on the same graph. If False, plot each year on a separate graph. Default is True.
    save (bool): Whether to save the plot(s). Default is False.

    Example Usage:
    plot_target_by_year(cds_df, 'time', 'tcc', same_graph=True)
    """
    # Ensure the time column is a datetime type
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Extract the year from the time column
    df['year'] = df[time_column].dt.year

    # Group the data by year
    grouped = df.groupby('year')

    if same_graph:
        # Plot all years on the same graph
        plt.figure(figsize=(23, 8))
        for year, group in grouped:
            plt.plot(group[time_column], group[target_column], label=f'Year {year}')
        plt.title(f'{target_column} over Time by Year')
        plt.xlabel('Time')
        plt.ylabel(target_column)
        plt.grid(True)
        plt.legend()
        plt.show()
        if save:
            plt.savefig(f'{target_column}_by_year_same_graph.png')
    else:
        # Plot each year on a separate graph
        for year, group in grouped:
            plt.figure(figsize=(23, 8))
            plt.plot(group[time_column], group[target_column], label=f'Year {year}', color='orange')
            plt.title(f'{target_column} over Time in Year {year}')
            plt.xlabel('Time')
            plt.ylabel(target_column)
            plt.grid(True)
            plt.legend()
            plt.show()
            if save:
                plt.savefig(f'{target_column}_in_year_{year}.png')


def plot_target_comparison(df: pd.DataFrame, time_column: str, target_columns: list, period: str, overlap: bool = True, save: bool = False) -> None:
    """
    Plot the target columns, allowing comparison over a selected time period (week, month, or year) on the same graph.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    time_column (str): The name of the time column in the DataFrame.
    target_columns (list): A list of target columns to plot.
    period (str): The period to compare ('week', 'month', 'year').
    overlap (bool): Whether to overlap the periods on the same graph. Default is True.
    save (bool): Whether to save the plot(s). Default is False.

    Example Usage:
    plot_target_comparison(cds_df, 'time', ['tcc'], 'month')
    plot_target_comparison(cds_df, 'time', ['tcc', 'temperature'], 'month', overlap=True)
    """
    # Ensure target_columns is a list
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    # Ensure the time column is a datetime type
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Extract the relevant period from the time column
    if period == 'week':
        df['period'] = df[time_column].dt.isocalendar().week
    elif period == 'month':
        df['period'] = df[time_column].dt.month
    elif period == 'year':
        df['period'] = df[time_column].dt.year
    else:
        raise ValueError("Invalid period. Choose from 'week', 'month', or 'year'.")

    # Extract the year to differentiate periods across different years
    df['year'] = df[time_column].dt.year

    # Group the data by the selected period and year
    grouped = df.groupby(['year', 'period'])

    # Plot each target on a different graph if multiple targets are specified
    for target in target_columns:
        plt.figure(figsize=(23, 8))
        if overlap:
            for (year, period), group in grouped:
                plt.plot(group[time_column] - pd.to_datetime(f"{group[time_column].dt.year.iloc[0]}-01-01"), group[target], label=f'Year {year}')
            plt.title(f'{target} Comparison by {period}')
            plt.xlabel('Time since start of the year')
        else:
            for (year, period), group in grouped:
                plt.plot(group[time_column], group[target], label=f'Year {year} Period {period}')
            plt.title(f'{target} over Time by {period}')
            plt.xlabel('Time')
        plt.ylabel(target)
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(f'{target}_by_{period}.png')
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

def plot_cyclical_features(df: pd.DataFrame, time_column: str, feature_prefixes: list, save: bool = False) -> None:
    """
    Plot the time series data along with the added cyclical features to check alignment.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the time series data and cyclical features.
    time_column (str): The name of the time column in the DataFrame.
    feature_prefixes (list): A list of feature prefixes to plot (e.g., ['day', 'weekday', 'week', 'month']).
    save (bool): Whether to save the plot. Default is False.

    Example Usage:
    plot_cyclical_features(cds_df, 'time', ['day', 'weekday', 'week', 'month'])
    """
    # Set the time column as the index
    df = df.set_index(time_column)

    # Create subplots
    num_features = len(feature_prefixes)
    fig, axes = plt.subplots(num_features, 1, figsize=(23, 6 * (num_features)))

    if num_features == 1:
        axes = [axes]

    print(f"[INFO] PLotting cyclical features: {feature_prefixes}")
    
    # Plot the cyclical features
    for i, prefix in enumerate(feature_prefixes):
        sin_col = f"{prefix}_sin"
        cos_col = f"{prefix}_cos"
        print(f"[DEBUG] Plotting {prefix} feature on subplot index {i+1}")

        if sin_col in df.columns and cos_col in df.columns:
            df[sin_col].plot(ax=axes[i], label=f"{prefix}_sin", color='blue')
            df[cos_col].plot(ax=axes[i], label=f"{prefix}_cos", color='orange')
            axes[i].set_title(f"Cyclical Features for {prefix}")
            axes[i].legend()
            axes[i].set_ylabel(f"{prefix}_value")
        else:
            print(f"Columns {sin_col} or {cos_col} not found in DataFrame.")

    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig(os.path.join(f'cyclical features_plot.png'))


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
    
    for idx in range(1):
        fig, ax = plt.subplots(figsize=(23, 6))
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
        if show:
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

    y_pred = prediction.output if type(prediction.output) == torch.Tensor else prediction.output.prediction

    # Calculate predictions vs actuals
    predictions_vs_actuals = model.calculate_prediction_actual_by_variable(prediction.x, y_pred)

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