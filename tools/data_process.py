import pandas as pd
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from utils.dataset_utils import get_combined_dataset
from utils.dataframe_utils import convert_to_dataframe, save_to_csv
from datasets.cds.data_handling import preprocess_cds_df, create_cds_time_series_dataset

def dataloader(dataset: TimeSeriesDataSet, train: bool, batch_size: int, num_workers: int) -> pd.DataFrame:
    """
    Create a DataLoader from a TimeSeriesDataSet.

    Parameters:
    dataset (TimeSeriesDataSet): The TimeSeriesDataSet to convert into a DataLoader.
    train (bool): Whether the DataLoader is for training or validation.
    batch_size (int): The batch size for the DataLoader.
    num_workers (int): The number of workers for the DataLoader.

    Returns:
    DataLoader: A PyTorch DataLoader for the given TimeSeriesDataSet.
    """
    return dataset.to_dataloader(train=train, batch_size=batch_size, num_workers=num_workers)

def data_pipeline(data_root: str, data_source: str = 'cds', target_vars: list = [], time_column: str = 'time', max_encoder_length=365, max_prediction_length=365, min_prediction_length=1, save_dir: str = '') -> TimeSeriesDataSet:
    """
    Execute the data pipeline by loading, preprocessing, and saving the data.

    Parameters:
    data_root (str): The directory pattern to search for files (e.g., 'data/*.nc').
    data_source (str): The source of the data. Default is `cds`.
    target_vars (list): The list of target variables to include in the DataFrame. Default is empty list.
    time_column (str): The name of the time column. Default is `time`.
    save_dir (str): The directory to save the preprocessed data as `.csv` . Default is empty string.
    max_encoder_length (int): The maximum length of the encoder. Default is `365`.
    max_prediction_length (int): The maximum length of the prediction. Default is `365`.
    min_prediction_length (int): The minimum length of the prediction. Default is `1`.

    Returns:
    TimeSeriesDataSet: The created TimeSeriesDataSet.

    Usage:
    ts_dataset = data_pipeline(data_root='data/samples/*.nc', target_vars=['tcc', 'hcc', 'mcc', 'lcc', 'tciw', 'tclw'], save_dir='preprocessed_data.csv')
    """
    ts_dataset = None

    # Load the data
    ds = get_combined_dataset(data_root)
    df = convert_to_dataframe(ds, variables=target_vars)
    print("[INFO] Data loaded successfully.")
    
    # Preprocess the data
    if data_source == 'cds':
        df = preprocess_cds_df(df, time_column)
        ts_dataset = create_cds_time_series_dataset(df, max_encoder_length, max_prediction_length, target_vars, min_prediction_length)
    else:
        raise ValueError(f"[INFO] Data source {data_source} is not supported.")
    
    # Save the preprocessed data
    if save_dir:
        save_to_csv(df, save_dir)

    return ts_dataset