import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from utils.dataset_utils import get_combined_dataset
from utils.dataframe_utils import convert_to_dataframe, save_to_csv
from datasets.cds.data_handling import preprocess_cds_df, create_cds_time_series_datasets

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

    Usage:
    train_loader = dataloader(training_dataset, train=True, batch_size=16, num_workers=4)
    val_loader = dataloader(validation_dataset, train=False, batch_size=16, num_workers=4)
    """
    return dataset.to_dataloader(train=train, batch_size=batch_size, num_workers=num_workers)

def data_pipeline(data_root: str, data_source: str = 'cds', target_vars: list = [], time_column: str = 'time', max_encoder_length: int = 365, max_prediction_length: int = 365, min_prediction_length: int = 1, batch_size: int = 16, num_workers: int = 4, save_dir: str = '') -> tuple:
    """
    Execute the data pipeline by loading, preprocessing (save preprocessed data if requested), creating datasets, and DataLoaders.

    Parameters:
    data_root (str): The directory pattern to search for files (e.g., 'data/*.nc').
    data_source (str): The source of the data. Default is `cds`.
    target_vars (list): The list of target variables to include in the DataFrame. Default is empty list.
    time_column (str): The name of the time column. Default is `time`.
    max_encoder_length (int): The maximum length of the encoder. Default is `365`.
    max_prediction_length (int): The maximum length of the prediction. Default is `365`.
    min_prediction_length (int): The minimum length of the prediction. Default is `1`.
    batch_size (int): The batch size for DataLoader. Default is `16`.
    num_workers (int): The number of workers for DataLoader. Default is `4`.
    save_dir (str): The directory to save the preprocessed data as `.csv`. Default is empty string.

    Returns:
    tuple: A tuple containing the training DataLoader and validation DataLoader.

    Usage:
    train_dataloader, val_dataloader = data_pipeline(
        data_root='data/samples/*.nc', 
        target_vars=['tcc', 'hcc', 'mcc', 'lcc', 'tciw', 'tclw'], 
        time_column='time', 
        max_encoder_length=365, 
        max_prediction_length=365, 
        min_prediction_length=1, 
        batch_size=16, 
        num_workers=4, 
        save_dir='preprocessed_data.csv'
    )
    """
    # Load the data
    ds = get_combined_dataset(data_root)
    df = convert_to_dataframe(ds, variables=target_vars)
    print("[INFO] Data loaded successfully.")
    
    # Preprocess the data
    if data_source == 'cds':
        df = preprocess_cds_df(df, time_column)

        if save_dir:
            save_to_csv(df, save_dir)

        training_dataset, validation_dataset = create_cds_time_series_datasets(df, max_encoder_length, max_prediction_length, target_vars, min_prediction_length)
    else:
        raise ValueError(f"[INFO] Data source {data_source} is not supported.")
    
    # Create DataLoaders
    train_dataloader = dataloader(training_dataset, train=True, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = dataloader(validation_dataset, train=False, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader
