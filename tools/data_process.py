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
    print(f"[INFO] Creating DataLoader for {'training' if train else 'validation'}...")
    return dataset.to_dataloader(train=train, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)

def data_pipeline(data_root: str, data_config: dict, time_series_config: dict, batch_size: int = 16, num_workers: int = 4, mode: str = 'train', dataloading: bool = True) -> tuple:
    """
    Execute the data pipeline by loading, preprocessing (save preprocessed data if requested), creating datasets, and DataLoaders.

    Parameters:
    data_root (str): The directory pattern to search for files (e.g., 'data/*.nc').
    data_config (dict): Dictionary containing data configuration parameters.
    time_series_config (dict): Dictionary containing time series configuration parameters.
    batch_size (int): The batch size for DataLoader. Default is `16`.
    num_workers (int): The number of workers for DataLoader. Default is `4`.
    dataloading (bool): Whether to create DataLoaders. Default is `True`. Else return TimeSeriesDataSets.

    Returns:
    tuple: A tuple containing (training DataLoader, validation DataLoader) or (None, evaluation Dataloader).

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
    target_vars = time_series_config['target_vars']
    data_source = data_config['data_source']
    latitude_range = data_config['latitude_range']
    longtitude_range = data_config['longtitude_range']
    time_range = data_config['time_range']
    time_column = data_config['time_column']
    calendar_cycle = data_config['calendar_cycle']
    save_dir = data_config['save_dir']

    # Load the data
    ds = get_combined_dataset(data_root)
    df = convert_to_dataframe(ds, variables=target_vars)
    print("[INFO] Data loaded successfully.")
    
    # Preprocess the data
    if data_source == data_source:
        df = preprocess_cds_df(df, latitude_range, longtitude_range, time_range, calendar_cycle, time_column)

        if save_dir:
            save_to_csv(df, save_dir)

        training_dataset, validation_dataset = create_cds_time_series_datasets(df, time_series_config=time_series_config, mode=mode)
        
        if not dataloading:
            return training_dataset, validation_dataset
        elif mode == 'train':
            # Create DataLoaders
            train_dataloader = dataloader(training_dataset, train=True, batch_size=batch_size, num_workers=num_workers)
            val_dataloader = dataloader(validation_dataset, train=False, batch_size=batch_size, num_workers=num_workers)
            return train_dataloader, val_dataloader
        elif mode == 'eval':
            validation_dataset = dataloader(validation_dataset, train=False, batch_size=batch_size, num_workers=num_workers)
            return None, validation_dataset
        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose either 'train' or 'eval'.")
    else:
        raise ValueError(f"[INFO] Data source {data_source} is not supported.")