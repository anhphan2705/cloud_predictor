import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from utils.dataset_utils import get_combined_dataset
from utils.dataframe_utils import convert_to_dataframe
from datasets.cds.data_handling import preprocess_cds_df, create_cds_time_series_datasets
from tools.data_process import dataloader
import matplotlib.pyplot as plt

def evaluate_model(model_path: str, data_root: str, target_vars: list, time_column: str, max_encoder_length: int, max_prediction_length: int, min_prediction_length: int, batch_size: int, num_workers: int, new_data_root: str = None):
    """
    Evaluate the trained Temporal Fusion Transformer model.

    Parameters:
    model_path (str): The path to the trained model checkpoint.
    data_root (str): The directory pattern to search for files (e.g., 'data/*.nc').
    target_vars (list): The list of target variables to include in the DataFrame.
    time_column (str): The name of the time column.
    max_encoder_length (int): The maximum length of the encoder.
    max_prediction_length (int): The maximum length of the prediction.
    min_prediction_length (int): The minimum length of the prediction.
    batch_size (int): The batch size for DataLoader.
    num_workers (int): The number of workers for DataLoader.
    new_data_root (str): The directory pattern to search for new data files for prediction. Default is None.

    Example Usage:
    evaluate_model('model/checkpoints/best_model.ckpt', 'data/samples/*.nc', ['tcc', 'hcc'], 'time', 365, 365, 1, 16, 4)
    """
    if new_data_root:
        # Load the new data
        ds = get_combined_dataset(new_data_root)
        df = convert_to_dataframe(ds, variables=target_vars)
        print("[INFO] New data loaded successfully.")
    else:
        # Load the data
        ds = get_combined_dataset(data_root)
        df = convert_to_dataframe(ds, variables=target_vars)
        print("[INFO] Data loaded successfully.")

    # Preprocess the data
    df = preprocess_cds_df(df, time_column)

    # Create evaluation dataset
    eval_dataset = create_cds_time_series_datasets(df, max_encoder_length, max_prediction_length, target_vars, min_prediction_length, mode='eval')

    # Create DataLoader
    eval_dataloader = dataloader(eval_dataset, train=False, batch_size=batch_size, num_workers=num_workers)

    # Load the trained model
    model = TemporalFusionTransformer.load_from_checkpoint(model_path)

    # Make predictions
    predictions, index = model.predict(eval_dataloader, return_index=True)
    
    # Plot predictions
    for target in target_vars:
        plt.figure(figsize=(15, 5))
        plt.plot(index, predictions[:, target_vars.index(target)], label=f'Predicted {target}')
        plt.title(f'Predicted {target} over Time')
        plt.xlabel('Time')
        plt.ylabel(f'{target}')
        plt.legend()
        plt.grid(True)
        plt.show()