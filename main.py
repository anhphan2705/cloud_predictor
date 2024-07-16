import argparse
from tools.data_process import data_pipeline
from tools.train import train_pipeline

def main(data_source: str,
         data_root: str,
         target_vars: list,
         time_column: str,
         max_encoder_length: int,
         max_prediction_length: int,
         min_prediction_length: int,
         batch_size: int,
         num_workers: int,
         param_tuning_trial_count: int,
         save_dir: str):
    """
    Main function to load data, preprocess it, create datasets, and train the model.

    Parameters:
    data_source (str): The data source.
    data_root (str): The root directory pattern to search for files (e.g., 'data/*.nc').
    target_vars (list): The list of target variables.
    time_column (str): The name of the time column.
    max_encoder_length (int): The maximum length of the encoder.
    max_prediction_length (int): The maximum length of the prediction.
    min_prediction_length (int): The minimum length of the prediction.
    batch_size (int): The batch size for DataLoader.
    num_workers (int): The number of workers for DataLoader.
    param_tuning_trial_count (int): The number of trials for hyperparameter tuning.
    save_dir (str): The directory to save the preprocessed data as `.csv`.

    Example Usage:
    main('cds', 'data/samples/*.nc', ['tcc', 'hcc', 'mcc', 'lcc', 'tciw', 'tclw'], 'time', 365, 365, 24, 16, 4, 100, 'preprocessed_data.csv')
    """
    # Load the data and create DataLoaders
    train_dataloader, val_dataloader = data_pipeline(
        data_root=data_root,
        data_source=data_source,
        target_vars=target_vars,
        time_column=time_column,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_prediction_length=min_prediction_length,
        batch_size=batch_size,
        num_workers=num_workers,
        save_dir=save_dir
    )

    # Train the model
    train_pipeline(train_dataloader, val_dataloader, param_tuning_trial_count=param_tuning_trial_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Temporal Fusion Transformer for weather prediction.")
    parser.add_argument("--data_source", type=str, default="cds", help="The data source.")
    parser.add_argument("--data_root", type=str, default="data/samples/*.nc", help="The root directory pattern to search for files.")
    parser.add_argument("--target_vars", nargs='+', default=["tcc", "hcc", "mcc", "lcc", "tciw", "tclw"], help="The list of target variables.")
    parser.add_argument("--time_column", type=str, default="time", help="The name of the time column.")
    parser.add_argument("--max_encoder_length", type=int, default=365, help="The maximum length of the encoder.")
    parser.add_argument("--max_prediction_length", type=int, default=365, help="The maximum length of the prediction.")
    parser.add_argument("--min_prediction_length", type=int, default=24, help="The minimum length of the prediction.")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers for DataLoader.")
    parser.add_argument("--param_tuning_trial_count", type=int, default=100, help="The number of trials for hyperparameter tuning.")
    parser.add_argument("--save_dir", type=str, default="", help="The directory to save the preprocessed data as `.csv`.")
    
    args = parser.parse_args()
    
    main(data_source=args.data_source,
         data_root=args.data_root,
         target_vars=args.target_vars,
         time_column=args.time_column,
         max_encoder_length=args.max_encoder_length,
         max_prediction_length=args.max_prediction_length,
         min_prediction_length=args.min_prediction_length,
         batch_size=args.batch_size,
         num_workers=args.num_workers,
         param_tuning_trial_count=args.param_tuning_trial_count,
         save_dir=args.save_dir)
